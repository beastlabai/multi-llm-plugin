#!/usr/bin/env python3
"""Seed `expected_argv.yaml` from real call logs, keyed provider x profile x transport.

Run on the HOST after a container run, against whatever bundles are on disk:

    uv run --project skills/multi-llm -- \\
        python skills/multi-llm/tests/container/seed_expected_argv.py

**The profile is classified, not guessed.** A call log records argv, not intent,
and three of codex's five shapes differ only in a policy value — `--sandbox
read-only` is `agent-readonly`, `--sandbox workspace-write` is `generate` or
`agent-write`. So each observed shape is matched against what the adapter's own
`build_command()` / `build_agent_command()` produce for that provider, and takes
the profile whose normalized form it equals. Anything that matches none is
labelled `probe` — the `--version` and `codex login status` invocations, which are
real dispatches this table must still explain but which no profile emits.

Seeding is deliberately a separate step from asserting. The assertion
(`tests/test_container_argv.py`) never writes, so a drifted argv fails a test
rather than quietly rewriting the table it is checked against — the same rule the
goldens follow.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILL_ROOT = HERE.parent.parent
sys.path.insert(0, str(SKILL_ROOT))
sys.path.insert(0, str(SKILL_ROOT / "tests" / "golden"))

from normalize_argv import normalize_argv, PLACEHOLDER_PROMPT  # noqa: E402

# The SAME collapse the wrapper applies at write time. A reference shape built
# without it carries the raw prompt in its tail while every real record carries
# `<prompt:sha256:...>`, so nothing ever matches and every argv-transport row is
# silently demoted to `probe`. Two normalizers that disagree is precisely the
# seam this whole comparison exists to close, so it is imported rather than
# reimplemented.
sys.path.insert(0, str(HERE))
from wrapper import collapse_prompt, inject_argv  # noqa: E402


def adapter_shapes() -> dict[tuple[str, str, str], dict]:
    """What each adapter emits, per profile and transport, normalized the same way."""
    import yaml

    from utils.provider_registry import get_provider, load_base_config
    from utils.providers.base import (
        PROFILE_AGENT_READONLY,
        PROFILE_AGENT_WRITE,
        PROFILE_GENERATE,
    )

    rows = {
        row["cli"]: row
        for row in (yaml.safe_load((HERE / "redirect.yaml").read_text("utf-8")))["clis"]
        if row.get("lane") == "provider"
    }
    config = load_base_config()
    # The reference prompt has to be REPRESENTATIVE, not filler, on three counts.
    #
    # LENGTH: normalize_argv only collapses a token the wrapper already reduced,
    # and the wrapper's threshold is 200 chars — a short literal stays verbatim
    # and then matches nothing, demoting every argv-transport row to `probe`.
    #
    # CONTENT: at least one adapter derives FLAGS from the prompt's text.
    # `AiderProvider` scans it for absolute paths and passes each as `--read`,
    # because aider's mention auto-add never matches an absolute path.
    #
    # EXISTENCE: that scan is guarded by an existence check, so a path that does
    # not exist ON THE MACHINE RUNNING THIS SEEDER yields no `--read` at all —
    # and the reference then silently lacks a flag every real record carries.
    # The container's paths do not exist on the host, so the seeder makes its
    # own. Two of them, because aider emits one `--read` per path and the
    # normalized form of one (`<path>`) differs from several (`<path>+`); both
    # are real shapes the chain produces.
    import tempfile

    scratch = Path(tempfile.mkdtemp(prefix="mllm-argv-ref-"))
    one = scratch / "auth-feature.md"
    two = scratch / "second.md"
    one.write_text("# plan\n", encoding="utf-8")
    two.write_text("# more\n", encoding="utf-8")
    prompts = {
        "single-path": (f"Review the implementation plan at {one} and report "
                        f"findings. " + "P" * 400),
        "multi-path": (f"Review {one} together with {two} and report findings. "
                       + "P" * 400),
    }
    out: dict[tuple[str, str, str], dict] = {}
    for cli, row in rows.items():
        provider = get_provider(cli)
        # EXACTLY the spec test_container_*.py builds, not providers.yaml's first
        # model. grok has no `model_arg`, so falling back to base picked
        # `grok-4.5:high` — which makes the adapter add `--reasoning-effort high`
        # that no row ever emits, and the reference matches nothing.
        model = str(row.get("model_arg") or "stub-model")
        binary = row.get("binary", cli)
        for profile in (PROFILE_GENERATE, PROFILE_AGENT_READONLY, PROFILE_AGENT_WRITE):
          for variant, prompt in prompts.items():
            for transport in ("argv", "stdin"):
                try:
                    if profile == PROFILE_GENERATE:
                        argv, _stdin = provider.build_command_for_transport(
                            prompt, model, transport)
                    else:
                        argv, _stdin = provider.build_agent_command_for_transport(
                            prompt, model, profile, transport)
                except Exception:  # noqa: BLE001 - undeclared profile/transport
                    continue
                # The wrapper injects a redirect flag for any CLI whose only
                # override is argv (`muse exec --base-url`). That token is in
                # every real record, so a reference built without it matches
                # nothing — the rig's own injection is part of the dispatched
                # argv and the table must account for it.
                injected = inject_argv(list(argv[1:]), row, "{base_url}")
                shape = normalize_argv(
                    collapse_prompt([f"/opt/mllm/real/{binary}", *injected]))
                out[(cli, profile, transport, variant)] = shape
    return out


def observed(runs_root: Path) -> dict[tuple[str, str], list[dict]]:
    """Distinct normalized shapes per (cli, transport), with their counts."""
    seen: dict[tuple[str, str], dict[str, dict]] = {}
    for path in glob.glob(str(runs_root / "*" / "*" / "calls.jsonl")):
        for line in open(path, encoding="utf-8"):
            record = json.loads(line)
            if record.get("cli") == "uv":
                continue
            shape = normalize_argv(record["argv"])
            values = [v for v in shape["options"].values() if isinstance(v, str)]
            transport = ("argv" if (PLACEHOLDER_PROMPT in shape["tail"]
                                    or PLACEHOLDER_PROMPT in values) else "stdin")
            key = (record["cli"], transport)
            blob = json.dumps(shape, sort_keys=True)
            bucket = seen.setdefault(key, {})
            if blob in bucket:
                bucket[blob]["count"] += 1
            else:
                bucket[blob] = {"shape": shape, "count": 1}
    return {key: list(v.values()) for key, v in seen.items()}


def _equal(a: dict, b: dict) -> bool:
    return (a["binary"] == b["binary"] and a["subcommands"] == b["subcommands"]
            and a["options"] == b["options"] and a["tail"] == b["tail"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default=str(HERE / "runs"))
    parser.add_argument("--out", default=str(HERE / "expected_argv.yaml"))
    args = parser.parse_args(argv)

    shapes = adapter_shapes()
    entries = []
    for (cli, transport), variants in sorted(observed(Path(args.runs)).items()):
        for variant in variants:
            shape, count = variant["shape"], variant["count"]
            # ALL matching profiles, not the first. Some adapters render two
            # profiles identically — codex's `agent-write` IS its `build_command`,
            # so `generate/stdin` and `agent-write/stdin` are byte-identical — and
            # picking one arbitrarily would state something the argv cannot
            # support. Recording `generate|agent-write` says exactly what is
            # known: this argv is dispatched under either, and the table cannot
            # tell them apart because the product does not distinguish them here.
            matched = sorted({
                p for (c, p, t, _v), candidate in shapes.items()
                if c == cli and t == transport and _equal(shape, candidate)})
            profile = "|".join(matched) if matched else "probe"
            entries.append({
                "provider": cli, "profile": profile, "transport": transport,
                "binary": shape["binary"], "subcommands": shape["subcommands"],
                "options": shape["options"], "tail": shape["tail"],
                "prompt": transport, "min_records": 1, "_seen": count,
            })

    out_path = Path(args.out)
    header = out_path.read_text("utf-8").split("entries:")[0] \
        if out_path.exists() else "version: 1\n\n"
    lines = [header.rstrip("\n"), "", "entries:"]
    for e in sorted(entries, key=lambda x: (x["provider"], x["profile"],
                                            x["transport"], str(x["options"]))):
        lines.append(f"  - provider: {e['provider']}")
        lines.append(f"    profile: {e['profile']}")
        lines.append(f"    transport: {e['transport']}")
        lines.append(f"    binary: {e['binary']}")
        lines.append(f"    subcommands: {json.dumps(e['subcommands'])}")
        lines.append("    options:")
        for flag in sorted(e["options"]):
            lines.append(f"      {json.dumps(flag)}: {json.dumps(e['options'][flag])}")
        if not e["options"]:
            lines[-1] = "    options: {}"
        lines.append(f"    tail: {json.dumps(e['tail'])}")
        lines.append(f"    prompt: {e['prompt']}")
        lines.append(f"    min_records: {e['min_records']}")
        lines.append("")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")

    print(f"{len(entries)} entries -> {args.out}")
    for e in entries:
        print(f"  {e['provider']:12} {e['profile']:15} {e['transport']:6} "
              f"x{e['_seen']:<4} sub={e['subcommands']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
