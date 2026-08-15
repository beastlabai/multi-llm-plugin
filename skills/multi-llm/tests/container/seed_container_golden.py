#!/usr/bin/env python3
"""Seed `tests/golden/container/<cli>/bundle.json` from a green chain run (C8).

Run on the HOST after a provider-lane chain run, against the bundles on disk:

    uv run --project skills/multi-llm -- \\
        python skills/multi-llm/tests/container/seed_container_golden.py

The repo tree is never mounted into a container, so "the harness captured a
golden" is always a host-side write and always lands as a reviewable diff — which
is the whole point of doing it here rather than inside the image.

**Seeding is deliberately a separate step from asserting**, exactly as it is for
`expected_argv.yaml`. `tests/test_container_full_chain.py` only ever *reads* these
files: a drifted bundle fails a test instead of quietly rewriting the baseline it
is checked against. There is no `--update` flag on the assertion side, and adding
one would end the gate.

**Only a green row may become a baseline.** A row that hit its cap, exited
non-zero, ran fewer commands than it declared or did not record every chain phase
completed is refused by name. A capped CLI has no parseable output and a failed
row's artifacts are half a run; either one, frozen as "expected", would make every
later comparison agree with a broken run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SKILL_ROOT = HERE.parent.parent
sys.path.insert(0, str(SKILL_ROOT))
sys.path.insert(0, str(SKILL_ROOT / "tests" / "golden"))

import container_bundle  # noqa: E402

RUNS = HERE / "runs"
REDIRECT = HERE / "redirect.yaml"

#: Every phase `full --quick --yes` must have recorded completed for a chain row
#: to be worth freezing. The same list `test_container_full_chain.py` asserts, and
#: the reason a row that stopped early can never become "expected".
CHAIN_PHASES = (
    "review-plan", "apply-suggestions", "generate-tasks", "review-tasks",
    "apply-task-suggestions", "implement", "code-review", "apply-code-fixes",
)


def provider_clis() -> list[str]:
    import yaml

    data = yaml.safe_load(REDIRECT.read_text(encoding="utf-8"))
    return [row["cli"] for row in data["clis"] if row.get("lane") == "provider"]


def newest_bundle(cli: str, runs_root: Path) -> Path | None:
    """The most recent `runs/<utc>/chain-<cli>/` directory, or None."""
    row_id = container_bundle.row_id_for(cli)
    candidates = sorted(p for p in runs_root.glob(f"*/{row_id}") if p.is_dir())
    return candidates[-1] if candidates else None


def refusal(out_dir: Path) -> str | None:
    """Why this bundle must not become a baseline, or None when it may."""
    row = out_dir / "row-result.json"
    if not row.is_file():
        return f"{out_dir} carries no row-result.json — it is not a completed row"
    result = json.loads(row.read_text(encoding="utf-8"))
    if result.get("timed_out"):
        return (f"the row hit its cap ({result.get('cap_reason')}); a killed CLI "
                f"has no parseable output to freeze")
    if result.get("exit_code") != 0:
        return f"the container exited {result.get('exit_code')}, not 0"

    commands = out_dir / "commands.json"
    if not commands.is_file():
        return "no commands.json — no dispatcher invocation was recorded"
    records = json.loads(commands.read_text(encoding="utf-8"))
    failed = [r for r in records if r.get("exit") != 0]
    if failed:
        return f"a dispatcher invocation failed: {failed}"
    if len(records) < 3:
        return (f"only {len(records)} of the chain's 3 invocations ran — the row "
                f"stopped early")

    root = container_bundle.output_root(out_dir)
    if root is None:
        return "no artifacts/<plan>/ was exported — the entrypoint's finally did not run"
    state_path = root / "state.json"
    if not state_path.is_file():
        return f"no state.json under {root}"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    completed = set(state.get("phases_completed") or {})
    missing = [phase for phase in CHAIN_PHASES if phase not in completed]
    if missing:
        return f"state.json records no completion for {missing}"
    return None


def image_of(out_dir: Path) -> str | None:
    """The image tag the row ran, recovered from its own run command."""
    path = out_dir / "run-command.txt"
    if not path.is_file():
        return None
    for token in path.read_text(encoding="utf-8").split():
        if token.startswith("mllm-e2e:"):
            return token
    return None


def seed(cli: str, out_dir: Path) -> tuple[Path, dict[str, Any]]:
    bundle = container_bundle.capture_bundle(out_dir)
    manifest = {
        "bundle_version": container_bundle.CONTAINER_BUNDLE_VERSION,
        "cli": cli,
        "row_id": container_bundle.row_id_for(cli),
        "run_bundle": out_dir.parent.name,
        "image": image_of(out_dir),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = container_bundle.write_baseline(cli, bundle, manifest)
    return path, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cli", action="append", default=[],
                        help="seed only this CLI (repeatable); default: every "
                             "provider-lane CLI with a chain bundle on disk")
    parser.add_argument("--runs-root", default=str(RUNS),
                        help=f"where the run bundles live (default: {RUNS})")
    parser.add_argument("--run", default="",
                        help="one specific `runs/<utc>/` directory instead of the "
                             "newest bundle per CLI")
    args = parser.parse_args(list(argv) if argv is not None else None)

    runs_root = Path(args.run).parent if args.run else Path(args.runs_root)
    wanted = args.cli or provider_clis()
    unknown = [cli for cli in wanted if cli not in provider_clis()]
    if unknown:
        sys.stderr.write(
            f"seed-container-golden: {unknown} is not a provider-lane CLI in "
            f"redirect.yaml\n")
        return 2

    written, skipped, refused = [], [], []
    for cli in wanted:
        if args.run:
            out_dir = Path(args.run) / container_bundle.row_id_for(cli)
            out_dir = out_dir if out_dir.is_dir() else None
        else:
            out_dir = newest_bundle(cli, runs_root)
        if out_dir is None:
            skipped.append(cli)
            continue
        reason = refusal(out_dir)
        if reason:
            refused.append((cli, out_dir, reason))
            continue
        path, manifest = seed(cli, out_dir)
        written.append((cli, path, manifest))

    for cli, path, manifest in written:
        rel = path.relative_to(SKILL_ROOT.parent.parent)
        sys.stderr.write(f"wrote {rel} (from {manifest['run_bundle']}/"
                         f"{manifest['row_id']})\n")
    if skipped:
        sys.stderr.write(
            f"no chain bundle on disk for {skipped} — run "
            f"`pytest -m container skills/multi-llm/tests/test_container_full_chain.py` "
            f"first\n")
    for cli, out_dir, reason in refused:
        sys.stderr.write(f"REFUSED {cli}: {reason}\n  ({out_dir})\n")

    if written:
        sys.stderr.write(
            "\nRead the diff before committing: every line of it is a claim about "
            "what this CLI's chain produces.\n")
    # A refusal is the failure. Nothing on disk for a CLI is not: a contributor
    # seeding one row deliberately is the normal case.
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
