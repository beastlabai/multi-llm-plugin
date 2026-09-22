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

**Two axes, two namespaces, two flags.** `--cli` seeds the *reviewer* axis — one
provider CLI answering the eight-phase fan-out — into
`tests/golden/container/<cli>/bundle.json`. `--writer` seeds the *writer* axis
declared by `redirect.yaml`'s `chain_writer_axis:` block — the reviewer pinned
(today `pi`) and the write runner varied — into
`tests/golden/container/writer-<cli>/bundle.json`, from the `chain-writer-<cli>`
row. Both are repeatable and both default to their whole list. Merely widening
`--cli` to accept a writer name would NOT have done it: the two axes resolve to
different baseline paths, read a different run bundle and write a different
manifest (`axis`, the reviewer pin), so one flag would have had to guess which of
those a bare `codex` meant — and guessing wrong silently overwrites
`chain-codex`'s baseline with `chain-writer-codex`'s run.

**FROZEN and NO BASELINE are outcomes, not refusals.** `refusal()` says why a
bundle must not be frozen; it does not say whether the pass failed. A row that
declares `sandbox_skip_when` in `redirect.yaml` *and* whose own evidence carries
the product's `sandbox_unavailable` reason code is a host that cannot run that
row, which is a recorded fact and not a capture failure — so it joins `skipped`
and the pass still exits 0. Two words, depending on what is on disk:

* `FROZEN <cli>: <reason>, baseline left at <captured_at>` — a baseline exists
  and is deliberately left alone. Today `codex`, whose
  `tests/golden/container/codex/bundle.json` was captured `2026-08-27T14:10:33Z`,
  three weeks before the D6 sandbox probe landed, and records a green codex write
  path this host can no longer reproduce.
* `NO BASELINE <cli>: <blocker>, needs a permissive-userns host` — nothing was
  ever captured. Today `writer-codex`, which no host has yet produced.

**Every other refusal still fails the pass**, and the gate is the *evidence* —
the `sandbox_unavailable` reason code in the row's own artifacts — never the
row's name and never a bare non-zero exit. Without that, a `chain-codex` that
broke for some unrelated reason would ride out through the same door, and the
"one reviewed diff" rule this file exists to protect would be resting on an exit
code nobody trusts.

**The debt that keeps landing on this file, and the three lines it always owes.**
A product change that moves what a phase *writes* invalidates all 15 baselines at
once, and this lane is deselected by default (`addopts = -m 'not container'`) — so
the staleness is invisible until someone opts in, and then reads 15 identical
failures as 15 regressions. Whoever files that debt owes the next reader three
things, and they belong HERE rather than only in a plan file, because a plan file
is not what anyone has open at the moment they pay it:

1. **The command.** Host-side, from the repo root, after a green
   `full --quick --yes` chain run per CLI:

       uv run --project skills/multi-llm -- \\
           python skills/multi-llm/tests/container/seed_container_golden.py --cli <cli>

   `--cli` is repeatable and defaults to every provider-lane CLI in
   `redirect.yaml` with a chain bundle on disk, so one bare invocation after a
   full provider-lane sweep re-seeds all 15. It needs Docker and **no
   credential**: the provider lane redirects every CLI at an in-container stub
   and runs `--network none`, so a 15-row sweep spends wall clock (~25 min,
   measured 2026-09-10) and not one API call. Anyone reaching for a real
   provider key to re-seed a golden is in the wrong lane.
2. **The failure to expect until it is paid**, spelled out so nobody debugs it.
   The rule that generalises: a failure *identical across all 15 rows* is a shape
   change meeting a stale baseline, not a regression — the one thing those rows
   do not share is the binary in the middle, so nothing a single CLI does can
   move all 15 pointers the same way. The worked example is
   `plans/reaggregate-identity-and-salvage-gating.md`, whose
   `code-review/.status.json` change was predicted to fail every row on the two
   added `created_paths` (`.status.json`, `.status.lock`) plus a `steps` list
   grown from `["progress.subagents"]` to
   `["phase=code-review", "state=models_complete", "progress.subagents"]`. The
   response to that is a re-seed, never a hunt through the checkpoint writer.
3. **The close condition.** The regenerated `bundle.json` diffs are *reviewed and
   committed* — this script writes host-side precisely so they land as a
   reviewable diff — and `-m container` passes. Not "it passed on one machine":
   an unreviewed re-seed freezes whatever that run happened to do, which is the
   same failure as a `--force`, only slower to notice.

**Status of that particular debt: paid, and NOT by this docstring's own
instructions (verified 2026-09-10).** All 15 baselines carry
`manifest.captured_at = 2026-08-27T14:10:33Z` from run bundle
`20260827T131354Z`, and already hold both extra `created_paths` and the grown
`steps` — the opencode-plugin branch swept the lane after the fix landed and
re-seeded them in passing, before the plan that filed the debt was finished. A
full 15-row sweep on 2026-09-10 (image rebuilt from the working tree, 277 passed
in 24:48) reproduced every baseline at **zero diffs**, so no CLI is stale and
nothing here needs re-seeding. Recorded rather than deleted: the next shape
change gets the same three lines, and the reason this one closed without work is
itself the evidence that the lane, not the plan file, is what notices.
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


def _redirect() -> dict:
    import yaml

    return yaml.safe_load(REDIRECT.read_text(encoding="utf-8")) or {}


def provider_clis() -> list[str]:
    return [row["cli"] for row in _redirect()["clis"]
            if row.get("lane") == "provider"]


def writer_axis() -> list[dict]:
    """`redirect.yaml`'s `chain_writer_axis:` block, or [] while it is absent.

    Absence is tolerated rather than asserted: the block is a top-level sibling
    of `clis:` landing in its own change, and a seeder that died on a KeyError
    would take the reviewer axis's re-seed down with it mid-landing. With no
    block there are no writer rows, `--writer` has nothing to validate against
    and says so, and `--cli` behaves exactly as it always did.
    """
    return [dict(entry) for entry in (_redirect().get("chain_writer_axis") or [])
            if isinstance(entry, dict) and entry.get("writer")]


def writer_clis() -> list[str]:
    return [entry["writer"] for entry in writer_axis()]


def axis_entry(cli: str, axis: str) -> dict:
    """The declaration governing one (cli, axis) pair.

    The two axes declare `sandbox_skip_when` in two different places — a
    `clis:` row for the reviewer axis, a `chain_writer_axis:` entry for the
    writer one — and this is the single place that difference is resolved, so
    every caller below asks one question and gets the right table.
    """
    if axis == container_bundle.AXIS_WRITER:
        for entry in writer_axis():
            if entry["writer"] == cli:
                return entry
        return {}
    for row in _redirect()["clis"]:
        if row.get("cli") == cli and row.get("lane") == "provider":
            return dict(row)
    return {}


def label_for(cli: str, axis: str) -> str:
    """How a row is named on stdout — the same name as its baseline directory.

    `codex` and `writer-codex` are two rows of the same capture pass, and a
    message naming only `codex` for both would be ambiguous in exactly the pass
    where one is frozen and the other was never captured.
    """
    return container_bundle.baseline_dir_name(cli, axis)


def newest_bundle(cli: str, runs_root: Path,
                  axis: str = container_bundle.AXIS_REVIEWER) -> Path | None:
    """The most recent `runs/<utc>/<row-id>/` directory, or None."""
    row_id = container_bundle.row_id_for(cli, axis)
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


#: The product's own reason code for "this runner's sandbox cannot start"
#: (`utils/subagent_executor.py:1814`, D3k rung 1). The gate below is this
#: string and nothing else: not the row's name, not its exit code.
SANDBOX_REASON_CODE = "sandbox_unavailable"

#: What the dispatch prints when it refuses on that code
#: (`utils/subagent_executor.py:1806`). Matched verbatim rather than by keyword
#: so that a log merely *discussing* the sandbox cannot trip it.
_SANDBOX_REFUSAL_MARKER = f"(reason: {SANDBOX_REASON_CODE})"


def _write_reports(out_dir: Path) -> list[Path]:
    """The row's `write_batch_report.json` artifacts — `harness.write_outcomes`'
    own glob, reused rather than re-derived."""
    return sorted((out_dir / "artifacts").glob("*/*/write_batch_report.json"))


def _find_reason_code(value: Any) -> bool:
    """Whether `sandbox_unavailable` appears as a reason code anywhere inside.

    A recursive walk rather than a fixed pointer because the report's shape
    puts the code in different places depending on where the refusal landed —
    a batch record, a task, a nested refusal object — and a pointer written
    against one of those silently stops matching when the scheduler grows
    another.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if (key in ("reason_code", "reason", "write_outcome", "status")
                    and isinstance(item, str)
                    and SANDBOX_REASON_CODE in item):
                return True
            if _find_reason_code(item):
                return True
        return False
    if isinstance(value, list):
        return any(_find_reason_code(item) for item in value)
    return False


def sandbox_refusal(out_dir: Path) -> str | None:
    """The product's own words for a sandbox refusal in this row, or None.

    Two sources, in order, and the second exists because of a measurement:

    1. **`write_batch_report.json`.** The declared source (§3.1): the same
       artifacts `harness.write_outcomes` walks, scanned for the
       `sandbox_unavailable` reason code. This is where the evidence belongs
       and where it will be once the batch records carry it.
    2. **The container's own refusal line.** Measured on
       `runs/20260920T105458Z/chain-codex`, the report from a real
       sandbox-refused row reads
       `{"exit": 3, "batches_dispatched": [], "tasks": []}` — the refusal is
       raised by `preflight_runner` *before* the first batch record exists, so
       the reason code never reaches the file and source 1 alone would find
       nothing. The line the dispatch printed does carry it, verbatim, so it is
       read as a fallback — anchored on a write report that exists and records
       a non-zero exit, so a green row that merely mentioned the string cannot
       ride out on it.

    What is deliberately NOT a source: the row's name, and `exit_code != 0`.
    Either one would let a row that broke for an unrelated reason take the
    frozen door, which is the whole thing this gate is for.
    """
    reports = _write_reports(out_dir)
    for report in reports:
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if _find_reason_code(data):
            return f"{SANDBOX_REASON_CODE} (recorded in {report.name})"

    # Source 2. The anchor: a write phase really ran and really failed.
    failed_reports = []
    for report in reports:
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("exit") not in (0, None):
            failed_reports.append(report)
    if not failed_reports:
        return None
    for name in ("container-stderr.txt", "container-stdout.txt"):
        path = out_dir / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8",
                                   errors="replace").splitlines():
            if _SANDBOX_REFUSAL_MARKER in line:
                return _condense(line)
    return None


def _condense(line: str, limit: int = 160) -> str:
    """One refusal line, reduced to the blocker.

    The product's message is several sentences of explanation and repair
    instructions wrapped around two facts: the reason code, and the probe's
    own tail (`bwrap: No permissions to create a new namespace…`). Those two
    are what say *why this host cannot run this row*; the rest is in the log
    for whoever is going to act on it, and a 700-character line on a capture
    pass's summary is a line nobody reads.
    """
    text = " ".join(line.split())
    tail = ""
    marker = "Probe tail:"
    if marker in text:
        tail = text.split(marker, 1)[1].strip()
        # Stop at the next sentence of repair prose, which always follows.
        tail = tail.split(" Fix:", 1)[0].strip().strip('".')
    reason = SANDBOX_REASON_CODE + (f" — {tail}" if tail else f" — {text}")
    return reason if len(reason) <= limit else reason[:limit - 1] + "…"


def captured_at(cli: str, axis: str) -> str | None:
    """When the baseline being left alone was captured, or None if there is none."""
    path = container_bundle.baseline_path(cli, axis)
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8")).get("manifest")
    except (OSError, ValueError):
        return None
    return (manifest or {}).get("captured_at")


def image_of(out_dir: Path) -> str | None:
    """The image tag the row ran, recovered from its own run command."""
    path = out_dir / "run-command.txt"
    if not path.is_file():
        return None
    for token in path.read_text(encoding="utf-8").split():
        if token.startswith("mllm-e2e:"):
            return token
    return None


def seed(cli: str, out_dir: Path,
         axis: str = container_bundle.AXIS_REVIEWER) -> tuple[Path, dict[str, Any]]:
    bundle = container_bundle.capture_bundle(out_dir)
    entry = axis_entry(cli, axis)
    manifest = {
        "bundle_version": container_bundle.CONTAINER_BUNDLE_VERSION,
        "cli": cli,
        "axis": axis,
        # The reviewer pin, read from the declaration rather than assumed: on
        # the writer axis it is `chain_writer_axis:`'s own `reviewer` key, and
        # a bundle captured against a different one is not comparable to its
        # siblings. On the reviewer axis the CLI is its own reviewer.
        "reviewer": (entry.get("reviewer") if axis == container_bundle.AXIS_WRITER
                     else cli),
        "row_id": container_bundle.row_id_for(cli, axis),
        "run_bundle": out_dir.parent.name,
        "image": image_of(out_dir),
        # Which machine and kernel produced this capture. Passed here rather
        # than left to `write_baseline`'s back-fill so the seeder's own record
        # of the pass names it, and so a frozen sibling's provenance can be
        # read beside a fresh one's.
        "host": container_bundle.host_provenance(),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = container_bundle.write_baseline(cli, bundle, manifest, axis)
    return path, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cli", action="append", default=[],
                        help="seed only this provider-lane CLI on the REVIEWER "
                             "axis (repeatable); default: every provider-lane "
                             "CLI with a chain bundle on disk")
    parser.add_argument("--writer", action="append", default=[],
                        help="seed only this CLI on the WRITER axis "
                             "(repeatable), into writer-<cli>/bundle.json from "
                             "the chain-writer-<cli> row; default: every entry "
                             "in redirect.yaml's chain_writer_axis: block")
    parser.add_argument("--runs-root", default=str(RUNS),
                        help=f"where the run bundles live (default: {RUNS})")
    parser.add_argument("--run", default="",
                        help="one specific `runs/<utc>/` directory instead of the "
                             "newest bundle per CLI")
    args = parser.parse_args(list(argv) if argv is not None else None)

    runs_root = Path(args.run).parent if args.run else Path(args.runs_root)

    # Selection. A bare invocation sweeps BOTH axes, because a bare invocation
    # is "re-seed everything this host produced". Naming either flag narrows to
    # exactly what was named — asking for `--cli aider` and silently getting
    # three writer captures as well would be a surprise, and the two axes'
    # captures land in different directories where it would be an unreviewed
    # one.
    explicit = bool(args.cli or args.writer)
    wanted: list[tuple[str, str]] = []
    if args.cli or not explicit:
        wanted += [(cli, container_bundle.AXIS_REVIEWER)
                   for cli in (args.cli or provider_clis())]
    if args.writer or not explicit:
        wanted += [(cli, container_bundle.AXIS_WRITER)
                   for cli in (args.writer or writer_clis())]

    unknown = [cli for cli in args.cli if cli not in provider_clis()]
    if unknown:
        sys.stderr.write(
            f"seed-container-golden: {unknown} is not a provider-lane CLI in "
            f"redirect.yaml\n")
        return 2
    # Validated against the block, never against `clis:`: every writer is also
    # a provider-lane row, so the looser check would accept `--writer aider`
    # and capture a writer baseline for a CLI with no writer row.
    unknown_writers = [cli for cli in args.writer if cli not in writer_clis()]
    if unknown_writers:
        sys.stderr.write(
            f"seed-container-golden: {unknown_writers} is not a writer in "
            f"redirect.yaml's chain_writer_axis: block "
            f"({writer_clis() or 'the block is absent'})\n")
        return 2

    written, skipped, refused = [], [], []
    for cli, axis in wanted:
        label = label_for(cli, axis)
        if args.run:
            out_dir = Path(args.run) / container_bundle.row_id_for(cli, axis)
            out_dir = out_dir if out_dir.is_dir() else None
        else:
            out_dir = newest_bundle(cli, runs_root, axis)
        if out_dir is None:
            # No run bundle at all, so no evidence. Deliberately NOT upgraded
            # to NO BASELINE even for a `sandbox_skip_when` row: with nothing
            # on disk the blocker would have to be inferred from the row's
            # name, which is precisely what the frozen rule refuses to do.
            skipped.append((label, None))
            continue
        reason = refusal(out_dir)
        if reason:
            frozen = _frozen_disposition(cli, axis, label, out_dir)
            if frozen is not None:
                skipped.append(frozen)
            else:
                refused.append((label, out_dir, reason))
            continue
        path, manifest = seed(cli, out_dir, axis)
        written.append((label, path, manifest))

    for label, path, manifest in written:
        rel = path.relative_to(SKILL_ROOT.parent.parent)
        sys.stderr.write(f"wrote {rel} (from {manifest['run_bundle']}/"
                         f"{manifest['row_id']})\n")
    absent = [label for label, note in skipped if note is None]
    if absent:
        sys.stderr.write(
            f"no chain bundle on disk for {absent} — run "
            f"`pytest -m container skills/multi-llm/tests/test_container_full_chain.py` "
            f"first\n")
    for _label, note in skipped:
        if note is not None:
            sys.stderr.write(note + "\n")
    for label, out_dir, reason in refused:
        sys.stderr.write(f"REFUSED {label}: {reason}\n  ({out_dir})\n")

    if written:
        sys.stderr.write(
            "\nRead the diff before committing: every line of it is a claim about "
            "what this CLI's chain produces.\n")
    # A refusal is the failure. Nothing on disk for a CLI is not: a contributor
    # seeding one row deliberately is the normal case. Neither is a FROZEN or a
    # NO BASELINE row — a host that cannot run a declared row is a recorded
    # fact, and failing the pass on it would train the operator to ignore this
    # script's exit code, which is the one habit the "one reviewed diff" rule
    # depends on not forming.
    return 1 if refused else 0


def _frozen_disposition(cli: str, axis: str, label: str,
                        out_dir: Path) -> tuple[str, str] | None:
    """`(label, message)` when this refusal is a frozen row, else None.

    Both conditions are required and neither is the row's name:

    * the row **declares** `sandbox_skip_when` — a reviewed statement in
      `redirect.yaml` that this row may be unrunnable on some hosts; and
    * the row's own evidence carries the product's `sandbox_unavailable`
      reason code — proof that this is the declared blocker and not some other
      failure wearing the same exit code.

    A row satisfying only the first is an ordinary refusal and fails the pass,
    which is what stops the declaration becoming a blanket amnesty for
    everything `chain-codex` might ever do wrong.

    **And the DECLARATION half is what keeps the door narrow, measured.** On
    `runs/20260920T105458Z` all **16** rows' evidence carries
    `sandbox_unavailable`, not one — because the 13 helper rows still dispatch
    their writes through `chain_write_runner: codex` and so all hit codex's
    bwrap refusal. Gated on the evidence alone, a re-seed pass on this host
    would have frozen fifteen baselines it should have refused and exited 0.
    `sandbox_skip_when` is declared on `codex` and on the `writer-codex` axis
    entry and nowhere else, so those two are the only rows that can take this
    door — and once §3.2 repoints the 13 helpers off codex, the evidence half
    narrows to the same two by itself.
    """
    entry = axis_entry(cli, axis)
    if not entry.get("sandbox_skip_when"):
        return None
    evidence = sandbox_refusal(out_dir)
    if not evidence:
        return None
    stamp = captured_at(cli, axis)
    if stamp is None:
        # Nothing was ever captured for this row — `writer-codex` today. The
        # blocker is recorded and the gap is named as a gap, rather than being
        # reported as a baseline that exists.
        return (label,
                f"NO BASELINE {label}: {evidence}, needs a permissive-userns "
                f"host")
    return (label,
            f"FROZEN {label}: {evidence}, baseline left at {stamp}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
