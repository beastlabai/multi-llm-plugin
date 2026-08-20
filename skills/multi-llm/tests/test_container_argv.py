"""Command-construction regression: the argv we build, against our own table (C8).

**This is not drift detection, and the distinction matters.** Both sides of the
comparison originate in this repository. If an upstream CLI renames or removes a
flag, the wrapper keeps logging the argv *we* generate, byte-identical to
`expected_argv.yaml`, and every assertion here stays green while the real binary
rejects the call — there is no "new upstream argv" to print beside the old one,
because the CLI never told us one. The upstream-facing half of C8 is
`test_container_cli_surface.py`'s help probe.

What this *does* catch is a refactor on our side: a dropped `--json`, a reordered
positional, a lost `--` separator, a leaked stray flag. Those fail here at the
argv rather than inside a decoded stream three layers down, which is worth its
cost.

The comparison runs in **both directions**, like the golden allowlist: every
call-log record must match some entry, and every entry must match at least
`min_records` records. An entry matching nothing is a stale row — a profile we
stopped dispatching — which is as much a finding as an unmatched record.

**What it grades is this invocation's provider-lane evidence**, never "the newest
bundle under `runs/`". A grader pointed at whatever is on disk either skips
(nothing produced yet — this module collects before the lane that feeds it) or
passes against a previous invocation, and neither of those graded the run in front
of it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import yaml

from .container.docker_probe import requires_docker
from .container import harness
from .golden.normalize_argv import normalize_argv, PLACEHOLDER_PROMPT

pytestmark = [pytest.mark.container, requires_docker()]

EXPECTED = harness.HERE / "expected_argv.yaml"

#: The modules whose rows dispatch the argv this table describes. Alphabetically
#: this module collects *before* both of them, so `tests/conftest.py` defers its
#: items to the end of the session — see the fixture below for why a grader that
#: runs before its evidence exists cannot do its job.
PRODUCERS = ("test_container_full_chain.py", "test_container_provider_lane.py")


def _entries():
    return yaml.safe_load(EXPECTED.read_text(encoding="utf-8"))["entries"]


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def _gradeable_bundles(run_root: Path) -> list[Path]:
    """The call logs under one `runs/<utc>/` that this table may be graded against.

    Keyed on the row's own `row-result.json`, never on its id. Two row kinds leave
    a bundle that looks exactly like a gradeable one and is not: the host lane's
    credential-free discovery rows, which stage a shim and assert a command is
    listed without ever dispatching a provider argv, and the credentialed host
    rows, whose own invoke argv is typed into `host_matrix.yaml` rather than built
    by `build_command()` and so has no entry here to match. A run directory
    holding only those starves every entry — observed, and reported as "all 23
    entries matched nothing" when what really happened is that nothing this table
    describes ran at all. A discovery bundle is not a gradeable bundle.
    """
    return [path for path in sorted(run_root.glob("*/calls.jsonl"))
            if _read_json(path.parent / "row-result.json").get("lane") == "provider"]


def _session_bundles() -> list[Path]:
    """Every gradeable bundle THIS pytest invocation produced.

    `harness.MATRIX` is the lane's own per-invocation state — `run_row` claims a
    budget against it before every `docker run` and files the run root it wrote
    into — so this set is "evidence produced after the code under test was last
    edited" by construction, rather than by comparing timestamps and hoping the
    comparison covered every input.
    """
    bundles: list[Path] = []
    for root in harness.MATRIX.roots:
        bundles += _gradeable_bundles(Path(root))
    return bundles


def _newest_gradeable_on_disk() -> tuple[Path | None, list[Path]]:
    """The most recent run directory holding something this table can grade.

    The fallback for a deliberate solo run of this module, and deliberately not
    "the newest run directory": that one is routinely a discovery-only or
    host-only bundle, and pinning to it is what made every entry report zero
    records instead of reaching back to the lane run that produced the evidence.
    """
    if not harness.RUNS.is_dir():
        return None, []
    for root in sorted((path for path in harness.RUNS.iterdir() if path.is_dir()),
                       reverse=True):
        bundles = _gradeable_bundles(root)
        if bundles:
            return root, bundles
    return None, []


def _argv_inputs() -> list[Path]:
    """Everything an older call log could be graded WRONGLY against.

    Not just the provider adapters. `expected_argv.yaml` is *half* of the
    comparison, so an edit to it read against yesterday's logs is precisely the
    false green this guard exists to refuse; `normalize_argv.py` is the lens both
    halves are read through; and the executor and the write scheduler are what
    render an `agent-readonly`/`agent-write` dispatch, so a change there moves
    argv that no adapter mtime accounts for.
    """
    skill = harness.HERE.parents[1]
    paths = [
        EXPECTED,
        harness.HERE.parent / "golden" / "normalize_argv.py",
        skill / "utils" / "subagent_executor.py",
        skill / "utils" / "write_batch_scheduler.py",
        *sorted((skill / "utils" / "providers").glob("*.py")),
    ]
    return [path for path in paths if path.is_file()]


def _producers_selected(session) -> list[str]:
    """Which of the producing modules this invocation collected, by file name."""
    items = getattr(session, "items", None) or []
    collected = {Path(item.nodeid.split("::", 1)[0]).name for item in items}
    return sorted(name for name in PRODUCERS if name in collected)


@pytest.fixture(scope="module")
def records(request):
    """Every wrapper call-log record from THIS invocation's provider-lane rows.

    Reads back what the containers left rather than re-running them: the rows are
    driven by `test_container_provider_lane.py` and `test_container_full_chain.py`,
    and re-running eleven chains to re-read the same logs would triple the lane's
    wall clock for no new evidence. What is *not* negotiable is which logs — this
    invocation's, never the newest directory that happens to be on disk. Reading
    the latter is how `make e2e-container` used to either skip (nothing produced
    yet, because this module sorts first) or pass against a previous run: two
    different ways of not grading the run in front of you.

    `tests/conftest.py` defers this module's items until after the two that
    produce its evidence, which is what makes "this invocation's bundles" a set
    that exists by the time this fixture reads it.

    The read-back from disk survives as the fallback for a solo run of this module
    alone, and only behind the staleness guard below.
    """
    bundles = _session_bundles()
    origin = "this invocation"
    if not bundles:
        if harness.MATRIX.refused:
            # The lane ran out of clock before the rows this module grades could
            # start. That refusal already fails the invocation and names what
            # never ran; repeating it as three more failures buys nothing.
            pytest.skip(
                "the lane's wall-clock cap was spent before the rows this module "
                "grades could run:\n" + harness.MATRIX.report())
        selected = _producers_selected(request.session)
        if selected:
            pytest.fail(
                f"{', '.join(selected)} ran in this invocation and left no "
                f"gradeable provider-lane bundle, so there is nothing from this "
                f"run to grade. This module does NOT fall back to an earlier "
                f"run's logs when the lane that feeds it just failed — grading "
                f"the current table against a previous invocation's argv is the "
                f"false green it exists to prevent. Fix the lane failure this "
                f"one follows."
            )
        root, bundles = _newest_gradeable_on_disk()
        if not bundles:
            pytest.skip(
                "no container call logs on disk — run "
                "`pytest -m container skills/multi-llm/tests/test_container_full_chain.py` "
                "first; this module grades what that one produced"
            )
        # And the evidence must not predate the code it grades. Without this,
        # editing a `build_command()` (or the table itself) and running only this
        # module would compare the NEW expectation against the OLD logs and pass.
        inputs = _argv_inputs()
        # A default that means "1970" is indistinguishable from a real answer, so
        # there is no default: an empty match is a bug in this module, not a
        # stale bundle.
        assert inputs, "no argv input resolved — this module cannot judge staleness"
        newest_input = max(inputs, key=os.path.getmtime)
        bundle_age = max(os.path.getmtime(path) for path in bundles)
        if bundle_age < os.path.getmtime(newest_input):
            pytest.skip(
                f"the newest gradeable call-log bundle ({root.name}, "
                f"{time.ctime(bundle_age)}) predates {newest_input} "
                f"({time.ctime(os.path.getmtime(newest_input))}), so it would "
                f"grade the current table against old argv. Re-run "
                f"`pytest -m container skills/multi-llm/tests/test_container_full_chain.py` "
                f"and try again."
            )
        origin = f"the newest gradeable bundle on disk ({root.name})"

    out = []
    for path in bundles:
        for line in path.read_text(encoding="utf-8",
                                   errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                # A capped row is `docker stop`ped and, if it does not go,
                # SIGKILLed, which can cut the last line in half mid-write —
                # `harness._read_jsonl` skips those for the same reason. The cap
                # itself is already a failure on the row that hit it.
                continue
            if record.get("cli") == "uv":
                continue          # the dispatcher wrapper, not a provider argv
            out.append(record)
    assert out, (
        f"the provider-lane bundles from {origin} hold no CLI call records at "
        f"all, so every entry below would report zero and none of it would mean "
        f"anything. Bundles read: "
        f"{[str(path.parent) for path in bundles]}"
    )
    return out


def _view(record: dict) -> dict:
    return normalize_argv(record["argv"])


def _transport_of(view: dict) -> str:
    """`argv` when the prompt travels on argv, `stdin` when it does not.

    Not cosmetic: `build_command_for_transport(..., "stdin")` returns `cmd[:-1]`,
    so a refactor that stopped putting the prompt last would silently truncate a
    FLAG instead. Keying on transport is what makes that a failure.
    """
    values = [v for v in view["options"].values() if isinstance(v, str)]
    return "argv" if (PLACEHOLDER_PROMPT in view["tail"]
                      or PLACEHOLDER_PROMPT in values) else "stdin"


def _matches(view: dict, entry: dict) -> bool:
    return (
        view["binary"] == entry["binary"]
        and view["subcommands"] == (entry.get("subcommands") or [])
        and view["options"] == (entry.get("options") or {})
        and view["tail"] == (entry.get("tail") or [])
    )


class TestEveryRecordIsExplained:
    def test_each_call_matches_a_declared_entry(self, records):
        entries = _entries()
        unmatched = []
        for record in records:
            view = _view(record)
            transport = _transport_of(view)
            candidates = [e for e in entries
                          if e["provider"] == record["cli"]
                          and e["transport"] == transport]
            if not any(_matches(view, e) for e in candidates):
                unmatched.append((record["cli"], transport, view))
        assert not unmatched, (
            "call-log records that no expected_argv.yaml entry explains — either "
            "build_command() changed (fix the code or re-seed deliberately) or a "
            "new profile/transport is being dispatched:\n"
            + "\n".join(
                f"  {cli} [{transport}]: {json.dumps(view, sort_keys=True)}"
                for cli, transport, view in unmatched[:10])
        )

    def test_a_crossed_transport_would_fail(self, records):
        """The pair exists to be discriminating; prove it is."""
        entries = _entries()
        argv_rows = [e for e in entries if e["transport"] == "argv"]
        assert argv_rows, "no argv-transport entries to check"
        for entry in argv_rows:
            assert entry["prompt"] == "argv", entry["provider"]
            has_prompt = (PLACEHOLDER_PROMPT in (entry.get("tail") or [])
                          or PLACEHOLDER_PROMPT in
                          [v for v in (entry.get("options") or {}).values()
                           if isinstance(v, str)])
            assert has_prompt, (
                f"{entry['provider']}: declared transport `argv` but its "
                f"normalized view carries no {PLACEHOLDER_PROMPT} — the row and "
                f"its transport disagree"
            )


class TestRowDerivedTimeoutValues:
    """The other half of the `<timeout>` placeholder (§3.3, §7 step 6).

    `expected_argv.yaml` reduces `--max-wall-time` to `<timeout>` because its
    seconds come from the slot, so pinning them verbatim would freeze one run's
    slot length into the table. But a placeholder alone would normalize a WRONG
    number to green — 5 s, 0 s, `--max-wall-time 1170s` when the CLI wants bare
    seconds, or a value ABOVE the parent's deadline, all normalize identically.

    So the value is read RAW out of the call log here, and checked against the
    two things that actually matter: it is a bare integer, and it is strictly
    below the slot timeout the adapter was handed, by the shutdown margin. That
    inner-under-outer ordering is the whole reason the flag is wired — qwen must
    reach its own bound and write its structured `FatalBudgetExceededError`
    envelope BEFORE the parent kills it, or the harness reports a bare TIMEOUT
    instead of qwen's own reason.

    Note the bound compared against is the **provider's resolved slot timeout**
    (`providers.yaml`'s `default_timeout`, which `llm_client` also passes to
    `subprocess.run(timeout=…)`), NOT the row's `row_timeout_s`. Those are
    different bounds: `row_timeout_s` caps the whole `docker run`, while this
    flag is held under the per-dispatch deadline. Conflating them would assert
    something the product never promises.
    """

    FLAG = "--max-wall-time"

    def _raw_values(self, records):
        found = []
        for record in records:
            argv = record.get("argv") or []
            if self.FLAG in argv:
                index = argv.index(self.FLAG)
                if index + 1 < len(argv):
                    found.append((record.get("cli"), argv[index + 1]))
        return found

    def test_the_rendered_seconds_are_a_bare_integer(self, records):
        values = self._raw_values(records)
        if not values:
            pytest.skip(f"no call log in this bundle carries {self.FLAG}")
        for cli, value in values:
            assert value.isdigit(), (
                f"{cli}: {self.FLAG} rendered {value!r}, which is not bare "
                f"integer seconds. qwen accepts `30s`/`5m` too, but the adapter "
                f"renders bare seconds deliberately — that removes a rounding "
                f"rule instead of specifying one."
            )

    def test_the_inner_bound_is_strictly_under_the_slot_timeout(self, records):
        values = self._raw_values(records)
        if not values:
            pytest.skip(f"no call log in this bundle carries {self.FLAG}")

        import sys
        sys.path.insert(0, str(harness.HERE.parent.parent))
        from utils.provider_registry import get_provider_timeout

        for cli, value in values:
            outer = get_provider_timeout(cli)
            assert 0 < int(value) < outer, (
                f"{cli}: {self.FLAG}={value} against a resolved slot timeout of "
                f"{outer}s. The inner bound must be strictly under the outer one "
                f"or the CLI is still running when the parent's deadline fires, "
                f"and the harness reports TIMEOUT instead of the CLI's own "
                f"structured budget error."
            )


class TestEveryEntryStillDoesWork:
    def test_no_entry_matches_nothing(self, records):
        """A stale row is as much a finding as an unmatched record."""
        entries = _entries()
        views = [( r["cli"], _view(r)) for r in records]
        stale = []
        for entry in entries:
            hits = sum(1 for cli, view in views
                       if cli == entry["provider"] and _matches(view, entry))
            if hits < int(entry.get("min_records", 1)):
                stale.append((entry["provider"], entry["profile"],
                              entry["transport"], hits))
        assert not stale, (
            "expected_argv.yaml entries that matched fewer records than they "
            "declare — a profile we stopped dispatching, or a row that never "
            "ran:\n"
            + "\n".join(f"  {p} {prof}/{t}: {hits} record(s)"
                        for p, prof, t, hits in stale)
        )
