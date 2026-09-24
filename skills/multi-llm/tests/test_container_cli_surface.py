"""The upstream drift probe: is the surface our adapters depend on still there?

This is the **only** check in the design that can see upstream move. The argv
golden cannot: both sides of it originate in this repository, so when a CLI
renames or drops a flag the wrapper keeps logging the argv *we* generate,
byte-identical to the expected table, and that assertion stays green while the
real binary rejects the call. Read the argv table as coverage of our own
``build_command()``; read this as the drift report.

Three assertions, of three deliberately different strengths:

* **documented options — hard.** Every option in the contract must still appear
  in that version's help. A missing one fails naming the CLI, the pinned and
  installed versions, and the option: a pre-written bug report.
* **hidden options — by acceptance.** An option the binary accepts but does not
  document cannot be asserted by presence, and dropping it from the contract
  would mean never noticing its removal. So it is probed by *acceptance* instead,
  with a bogus-flag control proving the CLI validates flags at all — otherwise
  "accepted" would mean nothing.
* **help digest — soft, reported not failed.** Help text that changed while every
  option we need survived is the early warning a deprecation notice is.
  ``codex``'s ``--full-auto`` → ``--sandbox workspace-write`` is exactly this
  class: the argv golden stays green straight through it in both phases, while
  this probe sees the note appear and then, later, the removal.

The soft one is **written down**, not only skipped — see :class:`_Drift`.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import pytest
import yaml

from .container.docker_probe import requires_docker
from .container import harness
# One matching rule, shared with the seeder that wrote the contract. Presence is
# a whole-token match, never `opt in help_text`: `-m` is a substring of almost
# any help text, so a substring probe would make this assertion vacuous for
# every short option — green whatever upstream did to it.
from .container.seed_cli_surface import missing_options

pytestmark = [pytest.mark.container, requires_docker()]

SURFACE = harness.HERE / "cli-surface.yaml"
IMAGE = harness.DEFAULT_IMAGE


def _rows():
    data = yaml.safe_load(SURFACE.read_text(encoding="utf-8"))
    return [(row["cli"], row) for row in data["clis"]]


ROWS = _rows()
IDS = [cli for cli, _ in ROWS]


def _in_image(argv: list[str], timeout: int = 180,
              env_extra: dict | None = None) -> subprocess.CompletedProcess:
    """Run a command in the pinned image, offline and unprivileged.

    ``env_extra`` is the row's own ``redirect.yaml`` block and nothing more. One
    CLI needs it to print its own help at all: copilot's npm entry point is a
    loader that self-extracts its native runtime into ``$XDG_CACHE_HOME``, the
    image pre-warms ``/opt/mllm/copilot-cache``, and without the variable the
    loader lands on the noexec tmpfs ``$HOME`` and answers ``Failed to load
    package index ... Native addon "runtime" not found``. Probing without it
    grades the harness, not upstream — which is why copilot had no row here for
    as long as this file has existed.
    """
    env_argv = []
    for name, value in (env_extra or {}).items():
        env_argv += ["-e", f"{name}={value}"]
    return subprocess.run(
        ["docker", "run", "--rm", "--network", "none",
         "--user", "10001:10001", "--cap-drop", "ALL",
         "--security-opt", "no-new-privileges",
         "--tmpfs", "/home/e2e:rw,mode=0700,uid=10001,gid=10001",
         "--tmpfs", "/tmp:rw,mode=1777",
         *env_argv,
         "--entrypoint", argv[0], IMAGE, *argv[1:]],
        capture_output=True, text=True, timeout=timeout)


def _row_env(cli: str) -> dict:
    """The `redirect.yaml` `env_extra` for this row, or an empty mapping.

    Read straight from the file rather than through
    `seed_cli_surface.adapter_contract()`: that function resolves redirect.yaml
    at `/opt/mllm/rig/`, which exists only INSIDE the image, so calling it from
    this host-side module raises FileNotFoundError. Swallowing that would hand
    every row an empty env and quietly restore the bug this helper exists to
    fix — which is exactly what a first draft of it did.
    """
    data = yaml.safe_load(
        (harness.HERE / "redirect.yaml").read_text(encoding="utf-8")) or {}
    for row in data.get("clis") or []:
        if row.get("cli") == cli:
            return {str(k): str(v) for k, v in (row.get("env_extra") or {}).items()}
    return {}


#: Written beside the run bundles, like `host-lane-census.json` and
#: `matrix-census.json`. Ephemeral evidence (`make e2e-clean` prunes `runs/`);
#: the per-row run records in `records/` (gitignored, except the schema) have
#: no spelling for a surface comparison.
DRIFT_REPORT_NAME = "cli-surface-drift.json"

#: What a reader does about a drifted row. Deliberately not a `make` target: the
#: re-seed runs the seeder INSIDE the image, and `make e2e-lock-update` prints
#: that `docker run` in full rather than running it, because a digest is a
#: deprecation notice to read and re-seeding it as bookkeeping is how the one
#: upstream-facing signal in the design gets discarded.
RESEED_HINT = (
    "run tests/container/seed_cli_surface.py inside the pinned image, read the "
    "new help text for a deprecation notice, then move that row's `version` and "
    "`help_sha256` into tests/container/cli-surface.yaml. "
    "`make e2e-lock-update` prints the exact `docker run`."
)


class _Drift:
    """The digest comparison, on disk, in the form a reader needs it.

    The digest half is soft by design and ``pytest.skip`` is how it stays soft —
    but a skip is an **anonymous count** in pytest's summary line. It names the
    CLI and the digest only when the invocation happened to pass ``-rs`` *and*
    someone kept the scrollback, which is a weak home for the one signal in this
    design that can see upstream move: the two targets that exist to produce it
    (`e2e-container-latest`, `e2e-container-latest-host`) both end in a build
    log long enough to bury it.

    So every row this module compares is also written to
    ``runs/<utc>/cli-surface-drift.json``: which CLI, the version the contract
    was captured at, the version the image really installed (the two differ
    exactly under `MLLM_PIN_MODE=latest`, which is the mode that produces drift),
    both digests, both help sizes, and what to do about it. It is rewritten after
    **every** row rather than at teardown, so a lane stopped by its own
    wall-clock cap — or a row that fails a hard assertion — still leaves the
    comparisons it got through.

    Rows the contract exempts (``probe: none``) are recorded too: a CLI missing
    from the report would read as one nobody probed, which is the same shape as
    a gap.
    """

    utc: str | None = None
    rows: list[dict] = []

    @classmethod
    def path(cls) -> Path:
        if cls.utc is None:
            cls.utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        return harness.RUNS / cls.utc / DRIFT_REPORT_NAME

    @classmethod
    def report(cls) -> dict:
        drifted = sorted(row["cli"] for row in cls.rows
                         if row["status"] == "drifted")
        return {
            "probe": "cli-surface",
            "contract": SURFACE.name,
            "image": IMAGE,
            "status": "drift" if drifted else "match",
            "counts": {
                status: sum(1 for row in cls.rows if row["status"] == status)
                for status in ("match", "drifted", "exempt")
            },
            "drifted": drifted,
            "rows": sorted(cls.rows, key=lambda row: row["cli"]),
            "reseed": RESEED_HINT,
        }

    @classmethod
    def _record(cls, entry: dict) -> dict:
        # Keyed by CLI rather than appended: the module fixture is cached per
        # param, but a re-recorded row must replace its own entry rather than
        # double-count itself into the census.
        cls.rows = [row for row in cls.rows if row["cli"] != entry["cli"]]
        cls.rows.append(entry)
        path = cls.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cls.report(), indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        return entry

    @classmethod
    def record_exempt(cls, cli: str, row: dict) -> dict:
        return cls._record({
            "cli": cli,
            "status": "exempt",
            "contract_version": str(row.get("version") or ""),
            "reason": str(row.get("reason") or "recorded `probe: none`"),
        })

    @classmethod
    def record_probe(cls, cli: str, row: dict, help_text: str) -> dict:
        """Compare one row's help text against the contract and file the result."""
        digest = hashlib.sha256(help_text.encode("utf-8", "replace")).hexdigest()
        expected = str(row.get("help_sha256") or "")
        entry = {
            "cli": cli,
            "status": "drifted" if digest != expected else "match",
            "contract_version": str(row.get("version") or ""),
            "help_invocation": str(row.get("help_invocation") or ""),
            "expected_help_sha256": expected,
            "observed_help_sha256": digest,
            "expected_help_bytes": row.get("help_bytes"),
            "observed_help_bytes": len(help_text),
        }
        # What the image really installed, never what the lockfile asked for —
        # the same distinction the run records make, and the column that turns
        # "the help text moved" into "it moved with THIS version bump".
        installed = harness.cli_version(cli, IMAGE)
        if installed:
            entry["installed_version"] = installed
        return cls._record(entry)


@pytest.fixture(scope="module", params=ROWS, ids=IDS)
def probe(request):
    """Run one CLI's recorded help invocation against the installed binary.

    Claimed against the lane's own wall-clock cap (C1) like any other row: this
    module starts its own containers rather than going through
    `harness.run_row`, so a cap enforced only there would let the surface probe
    run on past a budget every other lane had already stopped for.
    """
    cli, row = request.param
    if row.get("probe") == "none":
        _Drift.record_exempt(cli, row)
        pytest.skip(f"{cli}: recorded `probe: none` — {row.get('reason')}")
    harness.MATRIX.claim(f"surface-{cli}")
    parts = str(row["help_invocation"]).split()
    real = f"/opt/mllm/real/{row['binary']}"
    started = time.time()
    completed = _in_image([real, *parts[1:]], env_extra=_row_env(cli))
    # "It ran, and it took this long" — deliberately not a verdict. A help
    # invocation's exit status is not what this module grades (several CLIs
    # print help and exit non-zero), so the census records the clock the row
    # spent and leaves the pass/fail to the assertions below.
    harness.MATRIX.note(f"surface-{cli}", "pass", time.time() - started)
    return cli, row, (completed.stdout or "") + (completed.stderr or "")


class TestTheOptionsWeDependOnStillExist:
    def test_every_documented_option_is_still_in_the_help(self, probe):
        cli, row, help_text = probe
        missing = missing_options(row["options"], help_text)
        assert not missing, (
            f"{cli} {row['version']}: the installed version's help no longer "
            f"offers {missing}, which this repo's adapter emits.\n"
            f"probe: {row['help_invocation']}\n"
            f"This is upstream drift, not a test bug: either the adapter must "
            f"stop emitting it, or the option moved and the contract needs the "
            f"new spelling."
        )

    def test_hidden_options_are_still_accepted(self, probe):
        """Absent from the help, but the binary takes them — proven, not assumed."""
        cli, row, _help_text = probe
        hidden = row.get("hidden_options") or []
        if not hidden:
            pytest.skip(f"{cli}: no hidden options recorded")
        real = f"/opt/mllm/real/{row['binary']}"

        # The control first: if this CLI ignores unknown flags, "accepted" below
        # would be worthless and the whole assertion vacuous.
        row_env = _row_env(cli)
        control = _in_image([real, "--mllm-definitely-not-a-flag", "--help"],
                            env_extra=row_env)
        assert control.returncode != 0, (
            f"{cli}: accepts a bogus flag, so flag acceptance proves nothing here "
            f"— the hidden-option assertion below would be vacuous"
        )
        for option in hidden:
            accepted = _in_image([real, option, "--help"], env_extra=row_env)
            assert accepted.returncode == 0, (
                f"{cli} {row['version']}: no longer accepts {option!r} "
                f"(exit {accepted.returncode}). It was undocumented but working "
                f"when the contract was captured; upstream has now removed it.\n"
                f"{(accepted.stderr or accepted.stdout)[:400]}"
            )


class TestHelpTextChangesAreReported:
    def test_the_help_digest_is_reported_not_enforced(self, probe, record_property):
        """Soft by design: a changed digest is a signal, never a failure.

        Failing on it would make every upstream wording tweak a red suite, which
        is how a drift report gets switched off. But *soft* must not mean
        *invisible*: the comparison is filed in `runs/<utc>/cli-surface-drift.json`
        (:class:`_Drift`) before this skips, so the report survives an invocation
        without `-rs` and a scrollback nobody kept.
        """
        cli, row, help_text = probe
        entry = _Drift.record_probe(cli, row, help_text)
        record_property("help_sha256", entry["observed_help_sha256"])
        if entry["status"] == "drifted":
            pytest.skip(
                f"DRIFT REPORT — {cli} {row['version']}: help text changed "
                f"({entry['expected_help_sha256'][:12]}… -> "
                f"{entry['observed_help_sha256'][:12]}…) while every option we "
                f"depend on survived.\n"
                f"  {RESEED_HINT}\n"
                f"  report: {_Drift.path()}"
            )
