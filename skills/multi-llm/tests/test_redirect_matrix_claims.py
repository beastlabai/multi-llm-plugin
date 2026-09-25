"""What `tests/container/redirect.yaml` claims, checked against the product — in CI.

The container matrix is data, and data drifts. Three things about it are
checkable without a daemon, and each of them **must** be, because the modules
that consume them are `container`-marked and therefore never run on a PR:

* **`chain_write_runner`** — the executor runner a chain row dispatches every
  subagent through. It is present exactly for the CLIs whose own adapter declares
  no `agent-write` profile, and it names one that does. Asserted here rather than
  inside `test_container_full_chain.py`, because a drift guard that only runs
  where Docker runs is a drift guard that does not run: the first sign of a
  mismatch would otherwise be a chain failing mid-write with a
  `ProfileUnsupported` refusal, which reads as a redirection problem and is
  nothing of the kind.

* **the gating itself** — every `test_container_*.py` module carries the marker
  *and* the no-Docker `skipif`, and the expression CI actually types selects
  nothing from them. Three separate mechanisms have to agree for that
  (`addopts`, the workflow's own `-m`, the module-level `skipif`), and the cost of
  them disagreeing is a red PR gate on three OS runners with no daemon.

* **the sandbox licence** — not the `sandbox_skip_when` declaration, which is
  the first family below, but what spending it does. §3.1's three states are
  decided by `_sandbox_policy`, which lives twice inside
  `test_container_full_chain.py` and therefore executes nowhere a PR can see;
  `TestTheSandboxLicenceIsSpentTheWayItIsDeclared` drives both copies here on
  a **forced** verdict — `ok` grades the row and fails a `sandbox-skip`
  outcome, `unknown` is a hard failure, `unavailable` buys exactly one strict,
  typed xfail — with no daemon and no permissive kernel.

Plain CI: no container, no Docker, no network. One subprocess, for the collection
check, because "what does pytest select" cannot be asserted from inside the run
that pytest already selected.

**The filename is deliberately not `test_container_*`.** That glob is what the
build context ships into the image and what the gating check below enumerates;
a plain-CI module answering to it would be shipped for nothing and would fail its
own marker assertion.
"""

from __future__ import annotations

import ast
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "container"))

from utils.provider_registry import _deep_merge, load_base_config  # noqa: E402
from utils.provider_registry import get_provider  # noqa: E402

import seed_workspace  # noqa: E402  (tests/container/seed_workspace.py)

from .container import harness  # noqa: E402

#: The chain lane's own module, imported for the two `_sandbox_policy`
#: implementations `TestTheSandboxLicenceIsSpentTheWayItIsDeclared` drives on a
#: forced verdict. Importing a `container`-marked module is safe and nearly
#: free here: the marker and the `requires_docker()` skipif gate the
#: COLLECTION of its own tests, not its import, and the one thing the import
#: costs — `docker_probe.docker_skip_reason`'s `docker version` — is
#: `lru_cache`d and already paid by every plain-CI collection of the tests
#: directory.
from . import test_container_full_chain as chain_lane  # noqa: E402

TESTS_DIR = Path(__file__).resolve().parent
REDIRECT = TESTS_DIR / "container" / "redirect.yaml"
HOST_MATRIX = TESTS_DIR / "container" / "host_matrix.yaml"
CONTAINER_MODULES = tuple(sorted(TESTS_DIR.glob("test_container_*.py")))


def _redirect() -> dict:
    return yaml.safe_load(REDIRECT.read_text(encoding="utf-8"))


def _provider_rows() -> list[dict]:
    return [row for row in _redirect()["clis"] if row.get("lane") == "provider"]


PROVIDER_ROWS = _provider_rows()
PROVIDER_IDS = [row["cli"] for row in PROVIDER_ROWS]

#: The writer axis (`plans/chain-lane-write-runner-matrix.md` §3.1, §5.1): a
#: TOP-LEVEL block, sibling to `clis:`, never a row inside it. Three consumers
#: break if it is a row — `test_container_full_chain._rows()` would see a
#: duplicate pytest id and a clobbered `BY_CLI` key, this module's first guard
#: would trip on all three writers, and `container_bundle.baseline_path(cli)`
#: would collide `chain-writer-codex` with `chain-codex` on one file.
WRITER_AXIS = list(_redirect().get("chain_writer_axis") or [])
WRITER_IDS = [str(entry.get("writer")) for entry in WRITER_AXIS]


def _declares_write(cli: str) -> bool:
    provider = get_provider(cli)
    assert provider is not None, f"{cli} has no built-in adapter"
    return "agent-write" in (getattr(provider, "agent_profiles", None) or ())


def _probes_a_sandbox(cli: str) -> bool:
    """Does this adapter's `agent-write` need a kernel feature to dispatch?

    THE predicate of this plan, computed once and reused by three guards
    rather than re-derived (or, worse, spelled as the literal `codex`) in
    each. codex is the only adapter that implements `sandbox_probe_command`
    (`utils/providers/codex.py:430`; every other inherits base's `None` at
    `utils/providers/base.py:2465`), and its `agent-write` renders
    `--sandbox workspace-write`, which is bubblewrap, which needs an
    unprivileged user namespace.

    Written as a property of the adapter and not as a name, so a *second*
    adapter growing a sandbox probe is caught by the same three guards on the
    PR that adds it.
    """
    provider = get_provider(cli)
    assert provider is not None, f"{cli} has no built-in adapter"
    try:
        return bool(provider.sandbox_probe_command("agent-write"))
    except Exception:  # noqa: BLE001 - an adapter may not answer at all
        return False


def _effective_write_runner(row: dict) -> str:
    """What `test_container_full_chain._runner_cli` computes, in plain CI.

    `entry.get("chain_write_runner") or entry["cli"]` — be your own write
    runner if you declare `agent-write`, otherwise use the row's declared
    helper. Duplicated here rather than imported because that module is
    `container`-marked and this one is the guard that must run where Docker
    does not; the expression is one line and the guard below asserts the two
    stay in agreement by construction.
    """
    return str(row.get("chain_write_runner") or row["cli"])


class TestTheChainWriteRunner:
    """`chain_write_runner` against the adapters, in both directions."""

    @pytest.mark.parametrize("row", PROVIDER_ROWS, ids=PROVIDER_IDS)
    def test_it_is_declared_exactly_where_the_cli_has_no_agent_write(self, row):
        declared = row.get("chain_write_runner")
        if _declares_write(row["cli"]):
            assert declared is None, (
                f"{row['cli']} declares agent-write itself, so redirect.yaml must "
                f"not name a chain_write_runner ({declared}) — that would add a "
                f"second binary to the row for nothing, and the per-row "
                f"single-CLI assertion would then require it to have run"
            )
        else:
            assert declared is not None, (
                f"{row['cli']} declares no agent-write profile, so its chain "
                f"row's executor needs a different binary. Declare it as "
                f"chain_write_runner or the row fails at its first write "
                f"dispatch with a ProfileUnsupported refusal."
            )

    @pytest.mark.parametrize("row", PROVIDER_ROWS, ids=PROVIDER_IDS)
    def test_the_named_runner_can_actually_write(self, row):
        declared = row.get("chain_write_runner")
        if declared is None:
            pytest.skip(f"{row['cli']} is its own write runner")
        assert _declares_write(declared), (
            f"{declared} is named as {row['cli']}'s write runner but declares no "
            f"agent-write profile of its own"
        )

    def test_the_named_runner_is_itself_a_redirected_row(self):
        """A helper nobody redirects would reach for a real endpoint."""
        redirected = set(PROVIDER_IDS)
        for row in PROVIDER_ROWS:
            declared = row.get("chain_write_runner")
            if declared is not None:
                assert declared in redirected, (
                    f"{row['cli']}'s write runner {declared} is not a "
                    f"provider-lane row, so nothing redirects it at the stub"
                )


# --------------------------------------------------------------------------- #
# The three anti-regression guards this plan is NAMED for.
#
# Nothing asserted, before these, that the writer axis exists at all or that
# the 13 reviewer rows stay off codex. They fit here because this module
# already calls the shipped adapters with no Docker and already uses the
# both-directions shape, and each is one function. They cost no container time
# and no key, so neither task has an excuse to defer them.
# --------------------------------------------------------------------------- #


class TestTheWriterAxis:
    """§3.1's three rows: reviewer fixed, write runner varied."""

    def test_the_block_exists_and_is_a_sibling_of_clis(self):
        assert WRITER_AXIS, (
            "redirect.yaml declares no `chain_writer_axis:` block. Without it "
            "the chain lane grades one question per row and grades two things "
            "with it, which is how codex's write path came to be graded 14 "
            "times and the other two once each."
        )
        assert "chain_writer_axis" not in {
            key for row in _redirect()["clis"] for key in row
        }, "the axis belongs at the top level, not inside a `clis:` row"

    def test_it_holds_exactly_one_entry_per_write_capable_provider_cli(self):
        """Derived from the ADAPTERS, not hard-coded to three.

        A fourth write-capable CLI joining the roster then fails here until it
        has a writer row, rather than silently costing G1 a third of its
        coverage.
        """
        expected = {cli for cli in PROVIDER_IDS if _declares_write(cli)}
        assert set(WRITER_IDS) == expected, (
            f"the writer axis names {sorted(WRITER_IDS)} but the adapters "
            f"declare agent-write for {sorted(expected)}. Every write-capable "
            f"provider-lane CLI needs exactly one writer row, or its real "
            f"write path is graded nowhere."
        )
        assert len(WRITER_IDS) == len(set(WRITER_IDS)), (
            f"duplicate writer entries: {WRITER_IDS}"
        )

    @pytest.mark.parametrize("entry", WRITER_AXIS, ids=WRITER_IDS or None)
    def test_each_entry_carries_its_three_expected_keys(self, entry):
        assert entry.get("writer"), f"{entry}: no writer"
        assert entry.get("reviewer"), f"{entry}: no reviewer"
        # `sandbox_skip_when` on exactly those writers whose adapter probes.
        declared = entry.get("sandbox_skip_when")
        if _probes_a_sandbox(entry["writer"]):
            assert declared, (
                f"{entry['writer']}'s adapter answers a "
                f"sandbox_probe_command('agent-write'), so its writer row can "
                f"be refused before any child starts. Declare "
                f"sandbox_skip_when or the refusal reads as a bare red."
            )
        else:
            assert declared is None, (
                f"{entry['writer']} spawns no sandbox — its agent-write is an "
                f"application-level permission mechanism with nothing that can "
                f"fail to start — so a sandbox_skip_when declaration on it is "
                f"a standing licence to forgive a failure that has no cause."
            )

    @pytest.mark.parametrize("entry", WRITER_AXIS, ids=WRITER_IDS or None)
    def test_the_writer_names_a_clis_row_that_declares_agent_write(self, entry):
        assert entry["writer"] in PROVIDER_IDS, (
            f"{entry['writer']} is not a lane: provider row, so nothing "
            f"redirects it at the stub"
        )
        assert _declares_write(entry["writer"]), (
            f"{entry['writer']} declares no agent-write profile of its own, so "
            f"it cannot be a writer-axis writer"
        )

    def test_the_reviewer_is_one_pin_on_every_entry_and_cannot_write(self):
        """§3.1's first pin constraint, both halves.

        The reviewer must be one of the CLIs that do NOT declare `agent-write`:
        a writer-axis row pairs a reviewer with a *different* write runner,
        which is exactly what `test_it_is_declared_exactly_where_the_cli_has_no_
        agent_write` forbids for a CLI that can write for itself. And it must be
        the SAME one on every entry, recorded once in the block, so the three
        baselines differ from one another only in the write half.
        """
        reviewers = {str(entry.get("reviewer")) for entry in WRITER_AXIS}
        assert len(reviewers) == 1, (
            f"the writer axis pins {sorted(reviewers)} as reviewers. It must "
            f"be one pin: a bundle captured against a different reviewer is "
            f"not comparable to its siblings."
        )
        reviewer = reviewers.pop()
        assert reviewer in PROVIDER_IDS, (
            f"the reviewer pin {reviewer} is not a lane: provider row"
        )
        assert not _declares_write(reviewer), (
            f"the reviewer pin {reviewer} declares agent-write, so pinning it "
            f"would make the axis fire the first-direction assertion that "
            f"keeps the reviewer axis honest, and would attach a second binary "
            f"to a row that does not need one."
        )


class TestTheReviewerAxisStaysOffASandboxedRunner:
    """§3.2's two guards: one declared helper, and never a probing runner."""

    def test_every_declared_helper_is_the_same_one_name(self):
        """Compared against a single declared value, not spelled 13 times.

        A row repointed by hand — or a row added later that copies a
        neighbour's old `chain_write_runner: codex` line — fails on the PR
        that does it. Written against "the declared shared helper" rather than
        a literal, because the choice is settleable; whichever it settles on,
        the rows must agree with it.
        """
        declared = {str(row["chain_write_runner"]) for row in PROVIDER_ROWS
                    if row.get("chain_write_runner")}
        assert len(declared) == 1, (
            f"the reviewer axis names {sorted(declared)} as its shared write "
            f"helper. It must be exactly one: a mixed lane is the failure mode "
            f"the one-reviewed-diff rule exists to prevent, and a half-repointed "
            f"matrix produces baselines that cannot be compared."
        )

    @pytest.mark.parametrize("row", PROVIDER_ROWS, ids=PROVIDER_IDS)
    def test_no_rows_effective_write_runner_probes_a_sandbox(self, row):
        """The general statement, which catches what the guard above does not.

        A *new* provider row shipped with `chain_write_runner: codex`, and any
        wholesale revert of the 13. Stated as a property of the adapter rather
        than by name: no row's effective write runner may be a CLI whose
        adapter declares a `sandbox_probe_command('agent-write')` unless that
        row is that CLI's own row — where the runner is not a choice and the
        `sandbox_skip_when` mechanism handles it instead.
        """
        runner = _effective_write_runner(row)
        if runner == row["cli"]:
            return
        assert not _probes_a_sandbox(runner), (
            f"{row['cli']} borrows {runner} as its write runner, and "
            f"{runner}'s agent-write needs a kernel feature to dispatch "
            f"(`sandbox_probe_command('agent-write')` is not None). On a host "
            f"whose kernel restricts unprivileged user namespaces that row "
            f"cannot run at all, and it is not even that row's own question."
        )


class TestTheSandboxSkipListIsExhaustive:
    """§3.1's plain-CI guard: the skip declaration, in both directions.

    A harness refactor that stops emitting `sandbox-skip`, or an adapter that
    grows a probe with no declaration, then fails on a PR rather than quietly
    turning the named skip back into a bare red — or, worse, into a silent
    pass.
    """

    @pytest.mark.parametrize("row", PROVIDER_ROWS, ids=PROVIDER_IDS)
    def test_it_is_declared_exactly_where_the_runner_probes(self, row):
        declared = row.get("sandbox_skip_when")
        probes = _probes_a_sandbox(_effective_write_runner(row))
        if probes:
            assert declared == "userns-unavailable", (
                f"{row['cli']}'s effective write runner "
                f"{_effective_write_runner(row)} answers a sandbox probe, so "
                f"its write batches can be refused before any child starts. "
                f"Declare `sandbox_skip_when: userns-unavailable` or the "
                f"refusal lands in the census as `container exit 3`, which "
                f"reads identically to a broken write path."
            )
        else:
            assert declared is None, (
                f"{row['cli']} declares sandbox_skip_when: {declared!r}, but "
                f"its effective write runner spawns no sandbox. A licence to "
                f"forgive a failure that has no cause outlives nothing and "
                f"excuses everything."
            )


# --------------------------------------------------------------------------- #
# §3.1's OTHER plain-CI guard: not that the licence is declared in the right
# rows — the guard above is that — but what the declaration buys when it is
# spent.
#
# That decision lives in `_sandbox_policy`, twice: once on the writer axis's
# test class and once on `_ReviewerSandboxPolicy`, the mixin the five reviewer
# classes carry. Both sit inside `test_container_full_chain.py`, which is
# `container`-marked behind `requires_docker()`, so on every CI runner and on
# this host neither line ever executes. A policy that xfailed an `ok` verdict,
# or that read `unknown` as a licence, would stay green until somebody spent a
# keyed Docker run to discover it — which is the same "guard that only runs
# where Docker runs" problem `chain_write_runner` is checked here to avoid.
#
# So the three states are driven here with the verdict FORCED rather than
# measured: no daemon, no image, no permissive userns. What is NOT faked is the
# classification — each case is graded from a row whose own
# `write_batch_report.json` carries the executor's refusal, through the real
# `harness.row_outcome`, so a `row_outcome` that stopped emitting
# `sandbox-skip` fails these too instead of quietly making them vacuous.
# --------------------------------------------------------------------------- #

#: The one CLI whose `agent-write` spawns a sandbox, on both axes (§0), and so
#: the only row either axis can ever license. Read as a name rather than
#: derived: the two tests below are about the mechanism, and the set of rows it
#: applies to is `TestTheSandboxSkipListIsExhaustive`'s question.
SANDBOX_CLI = "codex"

#: A test name deliberately OUTSIDE the writer axis's `GRADED_ON_REFUSAL` set,
#: so what these cases drive is the general licence branch and not the two
#: Tier 1 containment tests that are exempt from it.
POLICY_NODE = "test_the_row_exited_zero"


def _fixture_body(owner: type, name: str):
    """The plain function under an `@pytest.fixture`, across pytest versions.

    pytest 9 hands back a `FixtureFunctionDefinition` carrying `__wrapped__`;
    pytest 7/8 hand back a wrapper carrying `__pytest_wrapped__.obj` that
    refuses to be called directly. Both are unwrapped here so the policy can be
    driven as the ordinary function it is, with a request object of our own.
    """
    attr = owner.__dict__.get(name, None) or getattr(owner, name)
    wrapped = getattr(attr, "__wrapped__", None)
    if wrapped is not None:
        return wrapped
    marker = getattr(attr, "__pytest_wrapped__", None)
    if marker is not None:
        return marker.obj
    return attr


class _Node:
    """Just enough of a pytest item for the policy: a name, and `add_marker`."""

    def __init__(self, name: str = POLICY_NODE):
        def function():  # pragma: no cover - only its `__name__` is read
            raise AssertionError("the policy must never call the test body")

        function.__name__ = name
        self.function = function
        self.markers: list = []

    def add_marker(self, marker) -> None:
        self.markers.append(marker)


class _Case:
    """One axis's policy, bound to one forced verdict and one refused row."""

    def __init__(self, policy, fixture_value, result: harness.RowResult):
        self.policy = policy
        self.value = fixture_value
        self.result = result
        self.node = _Node()
        self.body = _fixture_body(type(policy), "_sandbox_policy")

    @property
    def markers(self) -> list:
        return self.node.markers

    def apply_policy(self) -> list:
        """Run the autouse fixture's body against a request of our own."""
        request = type("_Request", (), {"node": self.node})()
        self.body(self.policy, request, self.value)
        return self.markers

    def refuse(self) -> None:
        """Run what every graded test body calls first."""
        self.policy._refuse_if_skipped(self.value)


def _refused_row_result(tmp_path: Path, cli: str,
                        row_id: str) -> harness.RowResult:
    """A row whose OWN evidence `harness.row_outcome` classifies `sandbox-skip`.

    The product's shape, not a stand-in for the classifier: a
    `write_batch_report.json` under `artifacts/<phase>/<batch>/` carrying the
    `reason_code` the executor records when it refuses to dispatch, plus the
    non-zero row exit that refusal produces and no cap hit. Grading through the
    real classification is the point — a policy pinned against a hand-made
    "sandbox-skip" string would survive `row_outcome` ceasing to emit one.
    """
    report = (tmp_path / "artifacts" / "apply-suggestions" / "batch-1"
              / "write_batch_report.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "exit": 3,
        "batches_dispatched": [],
        "tasks": [{
            "reason_code": "sandbox_unavailable",
            "detail": (f"Refusing to dispatch: {cli} agent-write "
                       f"(reason: sandbox_unavailable) — the sandbox could "
                       f"not start"),
        }],
    }), encoding="utf-8")
    return harness.RowResult(
        row=harness.Row(row_id=row_id, lane="provider", cli=cli),
        exit_code=3,
        wall_clock_s=1.0,
        out_dir=tmp_path,
    )


def _reviewer_case(monkeypatch, tmp_path: Path, state: str,
                   tail: str) -> _Case:
    """`_ReviewerSandboxPolicy`, the mixin all five reviewer classes carry."""
    entry = chain_lane.BY_CLI[SANDBOX_CLI]
    runner = chain_lane._runner_cli(entry)
    # The memo the `chain` fixture fills from an in-image probe. Seeding it is
    # what makes all three states reachable with no daemon and no kernel
    # feature — and it is restored on teardown, so no container run can
    # inherit a verdict this module invented.
    monkeypatch.setitem(chain_lane._SANDBOX_VERDICTS, runner, (state, tail))
    result = _refused_row_result(tmp_path, entry["cli"], f"chain-{entry['cli']}")
    return _Case(chain_lane._ReviewerSandboxPolicy(), result, result)


def _writer_case(monkeypatch, tmp_path: Path, state: str, tail: str) -> _Case:
    """The writer axis's copy of the same three-state rule."""
    exempt = chain_lane.TestTheWriterAxisGradesTheWritePath.GRADED_ON_REFUSAL
    assert POLICY_NODE not in exempt, (
        f"{POLICY_NODE} is now one of the writer axis's Tier 1 tests, which "
        f"are exempt from the licence. These cases would then drive the "
        f"carve-out and not the policy — name a test outside {sorted(exempt)}."
    )
    entries = [entry for entry in chain_lane.WRITER_AXIS
               if entry.get("writer") == SANDBOX_CLI]
    assert entries, (
        f"redirect.yaml's chain_writer_axis declares no {SANDBOX_CLI} writer, "
        f"so the only row either axis can license is gone. Retire this guard "
        f"deliberately or restore the row."
    )
    entry = entries[0]
    result = _refused_row_result(tmp_path, entry["reviewer"],
                                 f"chain-writer-{SANDBOX_CLI}")
    run = chain_lane.WriterRun(entry, result, (state, tail))
    return _Case(chain_lane.TestTheWriterAxisGradesTheWritePath(), run, result)


#: Both copies of the policy, so a fix applied to one and not the other fails
#: here rather than on the axis nobody re-ran.
AXES = {"reviewer": _reviewer_case, "writer": _writer_case}


class TestTheSandboxLicenceIsSpentTheWayItIsDeclared:
    """§3.1's three states, proved on this host with the verdict forced.

    `unavailable` is the state a restricted-userns host reaches by itself; the
    other two are the ones that decide whether the mechanism is a licence or a
    loophole, and neither is reachable here without forcing. All three are
    driven against the real `harness.row_outcome` classification.
    """

    @pytest.mark.parametrize("axis", sorted(AXES), ids=sorted(AXES))
    def test_an_ok_verdict_grades_the_row_and_fails_a_sandbox_skip(
            self, axis, monkeypatch, tmp_path):
        """`ok` / `not-required`: no skip is available at all.

        Two assertions, because the second is the one a loophole would slip
        through: no marker is added, *and* a row that nonetheless recorded a
        sandbox refusal fails — the classification and the capability disagree
        and one of them is wrong. The failure must not be the typed one an
        xfail forgives, or a capable host would still be able to record a
        refusal and be green.
        """
        case = AXES[axis](monkeypatch, tmp_path, "ok", "")
        assert harness.row_outcome(case.result)[0] == "sandbox-skip", (
            f"{axis}: the fabricated row is no longer classified sandbox-skip "
            f"by harness.row_outcome, so this case grades nothing: "
            f"{harness.row_outcome(case.result)}"
        )
        markers = case.apply_policy()
        assert not markers, (
            f"{axis}: the policy added {markers} on an `ok` verdict. `ok` "
            f"means no skip is available — the row is graded like every other "
            f"row, and an xfail here forgives a failure the host can run."
        )
        with pytest.raises(AssertionError) as excinfo:
            case.refuse()
        assert not isinstance(excinfo.value, harness.SandboxSkipped), (
            f"{axis}: a `sandbox-skip` outcome under an `ok` verdict raised "
            f"the typed failure the xfail is declared against, so on a capable "
            f"host the refusal would still be forgiven. It must be an ordinary "
            f"red: {excinfo.value}"
        )

    @pytest.mark.parametrize("axis", sorted(AXES), ids=sorted(AXES))
    def test_an_unknown_verdict_is_a_hard_failure(
            self, axis, monkeypatch, tmp_path):
        """`unknown`: nothing was learned, so nothing may be forgiven.

        `pytest.raises(pytest.fail.Exception)` also pins that it is not a
        `pytest.skip` — `Skipped` is not a `Failed`, so a policy that skipped
        here fails this test rather than passing it.
        """
        case = AXES[axis](monkeypatch, tmp_path, "unknown",
                          "the probe could not be run")
        with pytest.raises(pytest.fail.Exception) as excinfo:
            case.apply_policy()
        assert "unknown" in str(excinfo.value), (
            f"{axis}: the hard failure does not say the verdict was unknown, "
            f"so the reader cannot tell it from an ordinary red: "
            f"{excinfo.value}"
        )
        assert not case.markers, (
            f"{axis}: the policy added {case.markers} on an `unknown` verdict. "
            f"An unreadable verdict must not become the cheap path to a green "
            f"run."
        )

    @pytest.mark.parametrize("axis", sorted(AXES), ids=sorted(AXES))
    def test_an_unavailable_verdict_spends_one_strict_typed_xfail(
            self, axis, monkeypatch, tmp_path):
        """The control without which the two above are satisfied by no policy.

        A `_sandbox_policy` that did nothing at all would pass "no marker on
        `ok`"; this is what makes that assertion mean *no marker on `ok`,
        rather than never a marker*. Strict and typed are both asserted: strict
        so the day the sandbox starts the row's XPASS is red, typed so a row
        failing for any other reason stays an ordinary red.
        """
        case = AXES[axis](monkeypatch, tmp_path, "unavailable",
                          "bwrap: setting up uid map: Permission denied")
        markers = case.apply_policy()
        assert len(markers) == 1 and markers[0].name == "xfail", (
            f"{axis}: an `unavailable` verdict on a declared row produced "
            f"{markers} instead of exactly one xfail"
        )
        assert markers[0].kwargs.get("strict") is True, (
            f"{axis}: the xfail is not strict, so the day the sandbox starts "
            f"here the row passes silently and the declaration outlives its "
            f"blocker"
        )
        assert markers[0].kwargs.get("raises") is harness.SandboxSkipped, (
            f"{axis}: the xfail is untyped "
            f"({markers[0].kwargs.get('raises')!r}), so the row failing for "
            f"any other reason would be forgiven too"
        )
        with pytest.raises(harness.SandboxSkipped):
            case.refuse()


# --------------------------------------------------------------------------- #
# §3.3 step 4: the host lane's provider partition, asserted in plain CI where
# Docker is not.
#
# The statement, once: *every `lane: provider` CLI in `redirect.yaml` appears
# in exactly one fan-out row's `fan_out` across the three hosts, where a
# fan-out row is one that is neither `remove_seed` nor `fan_out: native`; and
# every name appearing in any `fan_out` is a `lane: provider` CLI.*
#
# Both directions, so a CLI added to `redirect.yaml` with no host assignment
# fails loudly and cheaply here rather than being silently uncovered. The
# derivation itself is `harness`'s, shared with the fixture that builds the
# row — a guard that re-derives the partition asserts its own copy of it.
#
# **...LESS THE CARVED-OUT IDS, AND THAT HALF ONLY EXISTS AS OF 2026-09-22.**
# `host_matrix.yaml`'s partition header, `redirect.yaml`'s `grok` row and
# `plans/chain-lane-write-runner-matrix.md` (§3.3, §6, §8) have all said "the
# union guard reads those cells" since the partition was authored. It did not:
# `expected` was `set(PROVIDER_IDS)` with nothing subtracted, so the remedy
# this guard's own failure message offers — "or record a dated carve-out
# cell" — could not be taken, and the first CLI the lane proved it cannot
# carry (`grok`, refused by xAI under this lane's open egress, measured on
# `runs/20260922T084502Z`) had no expressible home. `_fan_out_carve_outs`
# below closes that, and deliberately closes it as a TIGHTENING: a cell must
# be complete, dated, run-bearing and name what retires it, and its CLI must
# be in no host's list, so the subtraction is a recorded exemption and never a
# bare key that drops a provider. `test_a_cli_missing_from_every_list_with_no
# _carve_out_still_fails` is the negative control on exactly that property,
# because a guard that can be satisfied by deletion is the hole this replaces.
#
# This grades CONFIGURATION coverage only; it never starts a container. The
# runtime half — each assigned provider correlated, after the host-start
# watermark, with a `/opt/mllm/bin` invocation, a stub session and a result
# file of its own — lives in `test_container_host_lane.py`, because a row
# could fan out to six providers, have five never invoked, and stay green on
# every assertion this module can make.
# --------------------------------------------------------------------------- #

def _load_matrix() -> dict:
    """The matrix as data, so the negative control can drive the real
    reduction over a MUTATED copy instead of re-implementing the rule."""
    return yaml.safe_load(HOST_MATRIX.read_text(encoding="utf-8"))


def _host_rows() -> list[tuple[str, dict, dict]]:
    matrix = _load_matrix()
    return [(host, entry, row)
            for host, entry in (matrix.get("hosts") or {}).items()
            for row in (entry.get("rows") or [])]


HOST_ROWS = _host_rows()
HOST_ROW_IDS = [row.get("id", "?") for _h, _e, row in HOST_ROWS]


def _fan_out_hosts(matrix: dict | None = None) -> dict[str, list[str]]:
    """Each host's DECLARED list — the subject of the union and disjointness.

    The host's own list, not the rows': a row-level override may narrow a row
    and may never widen the partition or move a CLI between hosts, which is
    the separate assertion below.
    """
    matrix = _load_matrix() if matrix is None else matrix
    return {host: [str(c) for c in (entry.get("fan_out") or [])]
            for host, entry in (matrix.get("hosts") or {}).items()}


#: The per-CLI carve-out's four cells, in the `full_row: not-applicable` shape
#: the matrix already uses for a committed, measured "not applicable, because
#: …". All four are required: a verdict nobody can read as an oversight, a
#: provenance word from the matrix's OWN vocabulary, the task that owns the
#: exemption, and the reason.
CARVE_OUT_CELLS = ("member", "provenance", "task", "reason")

#: `host_matrix.yaml`'s header defines these five and no others. `measured-red`
#: is the interesting one here: "a named run exercised this cell and
#: CONTRADICTED it … carries the bundle id AND the task that owns the
#: correction", which is why a cell claiming it must cite a run id below.
CARVE_OUT_PROVENANCE = {
    "verified", "measured-red", "unmeasured", "documented", "derived",
}

#: A run bundle id (`runs/20260922T084502Z`) and an ISO date. A carve-out with
#: neither is an opinion; the matrix's whole provenance scheme exists to keep
#: opinions out of cells that suppress coverage.
RUN_ID_RE = re.compile(r"\b\d{8}T\d{6}Z\b")
DATE_RE = re.compile(r"\b20\d\d-\d\d-\d\d\b")


def _fan_out_carve_outs(matrix: dict | None = None
                        ) -> list[tuple[str, str, object]]:
    """`(host, cli, cell)` for every `fan_out_carve_outs` entry in the file.

    A list rather than a dict, so the same CLI carved on two hosts is visible
    to the guard below instead of one entry silently winning the merge.
    """
    matrix = _load_matrix() if matrix is None else matrix
    return [(host, str(cli), cell)
            for host, entry in (matrix.get("hosts") or {}).items()
            for cli, cell in ((entry.get("fan_out_carve_outs") or {}).items())]


def _carved_ids(matrix: dict | None = None) -> set[str]:
    return {cli for _host, cli, _cell in _fan_out_carve_outs(matrix)}


def _partition_shortfall(matrix: dict) -> tuple[list[str], list[str]]:
    """The union guard's whole statement, as a function of the file.

    `missing` — a `lane: provider` CLI that no host fans out to AND that no
    carve-out cell accounts for. `extra` keeps its original meaning exactly
    (a name in some `fan_out` that is not a provider-lane row): a carved CLI
    left in a list is NOT reported here, because its own guard names that
    contradiction far better than "not a lane: provider row" would.
    """
    union = {cli for members in _fan_out_hosts(matrix).values() for cli in members}
    missing = sorted((set(PROVIDER_IDS) - _carved_ids(matrix)) - union)
    extra = sorted(union - set(PROVIDER_IDS))
    return missing, extra


class TestTheHostFanOutPartition:

    def test_every_host_declares_a_fan_out(self):
        declared = _fan_out_hosts()
        empty = sorted(host for host, members in declared.items() if not members)
        assert not empty, (
            f"{empty} declare no `fan_out`. Until 2026-09-22 the lane's "
            f"fan-out was the hard-coded `codex:stub-model` at "
            f"test_container_host_lane.py:530, for every row on every host, "
            f"and 15 of the 16 stub-capable provider CLIs never appeared in "
            f"the lane at all."
        )

    def test_the_three_subsets_are_disjoint(self):
        declared = _fan_out_hosts()
        seen: dict[str, str] = {}
        clashes: list[str] = []
        for host, members in sorted(declared.items()):
            for cli in members:
                if cli in seen:
                    clashes.append(f"{cli} is on both {seen[cli]} and {host}")
                seen[cli] = host
        assert not clashes, (
            "the instruction is a UNIQUE subset per host: " + "; ".join(clashes)
        )

    def test_their_union_is_every_provider_lane_cli(self):
        matrix = _load_matrix()
        missing, extra = _partition_shortfall(matrix)
        carved = sorted(_carved_ids(matrix))
        assert not missing, (
            f"{missing} are lane: provider rows assigned to no host and "
            f"accounted for by no carve-out cell, so the lane covers them "
            f"nowhere. Assign each to exactly one host's `fan_out`, or record "
            f"a dated `fan_out_carve_outs` cell on the host that would have "
            f"carried it (carved today: {carved or 'none'})."
        )
        assert not extra, (
            f"{extra} appear in a host's `fan_out` but are not lane: provider "
            f"rows, so nothing redirects them at the stub"
        )

    def test_every_carve_out_cell_is_complete_and_dated(self):
        """A carve-out is a RECORDED EXEMPTION, never an escape hatch.

        The union guard subtracts these ids, so the cell is the only thing
        standing between "this provider is covered nowhere" and a green lane.
        A bare `grok: true` would replace one hole with another, so all four
        cells of the `full_row: not-applicable` shape are required, the
        provenance word must come from the matrix's own vocabulary, and the
        reason must carry three things a later reader cannot reconstruct: a
        date, the run that measured it, and **what retires the cell**. An
        exemption that cannot expire is how a temporary one becomes permanent.
        """
        problems: list[str] = []
        for host, cli, cell in _fan_out_carve_outs():
            where = f"{host}.fan_out_carve_outs.{cli}"
            if not isinstance(cell, dict):
                problems.append(f"{where} is {type(cell).__name__}, not a cell "
                                f"with {list(CARVE_OUT_CELLS)}")
                continue
            for key in CARVE_OUT_CELLS:
                if not str(cell.get(key) or "").strip():
                    problems.append(f"{where} declares no `{key}`")
            if cell.get("member") not in (None, "not-applicable"):
                problems.append(
                    f"{where}.member is {cell.get('member')!r}; the verdict "
                    f"word is `not-applicable`, as `full_row` already uses")
            provenance = cell.get("provenance")
            if provenance is not None and provenance not in CARVE_OUT_PROVENANCE:
                problems.append(
                    f"{where}.provenance is {provenance!r}, which is not one "
                    f"of host_matrix.yaml's own words "
                    f"{sorted(CARVE_OUT_PROVENANCE)}")
            reason = str(cell.get("reason") or "")
            if reason and not DATE_RE.search(reason):
                problems.append(f"{where}.reason carries no ISO date")
            if reason and "retire" not in reason.lower():
                problems.append(
                    f"{where}.reason never says what RETIRES the cell, so "
                    f"nothing tells a later reader when the provider comes "
                    f"back into the partition")
            if provenance == "measured-red" and not RUN_ID_RE.search(reason):
                problems.append(
                    f"{where} claims `measured-red` — a named run exercised "
                    f"this and contradicted it — and cites no run bundle id")
            if cli not in PROVIDER_IDS:
                problems.append(
                    f"{where} carves out {cli!r}, which is not a lane: "
                    f"provider row, so it subtracts nothing and hides nothing")
        assert not problems, "; ".join(problems)

    def test_a_carved_out_cli_is_in_no_hosts_fan_out(self):
        """The cell and the list cannot both be right.

        A CLI that is carved out AND still declared is the one state the
        subtraction cannot express: the union would pass on the list while
        the cell says the lane cannot carry it, and the runtime correlation
        in `test_container_host_lane.py` would go red for a provider the
        matrix had already excused in writing.
        """
        declared = _fan_out_hosts()
        clashes = [f"{cli} is carved out on {host} and still declared on "
                   f"{owner}"
                   for host, cli, _cell in _fan_out_carve_outs()
                   for owner, members in sorted(declared.items())
                   if cli in members]
        assert not clashes, "; ".join(clashes)

    def test_no_cli_is_carved_out_twice(self):
        """Two cells for one CLI are two reasons, and the second is unread."""
        seen: dict[str, str] = {}
        doubles = []
        for host, cli, _cell in _fan_out_carve_outs():
            if cli in seen:
                doubles.append(f"{cli} is carved out on {seen[cli]} and {host}")
            seen[cli] = host
        assert not doubles, "; ".join(doubles)

    def test_a_cli_missing_from_every_list_with_no_carve_out_still_fails(self):
        """The negative control on the property this guard just gained.

        A subtraction is only safe if the thing it subtracts is HARDER to
        write than the deletion it excuses. So: take a CLI the matrix really
        assigns, delete it from its host's list, and drive the SAME reduction
        the union guard runs. It must report the CLI missing. Add a carve-out
        cell for it, and the same reduction must fall silent — proving that
        the cell is what excuses the removal and that the deletion alone
        never could.

        Driven over a mutated deep copy through `_partition_shortfall`, never
        a second copy of the rule: a negative control that re-implements the
        thing it controls proves only that the test author is consistent.
        """
        matrix = _load_matrix()
        assigned = sorted({cli for members in _fan_out_hosts(matrix).values()
                           for cli in members})
        assert assigned, "no host declares a fan_out; nothing to control on"
        victim = assigned[0]

        mutated = copy.deepcopy(matrix)
        for entry in (mutated.get("hosts") or {}).values():
            if entry.get("fan_out"):
                entry["fan_out"] = [c for c in entry["fan_out"] if c != victim]
            (entry.get("fan_out_carve_outs") or {}).pop(victim, None)
        missing, _extra = _partition_shortfall(mutated)
        assert victim in missing, (
            f"deleting {victim} from every `fan_out` with no carve-out cell "
            f"left the union guard silent — the guard would then be "
            f"satisfiable by deletion, which is the exact hole the carve-out "
            f"reader was added to avoid"
        )

        host = next(h for h, members in _fan_out_hosts(matrix).items()
                    if victim in members)
        mutated["hosts"][host].setdefault("fan_out_carve_outs", {})[victim] = {
            "member": "not-applicable",
            "provenance": "measured-red",
            "task": "T3",
            "reason": "synthetic, 2026-09-22, runs/20260922T084502Z; retired "
                      "when this test stops mutating it",
        }
        missing_after, _ = _partition_shortfall(mutated)
        assert victim not in missing_after, (
            f"a complete carve-out cell for {victim} did not excuse its "
            f"absence, so the reader the union guard gained on 2026-09-22 is "
            f"not wired to the reduction"
        )

    def test_the_matrix_names_cli_ids_and_never_slugs(self):
        """Eight distinct model-arg spellings across the 16 rows; a hand-typed
        cell can get any of them wrong, and this lane has paid for exactly
        that once (opencode's `ProviderModelNotFoundError`)."""
        offenders = [f"{host}:{cli}"
                     for host, members in _fan_out_hosts().items()
                     for cli in members if ":" in cli or "/" in cli]
        assert not offenders, (
            f"{offenders} look like slugs. The matrix names ids; the spec is "
            f"built at read time from redirect.yaml's own `model_arg`."
        )

    @pytest.mark.parametrize("host,entry,row", HOST_ROWS, ids=HOST_ROW_IDS)
    def test_a_negative_auth_row_carries_neither_key(self, host, entry, row):
        """`remove_seed` rows are graded on the refusal landing BEFORE the
        first model call — upstream of any point a fan-out could act — and
        their runner must stay their own CLI or the executor's preflight sails
        past the very file the row removes."""
        if not row.get("remove_seed"):
            return
        assert "fan_out" not in row, (
            f"{row['id']} is a remove_seed row and declares a fan_out. Its "
            f"absence is asserted rather than the row merely skipped, so "
            f"'deliberately outside the partition' cannot be confused with "
            f"'somebody forgot'."
        )
        assert "runner" not in row, (
            f"{row['id']} is a remove_seed row and declares a runner. A "
            f"shared runner would sail straight past the credential file the "
            f"row removes and the row would assert nothing at all."
        )

    @pytest.mark.parametrize("host,entry,row", HOST_ROWS, ids=HOST_ROW_IDS)
    def test_no_rows_effective_list_leaves_its_own_hosts_subset(
            self, host, entry, row):
        effective = harness.host_fan_out(entry, row)
        if effective == harness.FAN_OUT_NATIVE:
            return
        allowed = set(_fan_out_hosts().get(host) or [])
        outside = sorted(set(effective) - allowed)
        assert not outside, (
            f"{row.get('id')} fans out to {outside}, which {host} does not "
            f"declare. A row-level override may NARROW a row; it may never "
            f"widen the partition or move a CLI between hosts."
        )

    @pytest.mark.parametrize("host,entry,row", HOST_ROWS, ids=HOST_ROW_IDS)
    def test_a_native_row_declares_it_rather_than_omitting_the_key(
            self, host, entry, row):
        """A native Task dispatch never crosses `/opt/mllm/bin`, so a fan-out
        list has no mechanism to act through — and a list-valued `fan_out` on
        such a row is an error reported by name."""
        declared = row.get("fan_out", entry.get("fan_out"))
        if row.get("remove_seed"):
            return
        assert declared is not None, (
            f"{row.get('id')} inherits no fan_out and declares none. Either "
            f"its host declares a list, or the row declares "
            f"`fan_out: native`."
        )

    @pytest.mark.parametrize("host,entry,row", HOST_ROWS, ids=HOST_ROW_IDS)
    def test_the_rows_runner_is_seeded_and_can_actually_write(
            self, host, entry, row):
        """T0a's two invariants, in the one place Docker is not needed.

        A runner with no seeded home fails preflight for a reason that has
        nothing to do with the phase under test; a runner that cannot write
        fails the write dispatch with a `ProfileUnsupported` refusal that
        reads as a redirection problem and is nothing of the kind.
        """
        runner = harness.host_runner_cli(host, entry, row)
        assert runner, f"{row.get('id')} resolves to no write runner at all"
        if row.get("remove_seed"):
            assert runner == host
            return
        assert runner in PROVIDER_IDS, (
            f"{row.get('id')}'s runner {runner} is not a lane: provider row, "
            f"so nothing redirects it at the stub"
        )
        assert _declares_write(runner), (
            f"{row.get('id')}'s runner {runner} declares no agent-write "
            f"profile of its own"
        )
        assert not _probes_a_sandbox(runner), (
            f"{row.get('id')}'s runner {runner} needs a kernel feature to "
            f"dispatch its agent-write profile. That is the §1.4 regression: "
            f"inside the image codex's probe spawns bwrap, Docker's default "
            f"seccomp denies CLONE_NEWUSER, and the verdict is `unavailable`."
        )
        assert runner in harness.host_helpers(host, entry, row), (
            f"{row.get('id')}'s runner {runner} is not in its own helper set, "
            f"so the entrypoint seeds it no home"
        )

    def test_a_hand_built_host_row_seeds_its_runner_too(self):
        """The credential-free fixtures build `Row` by hand, not from the matrix.

        Discovery, documented-invocation and the codex package probes name
        `runner="opencode:stub/stub-model"` and no helper. Before `row_helpers`
        the entrypoint seeded no `/work/homes/opencode` for them, so row 5's
        runner preflight read `opencode auth list` against an empty home, saw
        `0 credentials`, and refused every one of those rows
        `not_authenticated` (measured on opencode 1.18.32, 2026-09-25).
        """
        bare = harness.Row(row_id="discovery-codex", lane="host", host="codex",
                           creds="none", spec=["opencode:stub/stub-model"],
                           runner="opencode:stub/stub-model")
        assert harness.row_helpers(bare) == ["opencode"]
        # A matrix-built row's list already carries its runner: untouched, so
        # its rendered argv stays byte-identical.
        listed = harness.Row(row_id="r", lane="host", host="codex",
                             creds="openrouter", runner="opencode:stub/stub-model",
                             helpers=["pi", "opencode", "grok"])
        assert harness.row_helpers(listed) == ["pi", "opencode", "grok"]
        # The provider lane is not a host row: its helpers are its own business.
        provider = harness.Row(row_id="p", lane="provider", cli="codex",
                               runner="opencode:stub/stub-model")
        assert harness.row_helpers(provider) == []


# --------------------------------------------------------------------------- #
# T3a: the fan-out list has to survive every layer between the matrix row and
# the rendered config, and four of them transported a scalar. Every one of
# those layers renders with NO Docker, so every one of them is checked here.
#
# The dangerous failure is not a loud one. The override is deep-merged onto
# base (`provider_registry.load_config` -> `_deep_merge`), so a document that
# fails to parse — or that lands a string where a list belongs — falls straight
# through to base's TWELVE specs across five providers, and the row would then
# look like a fan-out while running someone else's roster. Hence the guard over
# the MERGED config below, not only over the rendered text.
# --------------------------------------------------------------------------- #

#: Every place a selection is made in the rendered override: the two global
#: lists, and both keys of each `defaults.modes.<mode>` entry. 22 positions.
def _selection_lists(document: dict) -> dict[str, list]:
    defaults = document.get("defaults") or {}
    found: dict[str, list] = {
        "defaults.models": defaults.get("models"),
        "defaults.quick_models": defaults.get("quick_models"),
    }
    for mode, entry in (defaults.get("modes") or {}).items():
        found[f"defaults.modes.{mode}.models"] = (entry or {}).get("models")
        found[f"defaults.modes.{mode}.quick"] = (entry or {}).get("quick")
    return found


_RUNNER = "opencode:stub/stub-model"


class TestTheRenderedOverrideCarriesTheWholeFanOut:

    def test_a_one_spec_render_still_selects_exactly_that_spec(self):
        """Parse-compatibility, NOT byte identity.

        Byte identity is given up deliberately: the slot lost its quotes and
        its brackets so a list could be rendered into it at all, and the flow
        list re-renders even a one-element list. What must not move is any
        existing row's *selection*, and that is a property of the parsed
        document.
        """
        document = yaml.safe_load(
            seed_workspace.render_override(["codex:gpt-5.5"], _RUNNER))
        for key, value in _selection_lists(document).items():
            assert value == ["codex:gpt-5.5"], f"{key} renders {value!r}"

    def test_a_multi_spec_render_is_loadable_yaml_in_the_given_order(self):
        specs = ["aider:openai/stub-model", "muse:muse-spark-1.2",
                 "pi:stub/stub-model"]
        document = yaml.safe_load(seed_workspace.render_override(specs, _RUNNER))
        assert document["defaults"]["models"] == specs
        assert document["defaults"]["quick_models"] == specs

    def test_every_selection_key_widens_together(self):
        """Walked over the PARSED document, never grepped from the template.

        `defaults.modes.<mode>` is consulted *before* the global lists and
        `--quick` reads `defaults.quick_models`, so a mode added to the
        template without the list-aware slot would leave a row fanning out on
        paper and calling `models[0]` in fact — and the disjoint/union guard
        would then assert a partition that never executes.
        """
        specs = ["aider:openai/stub-model", "muse:muse-spark-1.2"]
        document = yaml.safe_load(seed_workspace.render_override(specs, _RUNNER))
        lists = _selection_lists(document)
        assert len(lists) == 22, (
            f"expected 22 selection positions, walked {len(lists)}: "
            f"{sorted(lists)}"
        )
        wrong = {key: value for key, value in lists.items() if value != specs}
        assert not wrong, f"these selection keys did not widen: {wrong}"

    def test_the_runner_keys_stay_singular_and_are_not_widened(self):
        """`driver.subagent_runner` is one slug and write concurrency stays 1.

        Reviewed in the same pass as the spec sites precisely so the edit
        cannot widen the write runner by accident.
        """
        document = yaml.safe_load(
            seed_workspace.render_override(["a:b", "c:d"], _RUNNER))
        driver = document["driver"]
        assert driver["subagent_runner"] == _RUNNER
        assert driver["subagent_runner_cheap"] == _RUNNER
        assert driver["max_concurrent_write_subagents"] == 1

    def test_the_merged_config_never_falls_through_to_bases_roster(self):
        """The guard over the merged config rather than the rendered text.

        A malformed render is the dangerous failure, not a loud one — it
        deep-merges to base's twelve specs across five providers, which is the
        exact outcome the template's own header says it exists to prevent.
        """
        specs = ["aider:openai/stub-model", "muse:muse-spark-1.2",
                 "pi:stub/stub-model", "omp:stub/stub-model"]
        rendered = yaml.safe_load(
            seed_workspace.render_override(specs, _RUNNER))
        merged = _deep_merge(load_base_config(), rendered)
        lists = _selection_lists(merged)
        wrong = {key: value for key, value in lists.items() if value != specs}
        assert not wrong, (
            f"the merged config does not carry the row's fan-out at "
            f"{sorted(wrong)} — base's own selection is leaking into the row"
        )

    def test_base_ships_no_defaults_modes_key_the_template_misses(self):
        """The guard the template's own header CLAIMS and no test implemented.

        Verified 2026-09-22: no test read `providers.tmpl.yaml` at all. A mode
        key added to base — or added to the template without the list-aware
        slot — would otherwise fan a row out to `models[0]` mid-run.
        """
        base_modes = set(((load_base_config().get("defaults") or {})
                          .get("modes") or {}))
        rendered = yaml.safe_load(
            seed_workspace.render_override(["a:b"], _RUNNER))
        template_modes = set(rendered["defaults"]["modes"])
        missing = sorted(base_modes - template_modes)
        assert not missing, (
            f"base ships defaults.modes entries the container template does "
            f"not override: {missing}. A mode entry wins outright over the "
            f"global lists, and `modes: {{}}` deep-merges as a no-op rather "
            f"than clearing a base entry — the only way to beat one is to "
            f"write a leaf list of our own."
        )


class TestTheTwoHostTiersSelectDisjointRowSets:
    """§6's two collection checks — plain CI, no Docker, no key.

    Both directions are needed, and they fail differently:

    * a **missing mark** makes the cheap `make e2e-container-host` target
      quietly ~$3-5 more expensive every pass — the native phase rows are
      anchored on `claude-code-apply-native`'s measured $0.586302 pass, an
      order of magnitude above the $0.008-$0.036 the executor hosts' rows
      cost;
    * a **typo'd expression** makes the dedicated target collect zero, which
      `pytest` exits **5** for — the same empty-selection trap
      `pyproject.toml` already records for `-m package`.

    Asserted by running `pytest --collect-only` in a subprocess, because
    "what does pytest select" cannot be answered from inside the run pytest
    already selected.
    """

    HOST_MODULE = TESTS_DIR / "test_container_host_lane.py"
    PHASE_PREFIX = "claude-code-phase-"

    def _collect(self, expression: str) -> list[str]:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q",
             "-m", expression, str(self.HOST_MODULE)],
            cwd=str(TESTS_DIR.parent), capture_output=True, text=True,
            timeout=300,
        )
        assert completed.returncode in (0, 5), (
            f"collection failed for {expression!r}:\n"
            f"{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )
        return [line for line in completed.stdout.splitlines() if "::" in line]

    def test_the_ordinary_target_collects_no_phase_row(self):
        selected = self._collect(
            "container and live and not live_native_phase")
        leaked = [line for line in selected if self.PHASE_PREFIX in line]
        assert not leaked, (
            "the cheap host tier collects native phase rows, so every pass of "
            "`make e2e-container-host` now costs ~$3-5 more than its measured "
            "$0.1610:\n" + "\n".join(leaked[:10])
        )
        assert selected, (
            "the ordinary host expression collects nothing at all — pytest "
            "exits 5 for that, and the target would be green over an empty "
            "lane"
        )

    def test_the_dedicated_target_collects_exactly_those_rows(self):
        selected = self._collect("container and live and live_native_phase")
        assert selected, (
            "the dedicated phase expression collects nothing. pytest exits 5 "
            "for an empty selection, so a typo here is a target that grades "
            "nothing while looking like it ran."
        )
        strays = [line for line in selected if self.PHASE_PREFIX not in line]
        assert not strays, (
            "the dedicated phase expression collects rows that are not phase "
            "rows, so the expensive tier is paying for the cheap one's "
            "coverage:\n" + "\n".join(strays[:10])
        )


class TestTheContainerModulesAreGated:
    """The marker, the skipif, and the expression CI actually types."""

    def test_there_is_at_least_one_container_module(self):
        assert CONTAINER_MODULES, "no test_container_*.py modules found"

    @pytest.mark.parametrize("module", CONTAINER_MODULES,
                             ids=[m.stem for m in CONTAINER_MODULES])
    def test_each_module_carries_the_marker_and_the_skipif(self, module):
        """Source-level, so it holds on a machine that has no daemon to probe."""
        source = module.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assignments = [
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "pytestmark" for t in node.targets)
        ]
        assert assignments, f"{module.name} declares no module-level pytestmark"
        rendered = ast.unparse(assignments[-1].value)
        assert "pytest.mark.container" in rendered, (
            f"{module.name}'s pytestmark does not carry the container marker: "
            f"{rendered}"
        )
        assert "requires_docker()" in rendered, (
            f"{module.name}'s pytestmark does not carry the no-Docker skipif: "
            f"{rendered}. Without it the module fails rather than skips on a "
            f"machine with no daemon."
        )

    def test_the_ci_selection_collects_nothing_from_them(self):
        """The one assertion the other two cannot make.

        An explicit `-m` on the command line *replaces* `addopts` rather than
        composing with it, which is exactly how a container-marked module ends up
        collected on a runner with no daemon. This types the expression the
        workflow types.
        """
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q",
             "-m", "not live and not container",
             *[str(m) for m in CONTAINER_MODULES]],
            cwd=str(TESTS_DIR.parent), capture_output=True, text=True, timeout=300,
        )
        # Exit 5 is "no tests collected", which is the passing outcome here.
        assert completed.returncode in (0, 5), (
            f"collection failed:\n{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )
        selected = [
            line for line in completed.stdout.splitlines()
            if "::" in line and "test_container_" in line
        ]
        assert not selected, (
            "the default CI selection collects container-marked tests:\n"
            + "\n".join(selected[:10])
        )
