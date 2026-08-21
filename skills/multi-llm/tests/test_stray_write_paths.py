"""What a host-lane row's `expect.paths` still refuses — checked in CI (R2).

`tests/container/host_matrix.yaml` gives `claude-code-apply-native` the only
byte-level assertion either lane has through a real host: the SHA census of the
seeded workspace moved, and it moved **only** inside the trees the row declares.
The negative half of that pair —
`test_container_host_lane.test_no_write_landed_outside_the_declared_paths` — is
`live`-marked beneath a module-level `container` gate, so it answers only inside a
credentialed Docker run. Nobody widening a `paths:` glob in a PR is going to spend
a host-lane run to discover they widened it too far, and a glob that admits
everything fails nothing while still reading like a guard.

So the predicate itself lives in `tests/container/harness.stray_paths` — extracted
from that test body, behaviour unchanged — and this module pins it in **both**
directions on inputs that are fixed rather than measured on the day. No
credential, no daemon, no network: `harness` is imported for one pure function,
exactly as `test_capability_promotions.py` imports it for the record path.

THE MATCHER IS `fnmatch`, AND ITS `*` CROSSES `/`. That is the fact every case
below turns on: `src/**` and `src/*` are the same predicate, so a declaration
cannot be made stricter by spelling more stars, and `tests/test_*.py` admits
`tests/unit/deep/test_x.py` unless something refuses it. It does not — which is
why that path is in the stray direction here.

WHY THE DECLARATION IS A SHAPE AND NOT A FILENAME. `runs/20260820T093838Z` ran
`claude-code-apply-native` and its census diff was exactly three entries:
`M src/security.py`, `M src/auth.py`, `A tests/test_security.py`. The third comes
from the fifth issue in
`tests/fixtures/e2e/responses/code_review/container_issues.json`, which names
`src/security.py` in its `file` field but asks for a NEW TEST MODULE and states no
target filename anywhere. `tests/test_rate_limiter.py`, a split across two files,
or appending to the seeded `tests/test_auth.py` are equally faithful readings of
the same instruction, so `tests/test_*.py` is declared and that run's filename is
deliberately NOT pinned — a pin would trade today's red row for a flaky one.

**The filename is deliberately not `test_container_*`** — the same reason
`test_capability_promotions.py`, `test_redirect_matrix_claims.py` and
`test_golden_container_bundle.py` all give. That glob is what
`tests/container/context_manifest.txt` ships into the image and what the
marker-gating check enumerates; a plain-CI module answering to it would be shipped
for nothing and would fail that marker assertion. A guard that only runs where
Docker runs is a guard that does not run on the PR where someone widens the glob.
"""

from __future__ import annotations

import pytest
import yaml

from .container import harness

#: What `claude-code-apply-native` declares, and the single subject of this
#: module. Read from the matrix rather than retyped: a widening that lands in the
#: file and not here would leave this module pinning a declaration nothing uses.
ROW_ID = "claude-code-apply-native"

#: The census diff `runs/20260820T093838Z` measured for that row, source-filtered
#: exactly as `_source_delta` filters it (`plans/**` and `.multi-llm/**` dropped
#: as workflow output). Three entries, no fourth: the row was one file wide of its
#: old `["src/**"]`, not a set of files.
MEASURED = ("src/security.py", "src/auth.py", "tests/test_security.py")

#: What the declaration must keep refusing, and why each one is here:
#:
#: * `tests/conftest.py`      — inside `tests/`, not a test module. The reason
#:                              `tests/**` was rejected: `fnmatch`'s `*` crosses
#:                              `/`, so `tests/**` is `tests/*` and would admit it.
#: * `tests/fixtures/data.json` — a fixture tree an apply phase has no business in.
#: * `tests/unit/deep/test_x.py` — the `/`-crossing case stated outright: it ends
#:                              in `test_*.py` and is still a stray, because the
#:                              pattern is anchored at `tests/`.
#: * `README.md`              — outside both trees entirely.
STRAYS = ("tests/conftest.py", "tests/fixtures/data.json",
          "tests/unit/deep/test_x.py", "README.md")


@pytest.fixture(scope="module")
def declared() -> list[str]:
    """`expect.paths` as the matrix really declares it, by row id."""
    matrix = yaml.safe_load(
        (harness.HERE / "host_matrix.yaml").read_text(encoding="utf-8"))
    for entry in (matrix.get("hosts") or {}).values():
        for row in entry.get("rows") or []:
            if row.get("id") == ROW_ID:
                paths = (row.get("expect") or {}).get("paths")
                assert paths, f"{ROW_ID} declares no expect.paths"
                return list(paths)
    pytest.fail(f"host_matrix.yaml names no row {ROW_ID!r}")


class TestTheDeclarationAdmitsWhatThePhaseReallyWrites:
    """The positive direction: a widening that fails the real run is not a fix."""

    def test_the_measured_census_diff_yields_no_strays(self, declared):
        """The whole of `runs/20260820T093838Z`'s diff, not just the named file.

        This is the acceptance R2 states in the terms it states it: the
        declaration must be one that **would have passed the run before the run
        happened**, over every entry the run produced.
        """
        assert harness.stray_paths(MEASURED, declared) == []

    @pytest.mark.parametrize("name", MEASURED)
    def test_each_measured_name_individually(self, declared, name):
        assert harness.stray_paths([name], declared) == []

    @pytest.mark.parametrize("name", [
        # Every filename the fifth issue's instruction can reasonably produce.
        # None of them is the one that run happened to write, and that is the
        # point: the declaration is a property of the task, not a sample from a
        # model. `test_auth.py` is the seeded module the fixture's own
        # description points at (`seed_workspace.py` really creates it), so a
        # model appending its cases there rather than creating a file has done
        # the same legitimate work — which is also why there is no
        # "additions only" qualifier on the declaration.
        "tests/test_rate_limiter.py",
        "tests/test_auth.py",
        "tests/test_security.py",
    ])
    def test_the_other_filenames_that_instruction_could_produce(
            self, declared, name):
        assert harness.stray_paths([name], declared) == []

    def test_a_two_file_split_is_admitted_whole(self, declared):
        """A model that splits its cases across two modules still passes."""
        assert harness.stray_paths(
            ["src/security.py", "tests/test_rate_limiter.py",
             "tests/test_security.py"], declared) == []


class TestTheDeclarationStillRefusesEverythingElse:
    """The negative direction — the discrimination the row exists to defend.

    Inside-vs-outside the declared trees. A declaration that stopped refusing
    these would still be green on every run and would be asserting nothing, which
    is the failure this module is here to make loud on a PR.
    """

    @pytest.mark.parametrize("name", STRAYS)
    def test_each_stray_is_reported(self, declared, name):
        assert harness.stray_paths([name], declared) == [name]

    def test_every_stray_is_reported_together(self, declared):
        assert harness.stray_paths(STRAYS, declared) == sorted(STRAYS)

    def test_a_stray_is_reported_even_beside_the_expected_write(self, declared):
        """A stray fails the row even when the expected file also changed.

        The live assertion's own message says so; this pins it, because a
        predicate that only looked at whether ANY declared path matched would
        pass this input and the row would stop catching escapes.
        """
        assert harness.stray_paths(
            list(MEASURED) + ["README.md"], declared) == ["README.md"]

    def test_the_nested_test_module_is_a_stray_because_star_crosses_slash(
            self, declared):
        """`fnmatch`, not `pathlib`: `tests/test_*.py` is anchored at `tests/`.

        Stated as its own case because it is the one result a reader is most
        likely to guess wrong in either direction — `**` does not mean "recursive"
        here, and `*` does not stop at `/`.
        """
        assert harness.stray_paths(["tests/unit/deep/test_x.py"], declared) == [
            "tests/unit/deep/test_x.py"]
        # ...and the same star that refuses it is why `src/**` reaches a nested
        # source file without a second pattern.
        assert harness.stray_paths(["src/inner/deep/mod.py"], declared) == []


class TestThePredicateItself:
    """`stray_paths` on inputs of this module's own choosing, not the matrix's."""

    def test_nothing_moved_is_no_strays(self, declared):
        assert harness.stray_paths([], declared) == []

    def test_an_empty_declaration_makes_everything_stray(self):
        """A row declaring nothing is not a row admitting everything.

        The live test skips before it gets here (`if not declared: pytest.skip`),
        so this pins the helper's own behaviour rather than the row's: an empty
        pattern list matches nothing, which is the safe direction for a predicate
        whose job is to refuse.
        """
        assert harness.stray_paths(MEASURED, []) == sorted(MEASURED)

    def test_the_result_is_sorted_and_deduplicated_by_input(self, declared):
        assert harness.stray_paths(
            ["README.md", "tests/conftest.py"], declared) == [
            "README.md", "tests/conftest.py"]

    def test_declared_may_be_any_iterable(self, declared):
        """The live caller passes a list; a generator must not be consumed early.

        `stray_paths` materializes `declared` before the per-name scan, so a
        one-shot iterable does not silently stop refusing after the first name —
        the failure mode would be a second stray going unreported.
        """
        assert harness.stray_paths(
            STRAYS, iter(list(declared))) == sorted(STRAYS)
