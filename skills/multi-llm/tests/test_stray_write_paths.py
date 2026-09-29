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

FOUR PREDICATES NOW, NOT ONE. The module grew as the live module extracted
more of its grading into `harness`, and every one of them is here for the same
reason `stray_paths` is — it answers only inside a credentialed Docker run, and
a PR that weakens it is read on plain CI. `graded_delta`/`workflow_bookkeeping`
separate a plan edit from the workflow's own bookkeeping; `fan_out_correlation`
plus `error_envelope_reason` decide whether an assigned provider really ran;
`tool_artifact_paths` is the one deliberate LOOSENING of a byte-level net, and
its class is weighted accordingly — the case that matters most there is the
product write the net must still catch on a row that carries the exemption.

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


#: The bytes a `pre_commands` leg leaves behind, taken from a real census
#: (`runs/20260920T091849Z/chain-omp`, 2026-09-22). Every one is inside the
#: plan's OUTPUT DIRECTORY `plans/auth-feature/`, which is what makes them
#: distinguishable from the plan FILE beside it.
LEG_OUTPUT = (
    "plans/auth-feature/review-plan/report.md",
    "plans/auth-feature/review-plan/grouped.json",
    "plans/auth-feature/review-plan/validation.json",
    "plans/auth-feature/apply-suggestions/write_batch_report.json",
    "plans/auth-feature/state.json",
    ".multi-llm/runs/20260920T091849Z/run.json",
)

#: The one artifact `apply-suggestions` exists to edit.
PLAN_FILE = "plans/auth-feature.md"

#: The one artifact `apply-task-suggestions` exists to edit, and the case that
#: makes the declaration outrank the prefix rule. `utils/tasks_file.py:137-138`
#: resolves the tasks document to
#: `{plan_dir}/{sanitize_prefix(plan)}/tasks/tasks.md`, so it sits TWO levels
#: under `plans/` — inside the plan's own output directory, exactly where
#: `workflow_bookkeeping` says "not work". It is still that row's whole grade.
TASKS_FILE = "plans/auth-feature/tasks/tasks.md"


class TestTheGraderSeparatesAPlanEditFromWorkflowBookkeeping:
    """§3.4's grader change, pinned offline — no daemon, no credential.

    `_source_delta` drops every path under `plans/` and `.multi-llm/`. That is
    right for `apply-code-fixes` touching `src/**` and `tests/test_*.py`; it is
    exactly wrong for `apply-suggestions`, whose entire intended effect **is**
    an edit to the plan file under `plans/`. A row graded by that predicate
    could rewrite the plan, or fail to, and pass either way.

    Two regression cases, in both directions, exactly as the plan specifies:
    one where the only change is a plan edit and the row must PASS, one where
    the only change was made by the pre-command leg and the row must still
    FAIL as a no-op.
    """

    def test_a_plan_only_edit_is_graded_for_the_row_that_declares_it(self):
        """Case 1: `apply-suggestions` rewrote its plan, and nothing else."""
        owned = [PLAN_FILE]
        graded = harness.graded_delta([PLAN_FILE], owned)
        assert graded == [PLAN_FILE], (
            "the plan file is apply-suggestions' whole intended effect; a "
            "grader that filters it cannot tell the phase working from the "
            "phase doing nothing"
        )
        assert harness.stray_paths(graded, owned) == []

    def test_the_tasks_document_is_graded_for_the_row_that_declares_it(self):
        """Case 1b: `apply-task-suggestions`, whose artifact is nested.

        `plans/auth-feature/tasks/tasks.md` is two levels under `plans/`, so
        `workflow_bookkeeping` calls it output — and it is
        `claude-code-phase-apply-task-suggestions`' SOLE `editable_artifacts`
        entry, under `bytes_must_change: true`. A grader that filtered it
        before reading the declaration would leave that row unable to pass or
        fail on bytes whatever the host did: header finding (3) recurring on a
        third row. The declaration is therefore consulted first.
        """
        owned = [TASKS_FILE]
        assert harness.graded_delta([TASKS_FILE], owned) == [TASKS_FILE], (
            "the tasks document is apply-task-suggestions' whole intended "
            "effect; a grader that drops it for living inside the plan's "
            "output directory makes bytes_must_change unsatisfiable"
        )
        assert harness.stray_paths([TASKS_FILE], owned) == []
        # ...and the leg's output beside it is still bookkeeping, so the row
        # is graded on the document alone rather than on the phase dirs.
        assert harness.graded_delta([*LEG_OUTPUT, TASKS_FILE], owned) == [
            TASKS_FILE]

    def test_the_tasks_document_is_bookkeeping_for_a_row_that_does_not_own_it(
            self):
        """The other direction: nesting is only forgiven where it is declared.

        `apply-suggestions` owns the plan FILE, not the tasks document, so on
        that row a tasks.md write stays inside the plan's output directory and
        is not credited as the row's work — the rescue is scoped to the
        declaration, not a hole in the predicate.
        """
        assert harness.graded_delta([TASKS_FILE], [PLAN_FILE]) == []
        assert harness.graded_delta(
            [TASKS_FILE], ["src/**", "tests/test_*.py"]) == []
        assert harness.graded_delta([TASKS_FILE], []) == []

    def test_a_pre_command_leg_only_change_is_still_a_no_op(self):
        """Case 2: the leg wrote its artifacts and the host wrote nothing."""
        graded = harness.graded_delta(LEG_OUTPUT, [PLAN_FILE])
        assert graded == [], (
            f"the leg's own output was credited to the host: {graded}. Every "
            f"one of these is inside the plan's output directory or the run "
            f"registry — workflow bookkeeping, and none of it work the row did."
        )

    def test_the_plan_file_is_bookkeeping_for_a_row_that_does_not_own_it(self):
        """`finalize_tracking` writes a back-reference into it on every phase.

        So on `apply-code-fixes`, whose declared artifacts are `src/**` and
        `tests/test_*.py`, the plan file moving is bookkeeping — while on
        `apply-suggestions` the very same change is the grade. A path
        predicate alone cannot tell those apart, which is why the declaration
        is an input to `graded_delta` and not only a filter applied after it.
        """
        code_owned = ["src/**", "tests/test_*.py"]
        assert harness.graded_delta([PLAN_FILE], code_owned) == []
        assert harness.graded_delta(
            [PLAN_FILE, "src/auth.py"], code_owned) == ["src/auth.py"]

    def test_a_real_escape_is_still_caught_through_the_new_predicate(self):
        """The negative direction the whole grader exists for."""
        assert harness.graded_delta(
            [*LEG_OUTPUT, "README.md"], [PLAN_FILE]) == ["README.md"]

    @pytest.mark.parametrize("name,expected", [
        (".multi-llm/runs/x/run.json", True),
        ("plans/auth-feature/state.json", True),
        ("plans/auth-feature/review-plan/report.html", True),
        # The predicate itself still calls the tasks document output — it is
        # `graded_delta`'s declaration check, running FIRST, that rescues it
        # for the one row that declares it.
        ("plans/auth-feature/tasks/tasks.md", True),
        ("plans/auth-feature.md", False),
        ("plans/other-plan.md", False),
        ("src/auth.py", False),
        ("README.md", False),
    ])
    def test_the_bookkeeping_predicate_itself(self, name, expected):
        """One directory level under `plans/` is a plan FILE; two is output.

        Measured against a real census on 2026-09-22: 25 added paths on
        `runs/20260920T091849Z/chain-omp`, every one under
        `plans/auth-feature/`, with `plans/auth-feature.md` a sibling of that
        directory rather than a child of it.
        """
        assert harness.workflow_bookkeeping(name) is expected


#: One host's fan-out, as the partition assigns it.
FAN_OUT = ("opencode", "gemini", "aider", "copilot", "kimi")


def _result_name(cli):
    """The review phases' result filename for `cli`, as the product writes it.

    `harness.result_file_names` rather than a hand-spelled stem: these names
    are the fixture's whole input, and one typed by hand here would test a
    filename shape nothing produces.
    """
    return harness.result_file_names(cli)[0]


def _complete_evidence():
    """What a fully-covered fan-out row's three extracted name sets look like."""
    return {
        "invoked": set(FAN_OUT),
        "reached": set(FAN_OUT),
        "produced": [_result_name(cli) for cli in FAN_OUT],
    }


class TestTheFanOutCorrelationIsFalsifiable:
    """§3.3's negative control, as a plain-CI fixture over recorded evidence.

    The union guard proves the partition is *declared*; the per-provider
    correlation proves it *executed*. But a correlation that matched on the
    wrong field, or that iterated an empty list, reads exactly like complete
    coverage — so the control has to exist, and the plan says so: *a row in
    which one assigned provider is absent or fails while the others succeed
    must go red, and must name the missing provider.*

    A fixture here rather than a container row, because that is what §3.3
    itself recommends: cheaper, and it needs no key.
    """

    def test_a_complete_fan_out_reports_nothing_missing(self):
        assert harness.fan_out_correlation(
            FAN_OUT, **_complete_evidence()) == {}

    @pytest.mark.parametrize("absent", FAN_OUT)
    def test_one_absent_provider_goes_red_and_is_named(self, absent):
        """The control proper: one provider drops out, the rest succeed."""
        evidence = _complete_evidence()
        evidence["invoked"].discard(absent)
        evidence["reached"].discard(absent)
        evidence["produced"] = [name for name in evidence["produced"]
                                if name != _result_name(absent)]
        missing = harness.fan_out_correlation(FAN_OUT, **evidence)
        assert set(missing) == {absent}, (
            f"dropping {absent} should implicate {absent} and nothing else; "
            f"got {missing}"
        )
        assert set(missing[absent]) == {"invocation", "stub-session", "result"}

    def test_a_provider_that_ran_but_produced_no_result_is_red(self):
        """The shim ran and the model surface answered — and no file landed.

        Distinguishable from never being invoked, which is why the three
        facts are reported separately rather than as one boolean.
        """
        evidence = _complete_evidence()
        evidence["produced"] = [name for name in evidence["produced"]
                                if name != _result_name("copilot")]
        missing = harness.fan_out_correlation(FAN_OUT, **evidence)
        assert missing == {"copilot": ["result"]}

    def test_a_provider_that_died_in_argument_parsing_is_red(self):
        """opencode's `ProviderModelNotFoundError` shape: invoked, no session.

        It produced zero stub records while the row's total was non-zero —
        exactly what a count-only assertion cannot see.
        """
        evidence = _complete_evidence()
        evidence["reached"].discard("opencode")
        evidence["produced"] = [name for name in evidence["produced"]
                                if name != _result_name("opencode")]
        missing = harness.fan_out_correlation(FAN_OUT, **evidence)
        assert missing == {"opencode": ["stub-session", "result"]}

    def test_an_empty_fan_out_reports_nothing_rather_than_passing_vacuously(self):
        """A carve-out row has no partition to prove; the CALLER skips it.

        Pinned so the predicate's own behaviour on an empty list is a
        deliberate choice rather than an accident — the vacuous-green risk is
        handled by the caller's `fan_out: native` / `remove_seed` skip and by
        the plain-CI union guard, not by this function inventing a failure.
        """
        assert harness.fan_out_correlation([], invoked=set(), reached=set(),
                                           produced=[]) == {}

    def test_an_all_empty_evidence_set_implicates_every_provider(self):
        """The iterated-an-empty-list bug, caught rather than read as green."""
        missing = harness.fan_out_correlation(
            FAN_OUT, invoked=set(), reached=set(), produced=[])
        assert set(missing) == set(FAN_OUT)

    def test_a_siblings_result_file_does_not_satisfy_a_provider(self):
        """The substring collision, which the exactly-N rule cannot survive.

        `pi` is a substring of `copilot_stub-model.json`, so under the old
        `cli in name` predicate a `pi` that produced nothing was credited
        with copilot's file and reported as fully covered. The two sit on
        different hosts today — the defect was latent, not live — but a
        partition is re-cut by editing one list in `host_matrix.yaml`, and
        that is not an edit anybody would expect to weaken a grader.
        """
        pair = ("pi", "copilot")
        missing = harness.fan_out_correlation(
            pair, invoked=set(pair), reached=set(pair),
            produced=[_result_name("copilot")])
        assert missing == {"pi": ["result"]}, (
            f"copilot's {_result_name('copilot')} is not pi's result file; "
            f"got {missing}")

    def test_asks_markdown_answer_counts_as_that_providers_result(self):
        """`ask` writes `answer_<stem>.md`, and the caller collects it.

        The host lane's `produced` list is a `.json`/`.md` union precisely so
        an `ask` row can be graded, so the exact-name rule has to admit both
        names the product writes — tightening to the `.json` stem alone would
        report every `ask` provider as having produced nothing.
        """
        answers = [harness.result_file_names(cli)[1] for cli in FAN_OUT]
        assert all(name.startswith("answer_") and name.endswith(".md")
                   for name in answers), answers
        assert harness.fan_out_correlation(
            FAN_OUT, invoked=set(FAN_OUT), reached=set(FAN_OUT),
            produced=answers) == {}

    def test_the_expected_names_are_the_spec_the_fan_out_rendered(self):
        """`result_file_names` is `provider_spec` + the product's sanitizer.

        Pinned so the two cannot drift apart silently: the row renders
        `provider_spec(cli)` into the container's config, and the grader has
        to expect the filename THAT spec produces rather than one built from
        the CLI id alone (`pi` renders `pi:stub/stub-model`, whose stem is
        `pi_stub_stub-model`, not `pi`).
        """
        from utils.json_extractor import sanitize_model_name

        for cli in (*FAN_OUT, "pi", "copilot"):
            stem = sanitize_model_name(harness.provider_spec(cli))
            assert harness.result_file_names(cli) == (
                f"{stem}.json", f"answer_{stem}.md")


#: The one third-party artifact any recorded bundle has produced, and the whole
#: subject of the class below. Measured on four of them: it appears on every row
#: aider ran on — `runs/20260922T084502Z`'s four codex-host rows (aider is in
#: codex's fan-out) and `chain-aider` in `runs/20260920T105458Z` and
#: `runs/20260921T224840Z` — and on no other row in any of them. On
#: `codex-apply-executor` it was the ENTIRE census delta and failed the row with
#: `bytes moved on a row declared 'clean-failure'`.
AIDER_CACHE = ".aider.tags.cache.v4/cache.db"

#: A codex-host row's measured host-bracketed delta, source-filtered, from
#: `runs/20260922T084502Z/codex-ask`: the leg's and the phase's own output under
#: `plans/`, plus that one file. The `plans/` members are abbreviated — the
#: prefix rule does not read past the second separator — and `codex-ask`'s real
#: delta carried fifteen of them.
CODEX_ROW_DELTA = (
    AIDER_CACHE,
    "plans/auth-feature/ask/what-does-this-plan-change-8a3c7cf7/answers.md",
    "plans/auth-feature/ask/what-does-this-plan-change-8a3c7cf7/.status.json",
)

#: The product write planted into that row below. `src/auth.py` because it is
#: one of the three files the ONE applied-byte row in the same bundle really
#: moved (`MEASURED`) — a write of exactly the shape this net exists to catch.
PLANTED_WRITE = "src/auth.py"


def _bytes_graded(names, *, clis, declared=None):
    """The census paths a byte-graded row is answerable for.

    The two steps `test_container_host_lane._graded_delta` composes, in its
    order, through the two harness predicates it calls — so this module grades
    the composition and not only its halves. (That module's undeclared branch
    reaches the prefix rule through its own `_source_delta`; for these inputs
    `graded_delta` with no declaration is the same predicate, and the tool
    subtraction it gained is applied to both branches alike.)
    """
    kept = harness.graded_delta(names, declared)
    exempt = set(harness.tool_artifact_paths(kept, clis, declared))
    return sorted(name for name in kept if name not in exempt)


class TestAToolsOwnCacheIsNotAProductWrite:
    """The tool-artifact category, pinned in both directions on plain CI.

    This is a deliberate LOOSENING of a byte-level safety net, so the class
    that grades it is weighted towards what the net must still refuse. The
    exemption is declared per CLI in `redirect.yaml`, scoped to the row's own
    fan-out, and outranked by the row's own `editable_artifacts` — and each of
    those three is what one of the tests below is for.
    """

    def test_the_pattern_is_declared_in_the_matrix_and_not_in_the_grader(self):
        """`redirect.yaml` is the source, the way `seed_files` is.

        Asserted rather than assumed, because a path hardcoded in `harness.py`
        would make every test below pass while removing the property the whole
        design rests on: that a reviewer meets this claim about a vendor's
        binary on the row that describes that binary.
        """
        assert "aider" in FAN_OUT, (
            "this class reuses FAN_OUT as the codex row's partition; if aider "
            "has moved hosts, re-point these cases at the host that runs it")
        declared = harness.tool_artifacts()
        assert "aider" in declared, (
            "aider declares no `tool_artifacts:` in redirect.yaml, so the "
            "exemption the codex rows rely on is coming from somewhere else")
        assert harness.tool_artifact_paths([AIDER_CACHE], ["aider"]) == [
            AIDER_CACHE], (
            f"aider declares {declared['aider']}, which does not admit the "
            f"file four recorded rows measured: {AIDER_CACHE}")

    def test_the_measured_codex_row_grades_clean(self):
        """`codex-apply-executor`'s real failure, now a pass.

        Its whole source-filtered delta was the cache file, on a row declared
        `clean-failure` — so `bytes moved` was reported over bytes no subagent
        wrote.
        """
        assert _bytes_graded(CODEX_ROW_DELTA, clis=FAN_OUT) == []

    def test_a_real_product_write_is_still_caught_on_an_exempt_row(self):
        """THE TEST THAT KEEPS THIS FROM BEING A STRAIGHT LOOSENING.

        Same row, same fan-out, same exemption in force — and one real write
        beside the cache. The row must still go red, and must go red naming
        the source file and not the cache, or the exemption has swallowed the
        assertion instead of narrowing it.
        """
        graded = _bytes_graded(
            (*CODEX_ROW_DELTA, PLANTED_WRITE), clis=FAN_OUT)
        assert graded == [PLANTED_WRITE], (
            f"a product write on a row carrying the aider exemption must "
            f"still be graded; got {graded}")
        # ...and through the declared-paths half of the same row's grade, on a
        # row that declares one: the cache is forgiven, the source file is the
        # stray it always was.
        owned = ["tests/test_*.py"]
        assert harness.stray_paths(
            _bytes_graded((*CODEX_ROW_DELTA, PLANTED_WRITE),
                          clis=FAN_OUT, declared=owned),
            owned) == [PLANTED_WRITE]

    @pytest.mark.parametrize("name", [
        # Every shape a reader might expect the glob to reach and it must not.
        ".aider.tags.cache.v5/cache.db",   # a version bump: re-measure, don't guess
        ".aider.chat.history.md",          # aider's, and never measured here
        "aider.tags.cache.v4/cache.db",    # no leading dot
        "src/.aider.tags.cache.v4/cache.db",  # nested; the glob is repo-anchored
        "src/auth.py",
    ])
    def test_the_glob_refuses_everything_it_was_not_measured_on(self, name):
        assert harness.tool_artifact_paths([name], FAN_OUT) == []

    def test_the_exemption_is_scoped_to_the_rows_own_fan_out(self):
        """aider's declaration buys nothing on a row that does not run aider.

        A literal partition rather than one read back out of
        `host_matrix.yaml` — this is the same list `claude-code` declares
        today, but the property under test is "a fan-out without aider in
        it", which stays true however the matrix is re-cut. The identical
        byte on such a row is a finding, because nothing it dispatches to is
        declared to write it.
        """
        no_aider = ("codex", "muse", "cline", "kilocode", "pi", "goose")
        assert "aider" not in no_aider
        assert harness.tool_artifact_paths([AIDER_CACHE], no_aider) == []
        assert _bytes_graded(CODEX_ROW_DELTA, clis=no_aider) == [AIDER_CACHE]

    def test_a_native_row_gets_no_exemption_at_all(self):
        """`fan_out: native` is a string, and it must yield nothing.

        `claude-code-apply-native` is the ONE row on either lane graded on
        applied bytes, and it declares `native` — so the whole category is
        inert exactly where the byte grade is strictest. Spelled out because
        `harness.FAN_OUT_NATIVE` is a `str`: iterating it character by
        character is a silent way to reach the same empty answer for the wrong
        reason, and would stop being empty the day a one-letter CLI id exists.
        """
        assert harness.tool_artifact_patterns(harness.FAN_OUT_NATIVE) == []
        assert harness.tool_artifact_paths(
            [AIDER_CACHE], harness.FAN_OUT_NATIVE) == []
        # And that row's own measured delta is graded to the byte, unchanged.
        assert _bytes_graded(
            MEASURED, clis=harness.FAN_OUT_NATIVE,
            declared=["src/**", "tests/test_*.py"]) == sorted(MEASURED)

    def test_an_empty_fan_out_gets_no_exemption(self):
        """A `remove_seed` row inherits no fan-out, so it forgives nothing."""
        assert harness.tool_artifact_paths([AIDER_CACHE], []) == []

    def test_the_rows_own_declaration_outranks_the_exemption(self):
        """The same ordering `graded_delta` gives `editable_artifacts`.

        A path a row declares as its own editable artifact is that row's grade
        by definition, and must never be exempted out from under it — a row
        declaring `.aider.**` would otherwise become ungradeable in both
        directions at once.
        """
        assert harness.tool_artifact_paths(
            [AIDER_CACHE], FAN_OUT,
            declared=[".aider.tags.cache.v4/**"]) == []
        assert _bytes_graded(
            [AIDER_CACHE], clis=FAN_OUT,
            declared=[".aider.tags.cache.v4/**"]) == [AIDER_CACHE]

    def test_it_never_reaches_the_workflows_own_bookkeeping_category(self):
        """Two categories, not one widened one.

        `workflow_bookkeeping` is THIS product's record-keeping and must not
        learn about vendor caches; the cache is not under `plans/` or
        `.multi-llm/` and must not be smuggled in there either.
        """
        assert harness.workflow_bookkeeping(AIDER_CACHE) is False
        assert harness.tool_artifact_paths(
            ["plans/auth-feature/state.json", ".multi-llm/runs/x/run.json"],
            FAN_OUT) == []


#: grok's answer file from `runs/20260922T084502Z/opencode-ask`, byte for byte
#: (219 bytes). Both `ask` rows of that run carried an identical copy, and both
#: rows went GREEN on `test_every_assigned_provider_really_ran` while grok was
#: failing on all six of that host's fan-out rows.
GROK_ERROR_BODY = (
    '{"type":"error","message":"Not signed in. To authenticate without a '
    'browser, run:\\n  grok login --device-code\\n\\nAlternatively, set the '
    'XAI_API_KEY environment variable or run `grok login` on a machine with '
    'a browser."}'
)

#: Genuine result bodies, taken verbatim from the same two bundles, one per
#: distinct shape either of them produced. The class below asserts the rule
#: accepts every one — a rule that turns a currently-green genuine row red is a
#: worse defect than the one it fixes.
#:
#: * the stub's own two-byte answer — 20 of the 24 `ask` answer files in
#:   `runs/20260922T084502Z`;
#: * aider's, which prefixes its banner to the same answer (421 bytes,
#:   `codex-ask`);
#: * a real prose answer against a live model (1044 bytes, `chain-claude-code`
#:   and `chain-opencode` in `runs/20260920T105458Z`);
#: * a review phase's result, which is a JSON ARRAY — all 40 review result
#:   files across both bundles, and the reason the object check below is not
#:   also an array check.
GENUINE_BODIES = {
    "the stub's answer": "OK",
    "aider's banner + answer": (
        "Detected dumb terminal, disabling fancy input and pretty output.\n\n"
        "Aider v0.86.2\nModel: openai/stub-model with whole edit format\n"
        "Git repo: .git with 7 files\nRepo-map: using 1024 tokens, auto "
        "refresh\nAdded plans/auth-feature.md to the chat (read-only).\n\n\n"
        "tests/test_auth.py\nOK\n\nTokens: 1 sent, 1 received."),
    "a prose answer": (
        "## Short answer\n\nThe weakest part of the rate limiter is "
        "`src/security.py`: the attempt store is\nan unbounded module-level "
        "dict, the lockout comparison is off by one, and token\ncomparison is "
        "not constant time.\n"),
    "a review result": (
        '[\n  {\n    "title": "Add email format validation",\n'
        '    "desc": "The plan mentions validating email format using regex '
        'but doesn\'t specify the pattern.",\n    "type": "addition",\n'
        '    "importance": "MEDIUM"\n  }\n]'),
}


class TestAnErrorPayloadDoesNotCountAsAResult:
    """§3.3's `result` fact, and the run that showed a name is not enough.

    `runs/20260922T084502Z`: grok failed on all six opencode fan-out rows.
    Four went red. `opencode-ask` and `opencode-package-ask` went GREEN,
    because `ask_orchestrator.recover_answer_text` then fell back to the
    provider's raw stdout when it could recover no text, and a failed grok
    run's stdout is its error envelope — which landed at grok's own exact
    expected filename and satisfied the fact whose whole job is to prove the
    provider produced something. The ask lane no longer reads `raw`: it reads
    only the adapter's reduced `answer_text`, which is `""` for that body
    (plans/ask-answer-recovery-and-false-success.md D2.3, D3.1). This harness
    check does not depend on that fix and stays as the independent second
    gate.

    The discriminators are the CLIs' own, already measured inside this repo:
    `type: "error"` (`utils/providers/grok.py:186-192`, ten distinct injected
    failures; `utils/providers/cline.py:101-102` for a second CLI) and
    `is_error: true` (`utils/providers/result_envelope.py`, eleven committed
    claude-code captures). The shape gate in front of them is what keeps a
    genuine answer out of reach: the body must parse WHOLE as a JSON object,
    and across the 88 per-provider result files in the two bundles the only
    top-level object is grok's envelope.
    """

    def test_groks_recorded_error_body_is_refused(self):
        reason = harness.error_envelope_reason(GROK_ERROR_BODY)
        assert reason is not None, (
            "the 219-byte body that made two failing rows green is still "
            "being read as an answer")
        assert reason.startswith("Not signed in"), (
            f"the CLI's own message is what a failure should quote; got "
            f"{reason!r}")

    @pytest.mark.parametrize("label", sorted(GENUINE_BODIES))
    def test_every_recorded_genuine_body_is_still_accepted(self, label):
        """The direction that matters more: no currently-green row goes red."""
        assert harness.error_envelope_reason(GENUINE_BODIES[label]) is None, (
            f"{label} is a real result from a recorded bundle and would now "
            f"be refused, turning green rows red")

    @pytest.mark.parametrize("body", [
        '{"is_error": true, "result": "API error: 401", "type": "result"}',
        '{"type":"error","message":""}',            # no message to quote
        '{"type":"error"}',
        '  \n{"type": "error", "message": "boom"}\n ',  # leading whitespace
    ])
    def test_the_other_measured_envelope_shapes_are_refused(self, body):
        assert harness.error_envelope_reason(body) is not None

    @pytest.mark.parametrize("body", [
        "",                                    # nothing at all is not an envelope
        "The run failed with is_error: true.",  # prose ABOUT a failure
        '{"type": "result", "result": "OK", "is_error": false}',
        '{"error": null, "text": "the answer"}',   # an `error` key alone is not it
        '[{"title": "x", "type": "error"}]',       # an array of findings
        '{"broken": ',                              # not JSON
        '{"a": 1} trailing prose',                  # not a WHOLE object
    ])
    def test_it_refuses_nothing_it_was_not_measured_on(self, body):
        assert harness.error_envelope_reason(body) is None

    def test_the_row_goes_red_naming_only_the_failed_provider(self):
        """End to end, as `opencode-ask` should have graded.

        grok's file is dropped from `produced`, so the correlation reports
        `grok: result` and says nothing about the four providers that really
        answered — the same discrimination the absent-provider control above
        asserts, reached through a body rather than through a missing file.
        """
        fan_out = ("claude-code", "omp", "qwen-code", "grok", "vibe")
        bodies = {cli: GENUINE_BODIES["the stub's answer"] for cli in fan_out}
        bodies["grok"] = GROK_ERROR_BODY
        produced = [harness.result_file_names(cli)[1] for cli in fan_out
                    if harness.error_envelope_reason(bodies[cli]) is None]
        missing = harness.fan_out_correlation(
            fan_out, invoked=set(fan_out), reached=set(fan_out),
            produced=produced)
        assert missing == {"grok": ["result"]}, (
            f"the ask row that went green while grok was failing should now "
            f"report grok and only grok; got {missing}")


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
