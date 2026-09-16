#!/usr/bin/env python3
"""The Phase-3 instruction-refactor regression guard.

The instruction core is ~5k lines of markdown that only an agent reads, so no
unit test can assert that rewriting it left the workflow's *meaning* alone. What
this asserts is the complement, mechanically: one named, fully mocked workflow run
produces a **byte-identical normalized bundle** before and after the rewrite.

* **Which fixture** — ``tests/fixtures/e2e/plans/auth-feature.md`` driven by
  ``tests/fixtures/e2e/scenarios/full_workflow.yaml``, the pair the e2e suite
  already uses, with the baseline checked in under
  ``tests/fixtures/e2e/golden/instruction_refactor/``. Naming the fixture is what
  makes the two runs comparable at all; the baseline records the sha it was taken
  on (the ``phase3-pre-refactor`` tag, i.e. the Phase 2 merge commit), so "re-run
  after" means the same plan text, the same scenario and the same injected
  responses. Changing any of the three is a deliberate re-capture with its own
  commit.
* **What is compared** — orchestrator-produced artifacts only, never raw model
  text: the phase JSONs, ``state.json``, the ``.status.json`` step sequence, the
  set of relative paths each phase created, the *structure* of the metrics record,
  the ``MULTI_LLM_RESULT`` line each phase emitted, the marker sequence, plus
  derived scalars (suggestion/group counts, importance histogram, the validation
  partition) and the exit codes.
* **What is normalized away** — by one shared ``normalize_run()`` helper rather
  than per-test ad hoc rules: timestamps, durations, token/usage numbers, run ids,
  pids, absolute paths and dict ordering. Those are asserted for presence and type,
  never for value, which is why there is no timing or token threshold to tune.
* **Model non-determinism is eliminated, not tolerated.** Every provider call is
  served from the pinned scenario, so the acceptable diff threshold is zero.

The capture and normalization code lives in
``tests/golden/instruction_refactor_bundle.py`` — one self-contained file, because
the baseline was produced by copying *that file* into a worktree at the tag and
running it there. Both sides of the comparison therefore come from identical
capture code, and a baseline captured by the same commit it polices is impossible
by construction.

Marked ``slow``: it drives five real orchestrator invocations end to end.
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT / "tests" / "golden"))
sys.path.insert(0, str(SKILL_ROOT))

from instruction_refactor_bundle import (  # noqa: E402
    BUNDLE_VERSION,
    GUARD_PLAN,
    GUARD_SCENARIO,
    capture,
)

# The normalize/diff/allowlist machinery is shared with the container lane (C8):
# two guards comparing normalized run bundles must not grow two subtly different
# sets of rules for what "the same run" means.
from bundle_compare import (  # noqa: E402
    AllowlistError,
    diff_bundles,
    load_allowlist,
    normalize_run,
    partition,
)

BASELINE = (
    SKILL_ROOT / "tests" / "fixtures" / "e2e" / "golden" / "instruction_refactor"
    / "bundle.json"
)
ALLOWLIST = SKILL_ROOT / "tests" / "golden" / "expected_changes.yaml"

#: The commit the baseline must have been captured on. Pinned so a re-capture
#: taken on the wrong tree (in particular: on the very commit it is meant to
#: police) fails loudly instead of silently proving nothing.
BASELINE_COMMIT = "272f9469252842e0fff0e5238203f431e8bfc016"


def _load_allowlist():
    """The shared loader (C8), with this guard's own allowlist path.

    The *code* is shared with the container lane because two guards must not grow
    two subtly different notions of what "the same run" means. The *baseline and
    the allowlist* are deliberately not: this file's entries describe one specific
    refactor against one pinned tag, so pointing the container lane at them would
    silently excuse a container regression that happened to land under one — and
    fail container runs on entries whose diff exists only relative to that tag.
    """
    try:
        return load_allowlist(ALLOWLIST)
    except AllowlistError as exc:  # keep this guard's failures assertion-shaped
        raise AssertionError(str(exc)) from exc


def _partition(diffs, entries):
    """Split reported diffs into (allowed, unexplained) and count entry matches."""
    return partition(diffs, entries)


class TestBaselineIntegrity:
    """The baseline has to be the *pre*-refactor one, and it has to be readable."""

    def test_baseline_exists_and_names_its_capture_commit(self):
        assert BASELINE.exists(), (
            f"the pre-refactor golden bundle is missing: {BASELINE}. Re-capture it "
            "from the phase3-pre-refactor tag with "
            "`tests/golden/instruction_refactor_bundle.py --skill-dir … --out …`, "
            "never from the working tree."
        )
        bundle = json.loads(BASELINE.read_text(encoding="utf-8"))
        manifest = bundle["manifest"]
        assert manifest["commit"] == BASELINE_COMMIT, (
            "the baseline bundle records "
            f"{manifest['commit']!r} as its capture commit, expected "
            f"{BASELINE_COMMIT!r} (the phase3-pre-refactor tag). A baseline "
            "captured by the same commit it polices proves nothing."
        )
        assert manifest["bundle_version"] == BUNDLE_VERSION
        assert manifest["fixture_plan"].endswith(f"{GUARD_PLAN}.md")
        assert manifest["fixture_scenario"].endswith(f"{GUARD_SCENARIO}.yaml")

    def test_the_named_fixture_pair_is_on_disk(self):
        plan = SKILL_ROOT / "tests" / "fixtures" / "e2e" / "plans" / f"{GUARD_PLAN}.md"
        scenario = (
            SKILL_ROOT / "tests" / "fixtures" / "e2e" / "scenarios"
            / f"{GUARD_SCENARIO}.yaml"
        )
        assert plan.exists() and scenario.exists(), (plan, scenario)

    def test_baseline_is_not_vacuous(self):
        bundle = json.loads(BASELINE.read_text(encoding="utf-8"))
        assert set(bundle["exit_codes"]) >= {
            "review-plan", "apply-suggestions", "implement", "review-code",
            "apply-code-fixes",
        }
        assert all(code == 0 for code in bundle["exit_codes"].values()), (
            bundle["exit_codes"]
        )
        assert bundle["derived"]["group_count"] > 0
        for phase in ("review-plan", "apply-suggestions", "code-review",
                      "apply-fixes"):
            assert bundle["phases"][phase]["artifacts"], phase
        assert bundle["phases"]["review-plan"]["artifacts"]["grouped.json"]


class TestNormalization:
    """``normalize_run`` erases values, never keys — the property the guard rests on."""

    def test_volatile_scalars_become_type_tokens(self):
        before = normalize_run({"timestamp": "2026-01-01T00:00:00", "duration_ms": 12})
        after = normalize_run({"timestamp": "2026-08-08T11:22:33", "duration_ms": 9991})
        assert before == after == {"timestamp": "<str>", "duration_ms": "<int>"}

    def test_a_missing_volatile_key_still_fails(self):
        assert normalize_run({"timestamp": "x"}) != normalize_run({})

    def test_a_volatile_key_changing_type_still_fails(self):
        assert normalize_run({"tokens": 5}) != normalize_run({"tokens": "5"})

    def test_absolute_paths_are_rewritten_relative_to_the_run_root(self):
        left = normalize_run({"output_dir": "/tmp/a/plan/review-plan"}, "/tmp/a")
        right = normalize_run({"output_dir": "/var/b/plan/review-plan"}, "/var/b")
        assert left == right == {"output_dir": "<run-root>/plan/review-plan"}

    def test_dict_ordering_is_erased_but_list_ordering_is_not(self):
        assert normalize_run({"a": 1, "b": 2}) == normalize_run({"b": 2, "a": 1})
        assert normalize_run([1, 2]) != normalize_run([2, 1])

    def test_content_hashes_stay_compared(self):
        """Group ids are content hashes of the fixture text, so they are stable —
        and a changed grouping is exactly what this guard exists to catch."""
        assert normalize_run({"group_id": "c948ad24b1b8e432"}) != normalize_run(
            {"group_id": "0000000000000000"}
        )


class TestAllowlist:
    def test_allowlist_parses_and_every_entry_is_documented(self):
        _load_allowlist()  # the assertions live in the loader

    def test_allowlist_mechanism_is_not_vacuous(self):
        """Both allowlist rules, driven on a synthetic diff.

        Phase 3 shipped with an empty allowlist (the strongest outcome), so without
        this the file's machinery would never execute. Here it does: an undeclared
        diff is unexplained, a declared one is allowed, and an entry matching
        nothing is reported as stale.
        """
        diffs = [("/phases/review-plan/status_steps/0", "a", "b")]
        entries = [{"pointer": "/phases/review-plan/status_steps",
                    "rationale": "synthetic"}]
        allowed, unexplained, matched = _partition(diffs, entries)
        assert allowed == diffs and unexplained == []
        assert matched == {"/phases/review-plan/status_steps": 1}

        allowed, unexplained, matched = _partition(diffs, [])
        assert unexplained == diffs and allowed == []

        stale = [{"pointer": "/nowhere", "rationale": "synthetic"}]
        _allowed, _unexplained, matched = _partition(diffs, stale)
        assert matched == {"/nowhere": 0}, "a stale entry must report zero matches"


@pytest.mark.slow
class TestZeroDiffAgainstThePreRefactorBaseline:
    """The Done-when row: zero diff outside ``expected_changes.yaml``."""

    def test_bundle_reproduces_the_baseline(self, tmp_path):
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        entries = _load_allowlist()
        fresh = capture(SKILL_ROOT, tmp_path / "guard")
        assert fresh["bundle_version"] == baseline["manifest"]["bundle_version"], (
            "the bundle *shape* changed; a baseline from an older shape cannot be "
            "diffed meaningfully — re-capture it from the tag."
        )
        expected = dict(baseline)
        expected.pop("manifest", None)
        diffs = diff_bundles(expected, fresh)
        allowed, unexplained, matched = _partition(diffs, entries)

        assert not unexplained, (
            "the instruction refactor changed the normalized golden bundle "
            f"({len(unexplained)} undeclared diff(s)). Either the change is a "
            "regression, or it is intentional and belongs in "
            "tests/golden/expected_changes.yaml with a rationale:\n"
            + "\n".join(
                f"  {pointer}\n    baseline: {before!r}\n    now:      {after!r}"
                for pointer, before, after in unexplained[:20]
            )
        )
        stale = [pointer for pointer, hits in matched.items() if hits == 0]
        assert not stale, (
            "expected_changes.yaml entries that no longer match anything (delete "
            f"them, or the file decays into a blanket ignore): {stale}"
        )
        assert len(allowed) == len(diffs)


class TestBatchListsCompareAsMultisets:
    """A `batches` list is a set of batches plus a dispatch order. Read by
    position, a reorder is every leaf under the list changing at once — 67 in
    this guard's own fixture when the isolated-HIGH batches moved first — and
    the only entry that could absorb it, a pointer on the whole list, went on
    excusing any drift underneath. So `diff_bundles` aligns the fresh list
    onto the baseline's slots by each batch's member titles, reports leaves
    at the baseline's index, and reports the order once, as its own leaf."""

    @staticmethod
    def _bundle(*batches):
        return {"phases": {"apply-fixes": {"artifacts": {"orchestrator_output.json": {
            "batches": list(batches), "to_apply": [],
        }}}}}

    @staticmethod
    def _batch(*titles, score=1.0, prompt="p"):
        return {
            "fixes": [{"title": t, "importance": "HIGH"} for t in titles],
            "priority_score": score, "prompt": prompt, "batch_type": "mixed",
        }

    ORDER = "/phases/apply-fixes/artifacts/orchestrator_output.json/batches/@dispatch_order"
    LEAF = "/phases/apply-fixes/artifacts/orchestrator_output.json/batches"

    def test_a_pure_reorder_is_exactly_one_diff_at_the_order_leaf(self):
        mixed, high_a, high_b = (self._batch("A", "B"), self._batch("C"), self._batch("D"))
        diffs = diff_bundles(self._bundle(mixed, high_a, high_b),
                             self._bundle(high_a, high_b, mixed))
        assert diffs == [(self.ORDER, [["A", "B"], ["C"], ["D"]], [["C"], ["D"], ["A", "B"]])]

    def test_a_changed_leaf_inside_a_moved_batch_is_still_reported_at_its_baseline_slot(self):
        baseline = self._bundle(self._batch("A", "B"), self._batch("C"))
        fresh = self._bundle(self._batch("C", prompt="rewritten"), self._batch("A", "B"))
        diffs = diff_bundles(baseline, fresh)
        pointers = [pointer for pointer, _b, _a in diffs]
        assert f"{self.LEAF}/1/prompt" in pointers, pointers
        assert self.ORDER in pointers
        assert len(diffs) == 2, diffs

    def test_a_batch_whose_members_changed_diffs_in_place_not_as_absent_plus_added(self):
        baseline = self._bundle(self._batch("A"), self._batch("B"))
        fresh = self._bundle(self._batch("A"), self._batch("Z"))
        pointers = [pointer for pointer, _b, _a in diff_bundles(baseline, fresh)]
        assert f"{self.LEAF}/1/fixes/0/title" in pointers, pointers
        assert not any(p.startswith(f"{self.LEAF}/2") for p in pointers), pointers

    def test_an_identical_list_is_no_diff_at_all(self):
        same = self._bundle(self._batch("A"), self._batch("B", "C"))
        assert diff_bundles(same, self._bundle(self._batch("A"), self._batch("B", "C"))) == []

    def test_lists_under_any_other_key_stay_positional(self):
        left = {"phases": {"x": {"created_paths": ["a", "b"]}}}
        right = {"phases": {"x": {"created_paths": ["b", "a"]}}}
        assert [p for p, _b, _a in diff_bundles(left, right)] == [
            "/phases/x/created_paths/0", "/phases/x/created_paths/1",
        ]

    def test_the_container_lane_diffs_through_the_same_function(self):
        """One definition of "the same run" (C8): the container lane's
        `diff_bundles` is this one, alignment included."""
        import bundle_compare
        import container_bundle

        assert container_bundle.diff_bundles is bundle_compare.diff_bundles
