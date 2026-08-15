"""The container golden gate, checked where Docker is not — in CI (C8).

`tests/test_container_full_chain.py` owns the comparison, and it is
`container`-marked: it never runs on a PR. That is exactly how the gate went
missing in the first place — `tests/golden/bundle_compare.py` documented a
container guard, `tests/golden/container/expected_changes.yaml` documented its
allowlist, and no module imported either one. Nothing failed, because nothing ran.

So the three things that can be checked without a daemon are checked here:

* **the wiring** — the full-chain module really imports the capture module and the
  shared comparison machinery, and really asserts against
  `tests/golden/container/<cli>/bundle.json`. Asserted over the module's AST, so a
  gate that is deleted, commented out or renamed fails a PR rather than going
  quiet for another release;
* **the baselines** — every provider-lane CLI in `redirect.yaml` has one, it is
  current-shaped, it is filed under its own CLI and it is not vacuous. A CLI added
  to the lane without a baseline is ungraded, which is the failure mode this row
  makes loud and cheap;
* **the machinery** — capture folds what it claims to fold and keeps what it
  claims to keep, and both allowlist rules (including the container lane's own
  `cli` requirement) really refuse what they say they refuse.

**The filename is deliberately not `test_container_*`.** That glob is what the
build context ships into the image and what the marker-gating check enumerates; a
plain-CI module answering to it would be shipped for nothing and would fail its
own marker assertion.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest
import yaml

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR / "golden"))

import container_bundle  # noqa: E402
from bundle_compare import AllowlistError, load_allowlist, partition  # noqa: E402

CHAIN_MODULE = TESTS_DIR / "test_container_full_chain.py"
REDIRECT = TESTS_DIR / "container" / "redirect.yaml"
SEEDER = TESTS_DIR / "container" / "seed_container_golden.py"


def _provider_clis() -> list[str]:
    data = yaml.safe_load(REDIRECT.read_text(encoding="utf-8"))
    return [row["cli"] for row in data["clis"] if row.get("lane") == "provider"]


PROVIDER_CLIS = _provider_clis()


class TestTheGateIsWired:
    """The assertions whose absence let a documented guard run nothing."""

    def test_the_chain_module_imports_the_capture_and_comparison_machinery(self):
        tree = ast.parse(CHAIN_MODULE.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imported.add(f"{module}.{alias.name}")
        assert "golden.container_bundle" in imported, (
            "test_container_full_chain.py no longer imports the container golden "
            "capture module, so no row is compared against a baseline"
        )
        for name in ("load_allowlist", "partition"):
            assert f"golden.bundle_compare.{name}" in imported, (
                f"the chain module no longer imports {name} from the shared "
                f"bundle_compare — the allowlist rules are not being applied"
            )

    def test_the_chain_module_asserts_against_the_baseline(self):
        """A comparison, not just an import: `diff_bundles` reaching an assert."""
        source = CHAIN_MODULE.read_text(encoding="utf-8")
        assert "container_bundle.diff_bundles(" in source, (
            "the chain module imports the capture module but never diffs a "
            "bundle against a baseline"
        )
        assert "container_bundle.load_baseline(" in source
        tree = ast.parse(source)
        classes = [node.name for node in ast.walk(tree)
                   if isinstance(node, ast.ClassDef)]
        assert "TestTheNormalizedBundleReproducesItsBaseline" in classes, (
            "the golden gate class is gone from the chain module"
        )

    def test_the_seeder_exists_and_never_runs_from_the_assertion_side(self):
        """Capture is a host-side step, and the guard may not invoke it.

        A guard that can rewrite its own baseline agrees with whatever it just
        produced, which is the one thing this file is here to prevent.
        """
        assert SEEDER.is_file(), f"the capture CLI is missing: {SEEDER}"
        source = CHAIN_MODULE.read_text(encoding="utf-8")
        assert "write_baseline" not in source, (
            "the chain module references write_baseline — the assertion side must "
            "never write a baseline"
        )
        assert "seed_container_golden" in source, (
            "the chain module's failure messages no longer name the re-seed "
            "command, so a real drift leaves nobody a way to act on it"
        )


class TestEveryProviderCliHasABaseline:
    @pytest.mark.parametrize("cli", PROVIDER_CLIS, ids=PROVIDER_CLIS)
    def test_the_baseline_exists_and_is_current(self, cli):
        path = container_bundle.baseline_path(cli)
        assert path.is_file(), (
            f"{cli} is a provider-lane row in redirect.yaml with no container "
            f"golden at {path}, so its chain is ungraded. Capture one from a green "
            f"chain run:\n  python {SEEDER} --cli {cli}"
        )
        baseline = container_bundle.load_baseline(cli)
        assert (baseline.get("bundle_version")
                == container_bundle.CONTAINER_BUNDLE_VERSION)
        manifest = baseline.get("manifest") or {}
        assert manifest.get("cli") == cli, (path, manifest.get("cli"))
        assert manifest.get("row_id") == container_bundle.row_id_for(cli)

    @pytest.mark.parametrize("cli", PROVIDER_CLIS, ids=PROVIDER_CLIS)
    def test_the_baseline_is_not_vacuous(self, cli):
        """A baseline captured from a half-run would make every later row agree
        with a broken chain."""
        baseline = container_bundle.load_baseline(cli)
        completed = set(baseline["derived"]["phases_completed"])
        assert completed >= {
            "review-plan", "apply-suggestions", "generate-tasks", "review-tasks",
            "apply-task-suggestions", "implement", "code-review",
            "apply-code-fixes",
        }, (cli, sorted(completed))
        assert baseline["derived"]["group_count"] > 0
        assert all(record["exit"] == 0 for record in baseline["commands"])
        assert len(baseline["commands"]) == 3, baseline["commands"]
        for phase in ("review-plan", "code-review"):
            assert baseline["phases"][phase]["artifacts"]["grouped.json"], phase
        for phase in ("apply-suggestions", "apply-task-suggestions", "implement",
                      "apply-fixes"):
            report = baseline["phases"][phase]["artifacts"]["write_batch_report.json"]
            assert report["finalized"] is True, phase
            assert report["tasks"], phase

    def test_no_baseline_directory_is_orphaned(self):
        """A directory for a CLI that left the lane is a baseline nobody grades."""
        on_disk = {
            path.parent.name
            for path in container_bundle.BASELINE_DIR.glob(
                f"*/{container_bundle.BASELINE_NAME}")
        }
        assert on_disk == set(PROVIDER_CLIS), (
            f"tests/golden/container/ holds baselines for {sorted(on_disk)}, the "
            f"provider lane is {sorted(PROVIDER_CLIS)}"
        )


class TestTheAllowlist:
    def test_it_ships_empty_and_parses(self):
        entries = load_allowlist(container_bundle.ALLOWLIST_PATH)
        assert entries == [], (
            "the container allowlist is no longer empty. That is allowed — but "
            "every entry has to be a deliberate, reviewed exception, so this "
            "assertion is the place to record that it was one."
        )

    def test_an_entry_without_a_cli_is_refused(self, tmp_path):
        path = tmp_path / "expected_changes.yaml"
        path.write_text(
            "changes:\n  - pointer: /derived/group_count\n    rationale: why\n",
            encoding="utf-8")
        entries = load_allowlist(path)
        with pytest.raises(AssertionError, match="names no `cli`"):
            container_bundle.validate_entries(entries, path)

    def test_a_blanket_entry_and_an_undocumented_one_are_refused(self, tmp_path):
        blanket = tmp_path / "blanket.yaml"
        blanket.write_text("changes:\n  - pointer: /\n    rationale: no\n    cli: codex\n",
                           encoding="utf-8")
        with pytest.raises(AllowlistError):
            load_allowlist(blanket)
        undocumented = tmp_path / "undocumented.yaml"
        undocumented.write_text("changes:\n  - pointer: /derived\n    cli: codex\n",
                                encoding="utf-8")
        with pytest.raises(AllowlistError):
            load_allowlist(undocumented)

    def test_entries_are_scoped_to_one_cli(self):
        entries = [{"pointer": "/derived", "rationale": "r", "cli": "codex"},
                   {"pointer": "/state", "rationale": "r", "cli": "opencode"}]
        assert container_bundle.entries_for("codex", entries) == [entries[0]]
        assert container_bundle.entries_for("aider", entries) == []

    def test_both_rules_run_on_a_synthetic_diff(self):
        """The machinery must execute even while the file ships empty."""
        diffs = [("/derived/group_count", 4, 3)]
        entries = [{"pointer": "/derived", "rationale": "r", "cli": "codex"}]
        allowed, unexplained, matched = partition(diffs, entries)
        assert allowed == diffs and unexplained == []
        assert matched == {"/derived": 1}

        allowed, unexplained, _ = partition(diffs, [])
        assert unexplained == diffs and allowed == []

        _a, _u, matched = partition(diffs, [{"pointer": "/nowhere",
                                             "rationale": "r", "cli": "codex"}])
        assert matched == {"/nowhere": 0}, "a stale entry must report zero matches"


class TestCaptureFoldsWhatItClaims:
    """The folds, driven directly — the baselines cannot prove their own rules."""

    def test_a_prompt_becomes_a_digest_and_a_change_still_shows(self):
        one = container_bundle.fold({"prompt": "review this plan"})
        two = container_bundle.fold({"prompt": "review that plan"})
        assert one["prompt"].startswith("<text:sha256:")
        assert one != two, "a changed prompt must still be a diff"

    def test_long_text_is_digested_and_short_text_is_not(self):
        short = {"summary": "applied 2 files"}
        assert container_bundle.fold(short) == short
        long_text = "x" * (container_bundle.TEXT_DIGEST_THRESHOLD + 1)
        folded = container_bundle.fold({"summary": long_text})["summary"]
        assert folded.startswith("<text:sha256:") and folded.endswith(
            f":{len(long_text)}>")

    def test_runner_versions_lose_their_values_but_keep_their_keys(self):
        folded = container_bundle.fold(
            {"runner_versions": {"codex:gpt-5.5": "1.2.3"}})
        assert folded == {"runner_versions": {"codex:gpt-5.5": "<str>"}}
        # A NEW runner is still a diff: only the version is erased.
        assert folded != container_bundle.fold(
            {"runner_versions": {"codex:gpt-5.5": "1.2.3", "aider:x": "9"}})

    def test_content_hashes_stay_compared(self):
        """Group ids are content hashes of the fixture text, so a changed grouping
        is exactly the drift this gate is for."""
        assert container_bundle.fold({"group_id": "c948ad24b1b8e432"}) != (
            container_bundle.fold({"group_id": "0000000000000000"}))


class TestCaptureReadsAnExportedRowBundle:
    """Capture against a synthetic bundle, so the shape is asserted without Docker."""

    @pytest.fixture
    def bundle_dir(self, tmp_path):
        out = tmp_path / "chain-codex"
        root = out / "artifacts" / "auth-feature"
        (root / "review-plan").mkdir(parents=True)
        (root / "review-plan" / "grouped.json").write_text(json.dumps(
            {"groups": [{"group_id": "abc123", "importance": "HIGH",
                         "suggestions": [{"id": "s1", "importance": "HIGH"}]}]}),
            encoding="utf-8")
        (root / "review-plan" / "validation.json").write_text(json.dumps(
            {"groups": [{"status": "valid"}]}), encoding="utf-8")
        (root / "review-plan" / ".status.json").write_text(json.dumps(
            {"phase": "review-plan", "state": "complete", "pid": 4242,
             "run": {"state": "done", "exit_code": 0, "heartbeat": "2026-01-01"}}),
            encoding="utf-8")
        (root / "state.json").write_text(json.dumps(
            {"phases_completed": {"review-plan": "2026-01-01T00:00:00"},
             "metrics": {"review-plan": [{"duration_s": 1.5}]},
             "plan_file": "/work/sample-project/plans/auth-feature.md"}),
            encoding="utf-8")
        (out / "commands.json").write_text(json.dumps(
            [{"command": "full plans/auth-feature.md --quick", "exit": 0,
              "env": []}]), encoding="utf-8")
        (out / "container-stdout.txt").write_text(
            "[entrypoint] dispatch: full\n[AUTO_RESOLVED] models: defaults\n"
            'MULTI_LLM_RESULT: {"mode": "full", "state": "done", "exit": 0, '
            '"artifacts": ["/work/sample-project/plans/auth-feature/state.json"]}\n',
            encoding="utf-8")
        (out / "container-stderr.txt").write_text("", encoding="utf-8")
        (out / "workspace-census-before.json").write_text(json.dumps(
            {"plans/auth-feature.md": "aaa", "src/auth.py": "bbb"}),
            encoding="utf-8")
        (out / "workspace-census-after.json").write_text(json.dumps(
            {"plans/auth-feature.md": "ccc", "src/auth.py": "bbb",
             "plans/auth-feature/state.json": "ddd"}), encoding="utf-8")
        return out

    def test_the_bundle_carries_the_surfaces_c8_names(self, bundle_dir):
        bundle = container_bundle.capture_bundle(bundle_dir)
        assert bundle["bundle_version"] == container_bundle.CONTAINER_BUNDLE_VERSION
        assert bundle["commands"] == [
            {"command": "full plans/auth-feature.md --quick", "exit": 0, "env": []}]
        assert bundle["markers"] == ["[AUTO_RESOLVED]"], (
            "the entrypoint's own `[entrypoint]` prose is not a marker, and the "
            "orchestrator's markers are")
        assert bundle["result_lines"][0]["artifacts"] == [
            "<run-root>/plans/auth-feature/state.json"]
        assert bundle["phases"]["review-plan"]["artifacts"]["grouped.json"]
        assert bundle["phases"]["implement"] == {"exists": False}
        assert bundle["status"]["review-plan/.status.json"]["steps"] == [
            "phase=review-plan", "state=complete", "run.state=done",
            "run.exit_code=0"]
        assert bundle["derived"]["group_count"] == 1
        assert bundle["derived"]["importance_histogram"] == {"HIGH": 1}
        assert bundle["derived"]["validation_partition"] == {"valid": 1}
        assert "review-plan/grouped.json" in bundle["output_root_paths"]

    def test_the_workspace_delta_excludes_the_output_dir(self, bundle_dir):
        """The output dir's own churn is `output_root_paths`; this line answers
        the other question — did a write phase move bytes in the project."""
        bundle = container_bundle.capture_bundle(bundle_dir)
        assert bundle["workspace_delta"] == ["plans/auth-feature.md"]

    def test_volatile_values_never_reach_the_bundle(self, bundle_dir):
        bundle = container_bundle.capture_bundle(bundle_dir)
        # The completion *record* survives; the clock reading in it does not.
        assert bundle["state"]["phases_completed"]["review-plan"] == "<timestamp>"
        text = json.dumps(bundle)
        assert "4242" not in text, "a pid reached the bundle"
        assert "2026-01-01" not in text, "a heartbeat value reached the bundle"

    def test_a_changed_artifact_is_a_diff(self, bundle_dir):
        """The gate's own discriminating power, proven rather than assumed."""
        before = container_bundle.capture_bundle(bundle_dir)
        grouped = (bundle_dir / "artifacts" / "auth-feature" / "review-plan"
                   / "grouped.json")
        data = json.loads(grouped.read_text(encoding="utf-8"))
        data["groups"][0]["suggestions"] = []
        grouped.write_text(json.dumps(data), encoding="utf-8")
        after = container_bundle.capture_bundle(bundle_dir)
        pointers = [pointer for pointer, _b, _a
                    in container_bundle.diff_bundles(before, after)]
        assert "/derived/suggestion_count" in pointers, pointers

    def test_an_unexported_row_captures_empty_rather_than_raising(self, tmp_path):
        """A row whose `finally` export did not run is a failed row — reported by
        the chain's own assertion, not by a crash inside the capture."""
        bundle = container_bundle.capture_bundle(tmp_path)
        assert bundle["phases"] == {} and bundle["output_root_paths"] == []
