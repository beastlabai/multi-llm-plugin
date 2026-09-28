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
* **the baselines, on both axes** — every provider-lane CLI in `redirect.yaml`
  has one, and so does every `chain_writer_axis:` writer under its own
  `writer-<cli>/` namespace; each is current-shaped, filed under its own
  directory, carries a manifest naming its axis (and, on the writer axis, the
  reviewer it was pinned to) and is not vacuous. A CLI or a writer added
  without a baseline is ungraded, which is the failure mode this row makes loud
  and cheap. The **one** exemption is declared rather than discovered: a writer
  whose axis entry carries `sandbox_skip_when` and which has never been
  captured — `writer-codex` — is reported as an uncovered writer instead of
  being counted as present, and only while it is absent;
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


def _redirect() -> dict:
    return yaml.safe_load(REDIRECT.read_text(encoding="utf-8")) or {}


def _provider_clis() -> list[str]:
    data = _redirect()
    return [row["cli"] for row in data["clis"] if row.get("lane") == "provider"]


def _writer_axis() -> list[dict]:
    """The `chain_writer_axis:` block, or an empty list while it is absent.

    Tolerant of absence on purpose. The block is a top-level sibling of `clis:`
    and lands in its own change; a `data["chain_writer_axis"]` would make plain
    CI — the lane this whole module exists to be checkable in — die on a
    KeyError mid-landing, which is exactly the "fails for a reason unrelated to
    what it grades" failure the writer axis is being added to avoid. With no
    block there are no writer rows to grade, and nothing here has an opinion.
    """
    return [dict(entry) for entry in (_redirect().get("chain_writer_axis") or [])
            if isinstance(entry, dict) and entry.get("writer")]


PROVIDER_CLIS = _provider_clis()
WRITER_AXIS = _writer_axis()
WRITER_CLIS = [entry["writer"] for entry in WRITER_AXIS]


def _uncovered_writers() -> set[str]:
    """The writers exempt from the completeness check — by DECLARATION, and
    only while they are absent.

    Two conditions, both required, and the second is the one that keeps this
    from becoming a permanent hole:

    * the axis entry carries `sandbox_skip_when` — the row declares a host
      capability it may not find, so a host that cannot run it produces no
      capture and that is a recorded fact rather than an omission; and
    * **nothing is on disk**. Today that is `writer-codex` alone, which no host
      has yet produced. The exemption covers ABSENCE, nothing else: the moment
      a permissive-userns host captures `writer-codex/bundle.json`, this
      function stops returning it and that baseline is held to every assertion
      its two siblings are — version, manifest axis, reviewer pin, eight
      phases, four finalized write reports.

    Written as an `exists()`-conditioned subtraction rather than a
    `pytest.skip` on the row for that reason. A skip is unconditional: it would
    keep passing over a `writer-codex/` directory that had drifted, been
    half-written or been captured against the wrong reviewer, and nothing would
    ever say so. Retiring the `sandbox_skip_when` declaration likewise puts the
    writer straight back into both sets, so the declaration is the only thing
    holding the exemption open and retiring it is a visible, testable act.
    """
    exempt: set[str] = set()
    for entry in WRITER_AXIS:
        if not entry.get("sandbox_skip_when"):
            continue
        if container_bundle.baseline_path(
                entry["writer"], container_bundle.AXIS_WRITER).exists():
            continue
        exempt.add(entry["writer"])
    return exempt


#: Declared-and-absent writers: reported as *uncovered writers*, never as
#: present baselines.
UNCOVERED_WRITERS = _uncovered_writers()

#: The writer rows this host's plain CI actually grades.
GRADED_WRITER_AXIS = [entry for entry in WRITER_AXIS
                      if entry["writer"] not in UNCOVERED_WRITERS]
GRADED_WRITER_IDS = [f"writer-{entry['writer']}" for entry in GRADED_WRITER_AXIS]


def _assert_not_vacuous(baseline: dict, label: str) -> None:
    """The non-vacuity bar, shared by both axes rather than copied onto one.

    A writer baseline is the *same* chain — `full --quick --yes`, eight phases,
    three dispatcher invocations — with the reviewer pinned and the write
    runner varied, so it earns exactly the same assertions. One helper and not
    two copies, because a bar that drifted between the axes would let a writer
    baseline be frozen from half a run while the reviewer axis went on
    refusing that.
    """
    completed = set(baseline["derived"]["phases_completed"])
    assert completed >= {
        "review-plan", "apply-suggestions", "generate-tasks", "review-tasks",
        "apply-task-suggestions", "implement", "code-review",
        "apply-code-fixes",
    }, (label, sorted(completed))
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
        _assert_not_vacuous(container_bundle.load_baseline(cli), cli)

    # -- the writer axis ---------------------------------------------------
    # Parametrized over the *graded* writers only. A declared-and-absent one
    # (`writer-codex`) is reported by `test_a_declared_skip_is_the_only_way_out`
    # as an uncovered writer instead, which is a different claim from "graded
    # and green" and is kept a different claim.

    @pytest.mark.parametrize("entry", GRADED_WRITER_AXIS, ids=GRADED_WRITER_IDS)
    def test_the_writer_baseline_exists_and_is_current(self, entry):
        cli = entry["writer"]
        path = container_bundle.baseline_path(cli, container_bundle.AXIS_WRITER)
        assert path.is_file(), (
            f"{cli} is a writer-axis row in redirect.yaml's chain_writer_axis: "
            f"block with no container golden at {path}, so the one axis that "
            f"grades its write path is ungraded. Capture one from a green "
            f"writer run:\n  python {SEEDER} --writer {cli}"
        )
        baseline = container_bundle.load_baseline(
            cli, container_bundle.AXIS_WRITER)
        assert (baseline.get("bundle_version")
                == container_bundle.CONTAINER_BUNDLE_VERSION)
        manifest = baseline.get("manifest") or {}
        assert manifest.get("cli") == cli, (path, manifest.get("cli"))
        assert manifest.get("axis") == container_bundle.AXIS_WRITER, (
            f"{path} sits in the writer namespace but its manifest says "
            f"axis={manifest.get('axis')!r}. The two axes produce the same "
            f"shape, so nothing but this field tells them apart."
        )
        assert manifest.get("row_id") == container_bundle.row_id_for(
            cli, container_bundle.AXIS_WRITER)
        # The reviewer pin. The writer axis holds the reviewer fixed so the
        # three writer baselines differ from one another only in the write
        # half; one captured against a different reviewer is not comparable to
        # its siblings at all, and would read as a write-path diff.
        assert manifest.get("reviewer") == entry.get("reviewer"), (
            f"{path} was captured against reviewer "
            f"{manifest.get('reviewer')!r}, but chain_writer_axis: pins "
            f"{entry.get('reviewer')!r}. Re-capture, or move the pin "
            f"deliberately and re-seed all three."
        )

    @pytest.mark.parametrize("entry", GRADED_WRITER_AXIS, ids=GRADED_WRITER_IDS)
    def test_the_writer_baseline_is_not_vacuous(self, entry):
        baseline = container_bundle.load_baseline(
            entry["writer"], container_bundle.AXIS_WRITER)
        _assert_not_vacuous(baseline, f"writer-{entry['writer']}")

    def test_a_declared_skip_is_the_only_way_out(self):
        """The exemption is keyed on the DECLARATION, not on what is missing.

        An axis entry with no `sandbox_skip_when` has nothing to excuse it: its
        baseline must exist, full stop. Without this, "the directory is absent"
        would quietly become its own justification and a writer could ship
        ungraded simply by never having been captured — which is the failure
        mode `TestEveryProviderCliHasABaseline` exists for on the other axis.
        """
        undeclared = [entry["writer"] for entry in WRITER_AXIS
                      if not entry.get("sandbox_skip_when")]
        missing = [
            cli for cli in undeclared
            if not container_bundle.baseline_path(
                cli, container_bundle.AXIS_WRITER).is_file()
        ]
        assert not missing, (
            f"{missing} declare no `sandbox_skip_when` in chain_writer_axis: "
            f"and have no writer baseline, so their write path is ungraded "
            f"with nothing recording why. Capture them "
            f"(`python {SEEDER} --writer <cli>`), or — if a host genuinely "
            f"cannot run one — declare the skip on its axis entry, which is "
            f"the reviewed act this test requires."
        )
        # And the exemption can never reach further than the declarations.
        assert UNCOVERED_WRITERS <= {
            entry["writer"] for entry in WRITER_AXIS
            if entry.get("sandbox_skip_when")
        }

    def test_no_baseline_directory_is_orphaned(self):
        """A directory for a CLI that left the lane is a baseline nobody grades."""
        on_disk = {
            path.parent.name
            for path in container_bundle.BASELINE_DIR.glob(
                f"*/{container_bundle.BASELINE_NAME}")
        }
        # Both axes, in their own namespaces, minus the declared-and-absent
        # writers — subtracted from the EXPECTED set rather than ignored in the
        # on-disk one, so a `writer-codex/` that does appear is an ordinary
        # member of the comparison again and is graded, not tolerated.
        expected = (set(PROVIDER_CLIS)
                    | {f"writer-{cli}" for cli in WRITER_CLIS})
        expected -= {f"writer-{cli}" for cli in UNCOVERED_WRITERS}
        assert on_disk == expected, (
            f"tests/golden/container/ holds baselines for {sorted(on_disk)}; "
            f"the provider lane is {sorted(PROVIDER_CLIS)}, the writer axis is "
            f"{sorted(WRITER_CLIS)} and the declared-uncovered writers are "
            f"{sorted(UNCOVERED_WRITERS)}"
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

    def test_file_hashes_are_compared_by_value(self):
        """The rig pins SOURCE_DATE_EPOCH, so `tasks.md`'s digest repeats and
        every file hash — the ~40 stable ones included — grades again."""
        path = "plans/auth-feature/tasks/tasks.md"
        one = {"tasks_sha256": "a" * 16,
               "sha256_before": {path: "b" * 64, "src/new.py": None},
               "sha256_after": {path: "c" * 64}}
        assert container_bundle.fold(one) == one
        two = dict(one, sha256_after={path: "d" * 64})
        assert container_bundle.fold(one) != container_bundle.fold(two)
        assert container_bundle.fold(dict(one, tasks_sha256="e" * 16)) != (
            container_bundle.fold(one))

    def test_a_baseline_captured_under_the_erasure_is_named_stale(self):
        erased = {"outcomes": [{
            "tasks_sha256": "<str>",
            "sha256_before": {"plans/a/tasks/tasks.md": "<str>", "new.py": None},
            "sha256_after": {"plans/a/tasks/tasks.md": "f" * 64},
            "runner_versions": {"codex:gpt-5.5": "<str>"},
        }]}
        assert container_bundle.stale_hash_pointers(erased) == [
            "/outcomes/0/sha256_before/plans~1a~1tasks~1tasks.md",
            "/outcomes/0/tasks_sha256",
        ]
        # A real capture holds digests there; `runner_versions` is still erased
        # by design and is never a staleness signal.
        assert container_bundle.stale_hash_pointers(
            container_bundle.fold({"tasks_sha256": "a" * 16,
                                   "runner_versions": {"x:y": "1"}})) == []

    @pytest.mark.parametrize("row", [
        {"row_id": "chain-codex", "lane": "provider", "cli": "codex",
         "creds": "none"},
        {"row_id": "host-claude-code", "lane": "host", "host": "claude-code",
         "creds": "openrouter"},
    ])
    def test_every_container_is_pinned_to_one_source_date(self, row, tmp_path):
        """What un-erasing the file hashes rests on: the rig, not the product
        default, makes `tasks.md` repeat."""
        sys.path.insert(0, str(TESTS_DIR.parent))
        from tests.container import harness

        argv = harness.build_run_argv(harness.Row(**row), tmp_path,
                                      harness.load_posture())
        pins = [argv[i + 1] for i, token in enumerate(argv[:-1])
                if token == "-e" and argv[i + 1].startswith("SOURCE_DATE_EPOCH=")]
        assert pins == [f"SOURCE_DATE_EPOCH={harness.SOURCE_DATE_EPOCH}"]
        assert harness.SOURCE_DATE_EPOCH.isdigit()


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

    @staticmethod
    def _status_shape(bundle_dir, progress):
        status = (bundle_dir / "artifacts" / "auth-feature" / "review-plan"
                  / ".status.json")
        status.write_text(json.dumps(
            {"phase": "review-plan", "state": "complete",
             "progress": progress}), encoding="utf-8")
        return container_bundle.capture_bundle(
            bundle_dir)["status"]["review-plan/.status.json"]["shape"]

    def test_a_whole_and_a_fractional_elapsed_s_capture_the_same(self, bundle_dir):
        """`progress_tracker._seconds()` writes 2.0 s as `2` and 2.1 s as `2.1`,
        so the measured duration's int/float split is a coin toss per run
        (measured 2026-09-23 on `chain-gemini`/`chain-vibe`) and must not grade."""
        def model(elapsed):
            return {"models": {"vibe:m": {"elapsed_s": elapsed,
                                          "timeout_s": 600}}}
        whole = self._status_shape(bundle_dir, model(2))
        fractional = self._status_shape(bundle_dir, model(2.1))
        assert whole == fractional
        assert whole["progress"]["models"]["vibe:m"]["elapsed_s"] == (
            container_bundle.NUMBER_TOKEN)
        # Still graded: the key vanishing, or turning into a non-number.
        assert self._status_shape(bundle_dir, {"models": {"vibe:m": {
            "timeout_s": 600}}}) != whole
        assert self._status_shape(bundle_dir, model("2s")) != whole

    def test_a_non_duration_number_still_grades_its_type(self, bundle_dir):
        """The fold is by named key, never a global number fold: a counter or a
        configured `timeout_s` that changes int↔float is still a diff."""
        assert container_bundle.MEASURED_DURATION_KEYS == frozenset(
            {"elapsed_s", "queued_s"})
        for key in ("timeout_s", "interval_s", "models_done", "cost_usd"):
            as_int = self._status_shape(bundle_dir, {key: 2})
            as_float = self._status_shape(bundle_dir, {key: 2.5})
            assert as_int != as_float, key
            assert as_int["progress"][key] == "<int>", key

    def test_an_unexported_row_captures_empty_rather_than_raising(self, tmp_path):
        """A row whose `finally` export did not run is a failed row — reported by
        the chain's own assertion, not by a crash inside the capture."""
        bundle = container_bundle.capture_bundle(tmp_path)
        assert bundle["phases"] == {} and bundle["output_root_paths"] == []
