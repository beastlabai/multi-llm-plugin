"""Wiring tests for code_review_orchestrator.

These tests focus specifically on verifying that the review pipeline is
wrapped in the `intent_to_add_untracked` context manager so untracked files
are visible to git diff during review execution, validation prep, and
report generation.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import code_review_orchestrator as cro
from utils import cli_common, status_file
from utils import review_orchestrator_base as base


def _make_args(plan_path: Path, **overrides) -> argparse.Namespace:
    """Build a Namespace mimicking argparse output with safe defaults."""
    defaults = dict(
        plan_file=str(plan_path),
        models=["cursor-agent:auto"],
        interactive=False,
        quick=False,
        timeout=None,
        max_parallel=2,
        base_ref=None,
        skip_validation=True,  # skip the interactive validation branch
        validation_model=None,
        apply_fixes=False,
        force=False,
        rerun_all=False,
        reaggregate=False,
        report_style="pr",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class _TrackingContextManager:
    """Context manager stand-in that records enter/exit ordering.

    Attach it to a parent MagicMock so enter/exit calls interleave with
    other mock calls on the same parent. That lets us assert that pipeline
    functions run between __enter__ and __exit__.
    """

    def __init__(self, parent: MagicMock, touched: list):
        self._parent = parent
        self._touched = touched

    def __enter__(self):
        self._parent.context_enter()
        return self._touched

    def __exit__(self, exc_type, exc, tb):
        self._parent.context_exit()
        return False


def _call_names(parent: MagicMock) -> list:
    """Return just the attribute names from parent.mock_calls (no args)."""
    return [c[0] for c in parent.mock_calls]


def _assert_between(
    names: list, target: str, start: str, end: str
) -> None:
    """Assert target appears between the first `start` and the last `end`."""
    assert target in names, f"{target} not in call order: {names}"
    assert start in names and end in names
    start_idx = names.index(start)
    end_idx = len(names) - 1 - list(reversed(names)).index(end)
    tgt_idx = names.index(target)
    assert start_idx < tgt_idx < end_idx, (
        f"expected {target} between {start} and {end}; got order {names}"
    )


class TestMainWrapsPipelineWithIntentToAdd:
    """Verify main() wraps the review pipeline in intent_to_add_untracked()."""

    def test_main_calls_intent_to_add_with_changed_files(self, tmp_path):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")

        # Fake tracked_files from the implementation phase
        tracked_files = [
            {"path": "src/a.py"},
            {"path": "src/b.py"},
        ]

        fake_state = MagicMock()
        fake_state.get = MagicMock(
            side_effect=lambda k, default=None: {
                "tracked_files": tracked_files,
                "head_before_implement": "abc123",
                "head_at_start": "abc123",
                "pre_existing_changes": [],
            }.get(k, default)
        )
        fake_state.mark_phase_completed = MagicMock()
        fake_state.save = MagicMock()
        fake_state.is_phase_completed = MagicMock(return_value=False)
        fake_state.state = {"phases_completed": {}}

        # Parent mock to record call order across context + pipeline funcs
        parent = MagicMock()

        # F11: a model spec missing from `run_all_reviews`'s result is now
        # correctly treated as a fan-out failure ("no result (cancelled at
        # the fan-out deadline)") rather than silently passing through as
        # completed — so this wiring test's fake fan-out has to return a real
        # (if empty) result for the one model it configured, or main() exits
        # 3 before ever reaching generate_review_report.
        parent.run_all_reviews = AsyncMock(
            return_value={"cursor-agent:auto": (True, [], None)}
        )
        parent.generate_review_report = MagicMock(return_value="# report\n")

        def fake_ctx(files):
            parent.intent_to_add_untracked(files)
            return _TrackingContextManager(parent, list(files))

        args = _make_args(plan_path)

        with patch.object(cro, "parse_args", return_value=args), \
             patch.object(cro, "get_or_create_state", return_value=fake_state), \
             patch.object(cro, "resolve_models",
                          return_value=["cursor-agent:auto"]), \
             patch.object(cro, "is_model_valid", return_value=True), \
             patch.object(cro, "validate_git_ref",
                          side_effect=lambda r: r or "HEAD~1"), \
             patch.object(cro, "intent_to_add_untracked",
                          side_effect=fake_ctx), \
             patch.object(cro, "run_all_reviews",
                          side_effect=parent.run_all_reviews), \
             patch.object(cro, "generate_review_report",
                          side_effect=parent.generate_review_report), \
             patch.object(cro, "get_phase_dir",
                          return_value=tmp_path / "phase"):
            # main() returns its exit code rather than calling sys.exit(): the
            # dispatcher owns the process exit, and so does the __main__ block.
            asyncio.run(cro.main())

        # intent_to_add_untracked called exactly once with the changed files
        assert parent.intent_to_add_untracked.call_count == 1
        ((called_files,), _) = parent.intent_to_add_untracked.call_args
        assert sorted(called_files) == sorted(["src/a.py", "src/b.py"])

        # run_all_reviews and generate_review_report both ran inside the ctx
        names = _call_names(parent)
        _assert_between(
            names, "run_all_reviews", "context_enter", "context_exit"
        )
        _assert_between(
            names, "generate_review_report", "context_enter", "context_exit"
        )


class TestValidationInputsCarryTheStampedGroupHash:
    """F2: ``group_similar_suggestions`` returns raw ``SuggestionGroup``
    objects; ``stamp_stable_ids`` is run against a separate ``to_dict()`` copy
    (``grouped_list``) so ``grouped.json`` carries a ``group_hash`` per group.
    The validators and the post-validation merge used to be handed the
    un-stamped originals instead, so no group in the review's own HTML report
    ever carries a ``group_hash`` — the key ``_overlay_human_decisions`` joins
    decisions on — and the decisions overlay silently matched nothing.
    """

    def test_stamped_group_hash_reaches_the_validator_batcher_and_the_merge(
        self, tmp_path
    ):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")

        tracked_files = [{"path": "index.html"}]
        fake_state = MagicMock()
        fake_state.get = MagicMock(
            side_effect=lambda k, default=None: {
                "tracked_files": tracked_files,
                "head_before_implement": "abc123",
                "head_at_start": "abc123",
                "pre_existing_changes": [],
            }.get(k, default)
        )
        fake_state.mark_phase_completed = MagicMock()
        fake_state.save = MagicMock()
        fake_state.is_phase_completed = MagicMock(return_value=False)
        fake_state.state = {"phases_completed": {}}

        issue = {
            "title": "OOB teleport skips resync",
            "desc": "OOB teleport skips resync description",
            "type": "bug",
            "importance": "high",
            "file": "index.html",
        }

        captured: dict = {}

        def fake_prepare(groups, **kwargs):
            captured["prepare_groups"] = groups
            return {"total_batches": 1, "batching_stats": {}}

        def fake_merge(phase_dir, batch_metadata, grouped, validation_path):
            captured["merge_grouped"] = grouped
            return grouped

        args = _make_args(plan_path, skip_validation=False)

        parent = MagicMock()

        def fake_ctx(files):
            return _TrackingContextManager(parent, list(files))

        with patch.object(cro, "parse_args", return_value=args), \
             patch.object(cro, "get_or_create_state", return_value=fake_state), \
             patch.object(cro, "resolve_models",
                          return_value=["cursor-agent:auto"]), \
             patch.object(cro, "is_model_valid", return_value=True), \
             patch.object(cro, "validate_git_ref",
                          side_effect=lambda r: r or "HEAD~1"), \
             patch.object(cro, "intent_to_add_untracked", side_effect=fake_ctx), \
             patch.object(cro, "run_all_reviews", new=AsyncMock(
                 return_value={"cursor-agent:auto": (True, [issue], None)}
             )), \
             patch.object(cro, "build_validation_secondary_context",
                          return_value=("", "")), \
             patch.object(cro, "prepare_batched_validation_tasks",
                          side_effect=fake_prepare), \
             patch.object(cro, "dispatch_subagent_tasks", return_value=True), \
             patch.object(cro, "merge_validation_batches",
                          side_effect=fake_merge), \
             patch.object(cro, "generate_review_report",
                          return_value="# report\n"), \
             patch.object(cro, "get_phase_dir", return_value=tmp_path / "phase"):
            asyncio.run(cro.main())

        prepare_groups = captured.get("prepare_groups")
        merge_grouped = captured.get("merge_grouped")
        assert prepare_groups, "no groups reached prepare_batched_validation_tasks"
        assert merge_grouped, "no groups reached merge_validation_batches"
        assert all(g.get("group_hash") for g in prepare_groups), prepare_groups
        assert all(g.get("group_hash") for g in merge_grouped), merge_grouped
        # Same grouping reached both call sites - one set of stable ids, not
        # a second, un-stamped set computed independently.
        assert (
            [g["group_hash"] for g in prepare_groups]
            == [g["group_hash"] for g in merge_grouped]
        )

    def test_skip_validation_groups_also_carry_the_stamped_hash(self, tmp_path):
        """The ``--skip-validation`` branch had the identical bug (``grouped``,
        not ``grouped_list``) feeding the report and any later ``--apply-fixes``
        batcher.
        """
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")

        tracked_files = [{"path": "index.html"}]
        fake_state = MagicMock()
        fake_state.get = MagicMock(
            side_effect=lambda k, default=None: {
                "tracked_files": tracked_files,
                "head_before_implement": "abc123",
                "head_at_start": "abc123",
                "pre_existing_changes": [],
            }.get(k, default)
        )
        fake_state.mark_phase_completed = MagicMock()
        fake_state.save = MagicMock()
        fake_state.is_phase_completed = MagicMock(return_value=False)
        fake_state.state = {"phases_completed": {}}

        issue = {
            "title": "OOB teleport skips resync",
            "desc": "OOB teleport skips resync description",
            "type": "bug",
            "importance": "high",
            "file": "index.html",
        }

        captured: dict = {}

        def fake_report(plan_path, results, changed_files, validated_groups, **kwargs):
            captured["report_groups"] = validated_groups
            return "# report\n"

        args = _make_args(plan_path, skip_validation=True)

        parent = MagicMock()

        def fake_ctx(files):
            return _TrackingContextManager(parent, list(files))

        with patch.object(cro, "parse_args", return_value=args), \
             patch.object(cro, "get_or_create_state", return_value=fake_state), \
             patch.object(cro, "resolve_models",
                          return_value=["cursor-agent:auto"]), \
             patch.object(cro, "is_model_valid", return_value=True), \
             patch.object(cro, "validate_git_ref",
                          side_effect=lambda r: r or "HEAD~1"), \
             patch.object(cro, "intent_to_add_untracked", side_effect=fake_ctx), \
             patch.object(cro, "run_all_reviews", new=AsyncMock(
                 return_value={"cursor-agent:auto": (True, [issue], None)}
             )), \
             patch.object(cro, "generate_review_report",
                          side_effect=fake_report), \
             patch.object(cro, "get_phase_dir", return_value=tmp_path / "phase"):
            asyncio.run(cro.main())

        report_groups = captured.get("report_groups")
        assert report_groups, "no groups reached generate_review_report"
        assert all(g.get("group_hash") for g in report_groups), report_groups


class TestReaggregateWrapsPipelineWithIntentToAdd:
    """Verify reaggregate_from_existing_files() wraps its pipeline too."""

    def test_reaggregate_calls_intent_to_add_with_changed_files(self, tmp_path):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")

        out_dir = tmp_path / "out"
        out_dir.mkdir()

        phase_dir = tmp_path / "phase"
        phase_dir.mkdir()
        # Create a fake model result file that the function will pick up
        (phase_dir / "cursor-agent_auto.json").write_text(
            '[{"title": "x", "desc": "y", "importance": "low", '
            '"file": "src/a.py", "type": "style"}]',
            encoding="utf-8",
        )

        tracked_files = [
            {"path": "src/a.py"},
            {"path": "src/new.py"},
        ]

        fake_state = MagicMock()
        fake_state.get = MagicMock(
            side_effect=lambda k, default=None: {
                "tracked_files": tracked_files,
                "head_before_implement": "abc123",
                "head_at_start": "abc123",
            }.get(k, default)
        )
        fake_state.mark_phase_completed = MagicMock()
        fake_state.save = MagicMock()

        parent = MagicMock()
        parent.generate_review_report = MagicMock(return_value="# report\n")

        def fake_ctx(files):
            parent.intent_to_add_untracked(files)
            return _TrackingContextManager(parent, list(files))

        args = _make_args(plan_path, reaggregate=True, skip_validation=True)

        with patch.object(cro, "get_or_create_state", return_value=fake_state), \
             patch.object(cro, "get_phase_dir", return_value=phase_dir), \
             patch.object(cro, "validate_git_ref",
                          side_effect=lambda r: r or "HEAD~1"), \
             patch.object(cro, "build_unsanitize_map", return_value={}), \
             patch.object(cro, "intent_to_add_untracked",
                          side_effect=fake_ctx), \
             patch.object(cro, "generate_review_report",
                          side_effect=parent.generate_review_report):
            asyncio.run(
                cro.reaggregate_from_existing_files(
                    "plan", out_dir, plan_path, args
                )
            )

        assert parent.intent_to_add_untracked.call_count == 1
        ((called_files,), _) = parent.intent_to_add_untracked.call_args
        assert sorted(called_files) == sorted(["src/a.py", "src/new.py"])

        names = _call_names(parent)
        _assert_between(
            names, "generate_review_report", "context_enter", "context_exit"
        )


class TestAFanoutModelLostToTheGatherDeadline:
    """F11: a model whose task never returned — cancelled by
    `run_all_reviews`'s own `asyncio.wait_for` deadline, or dropped as an
    `Exception` item there — used to be in neither `results` nor
    `failed_models`: it was counted completed, inflated `models_done`, and
    appeared in no report section at all.
    """

    def test_a_missing_model_is_recorded_failed_not_silently_completed(
        self, tmp_path
    ):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")

        tracked_files = [{"path": "src/a.py"}]
        fake_state = MagicMock()
        fake_state.get = MagicMock(
            side_effect=lambda k, default=None: {
                "tracked_files": tracked_files,
                "head_before_implement": "abc123",
                "head_at_start": "abc123",
                "pre_existing_changes": [],
            }.get(k, default)
        )
        fake_state.mark_phase_completed = MagicMock()
        fake_state.save = MagicMock()
        fake_state.is_phase_completed = MagicMock(return_value=False)
        fake_state.state = {"phases_completed": {}}

        args = _make_args(
            plan_path, models=["cursor-agent:auto", "claude-code:sonnet"],
        )

        # `claude-code:sonnet` never made it into the fan-out's own results --
        # exactly what an exception item or a total gather timeout leaves
        # behind in `run_all_reviews`'s return value.
        async def fake_run_all_reviews(*a, **kw):
            return {"cursor-agent:auto": (True, [{
                "title": "Bug", "desc": "d", "importance": "low",
                "file": "src/a.py", "type": "bug",
            }], None)}

        with patch.object(cro, "parse_args", return_value=args), \
             patch.object(cro, "get_or_create_state", return_value=fake_state), \
             patch.object(cro, "resolve_models",
                          return_value=["cursor-agent:auto", "claude-code:sonnet"]), \
             patch.object(cro, "is_model_valid", return_value=True), \
             patch.object(cro, "validate_git_ref",
                          side_effect=lambda r: r or "HEAD~1"), \
             patch.object(cro, "run_all_reviews", new=fake_run_all_reviews):
            exit_code = asyncio.run(cro.main())

        assert exit_code == 0
        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        status = status_file.read_status(str(phase_dir))
        assert status["models_failed"]["claude-code:sonnet"] == (
            "no result (cancelled at the fan-out deadline)"
        )
        assert "claude-code:sonnet" not in status["models_completed"]
        report = (phase_dir / "report.md").read_text(encoding="utf-8")
        assert "## Failed Reviews" in report
        assert "claude-code:sonnet" in report


class TestReaggregateRecordsArtifacts:
    """F2/F3: `reaggregate_from_existing_files` never called `record_artifacts`
    or `record_output_files` at all — main()'s own completion site had the
    same gap and got fixed; this one, reached only by `--reaggregate`, did
    not. A detached `--reaggregate` run therefore reported `artifacts: []`
    and an empty `output_files`, with nothing to tell a host the report it
    just wrote exists.

    ``get_phase_dir`` is deliberately left unmocked here (unlike the sibling
    wiring tests above): this function's own local ``phase_dir`` comes from
    calling it directly, but ``report_path``/``grouped_path``/``issues_path``
    come from ``get_output_paths`` in ``utils/output_handler.py``, which calls
    its *own* module-level ``get_phase_dir`` — a name a `patch.object(cro, …)`
    cannot reach. Mocking it here would make the seeded per-model file and
    the function's real outputs land in two different directories.
    """

    def test_reaggregate_records_report_grouped_and_issues(self, tmp_path):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")

        out_dir = tmp_path / "out"
        out_dir.mkdir()

        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        phase_dir.mkdir(parents=True, exist_ok=True)
        (phase_dir / "cursor-agent_auto.json").write_text(
            '[{"title": "x", "desc": "y", "importance": "low", '
            '"file": "src/a.py", "type": "style"}]',
            encoding="utf-8",
        )

        tracked_files = [{"path": "src/a.py"}]
        fake_state = MagicMock()
        fake_state.get = MagicMock(
            side_effect=lambda k, default=None: {
                "tracked_files": tracked_files,
                "head_before_implement": "abc123",
                "head_at_start": "abc123",
            }.get(k, default)
        )
        fake_state.mark_phase_completed = MagicMock()
        fake_state.save = MagicMock()

        args = _make_args(plan_path, reaggregate=True, skip_validation=True)

        ctx = cli_common.activate_result(cli_common.ResultContext())
        try:
            with patch.object(cro, "get_or_create_state", return_value=fake_state), \
                 patch.object(cro, "validate_git_ref",
                              side_effect=lambda r: r or "HEAD~1"), \
                 patch.object(cro, "build_unsanitize_map", return_value={}):
                asyncio.run(
                    cro.reaggregate_from_existing_files(
                        "plan", out_dir, plan_path, args
                    )
                )
        finally:
            cli_common.deactivate_result()

        report_path = phase_dir / "report.md"
        assert report_path.exists()
        assert str(report_path) in ctx.artifacts
        assert str(phase_dir / "report.html") in ctx.artifacts
        assert str(phase_dir / "grouped.json") in ctx.artifacts
        assert str(phase_dir / "issues.json") in ctx.artifacts
        # No validation ran (--skip-validation), so validation.json was never
        # written and must not be claimed as an artifact.
        assert not (phase_dir / "validation.json").exists()
        assert str(phase_dir / "validation.json") not in ctx.artifacts


class TestEmptyReviewReplacesPreviousFindings:
    @pytest.mark.parametrize("reaggregate", [False, True])
    @pytest.mark.parametrize("skip_validation", [False, True])
    def test_clean_review_clears_old_groups_and_completes(
        self, tmp_path, reaggregate, skip_validation
    ):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        phase_dir.mkdir(parents=True, exist_ok=True)
        old_issue = {"title": "Old bug", "desc": "already fixed", "file": "src/a.py"}
        (phase_dir / "issues.json").write_text(json.dumps([old_issue]), encoding="utf-8")
        (phase_dir / "grouped.json").write_text(json.dumps([{
            "group_hash": "old", "suggestions": [old_issue],
        }]), encoding="utf-8")
        (phase_dir / "validation.json").write_text('[{"group_id": "old"}]', encoding="utf-8")
        (phase_dir / "validation_tasks.json").write_text("{}", encoding="utf-8")
        (phase_dir / "report.md").write_text("Old bug", encoding="utf-8")
        (phase_dir / "cursor-agent_auto.json").write_text("[]", encoding="utf-8")

        fake_state = MagicMock()
        fake_state.get.side_effect = lambda key, default=None: {
            "tracked_files": [{"path": "src/a.py"}],
            "head_at_start": "abc123",
            "pre_existing_changes": [],
        }.get(key, default)
        fake_state.is_phase_completed.return_value = False
        fake_state.state = {"phases_completed": {}}
        args = _make_args(plan_path, force=True, skip_validation=skip_validation)

        with patch.object(cro, "parse_args", return_value=args), \
             patch.object(cro, "get_or_create_state", return_value=fake_state), \
             patch.object(cro, "resolve_models", return_value=["cursor-agent:auto"]), \
             patch.object(cro, "is_model_valid", return_value=True), \
             patch.object(cro, "validate_git_ref", side_effect=lambda ref: ref or "HEAD~1"), \
             patch.object(cro, "build_unsanitize_map", return_value={"cursor-agent_auto": "cursor-agent:auto"}), \
             patch.object(cro, "run_all_reviews", new=AsyncMock(return_value={"cursor-agent:auto": (True, [], None)})), \
             patch.object(cro, "dispatch_subagent_tasks") as dispatch:
            if reaggregate:
                asyncio.run(cro.reaggregate_from_existing_files(
                    "plan", phase_dir.parent, plan_path, args
                ))
            else:
                assert asyncio.run(cro.main()) == 0

        dispatch.assert_not_called()
        assert json.loads((phase_dir / "issues.json").read_text()) == []
        assert cro.load_groups_payload(json.loads((phase_dir / "grouped.json").read_text())) == []
        assert json.loads((phase_dir / "validation.json").read_text()) == []
        report = (phase_dir / "report.md").read_text(encoding="utf-8")
        assert "**Total Issues:** 0" in report
        assert "Old bug" not in report
        fake_state.mark_phase_completed.assert_called_with("code-review")
        assert status_file.read_status(str(phase_dir))["state"] == "complete"


class TestGenerateReviewReportFailedModelDerivation:
    """F10: ``success and issues`` misclassified a reviewer that legitimately
    returned ``[]`` as failed — it was listed under ``## Failed Reviews`` as
    "No issues returned" even though the model completed cleanly and
    ``save_model_result``/``.status.json.models_completed`` already counted it
    done.
    """

    def test_an_empty_but_successful_model_is_not_a_failed_review(self, tmp_path):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        results = {
            "model-a": (True, [], None),
            "model-b": (
                True,
                [{
                    "title": "Bug", "desc": "d", "importance": "low",
                    "file": "x.py", "type": "bug",
                }],
                None,
            ),
        }

        report = cro.generate_review_report(plan_path, results, changed_files=["x.py"])

        assert "## Failed Reviews" not in report
        assert "No issues returned" not in report
        models_line = next(
            line for line in report.splitlines() if line.startswith("**Models:**")
        )
        assert "model-a" in models_line
        assert "model-b" in models_line

    def test_the_callers_authoritative_failed_models_map_is_trusted_verbatim(
        self, tmp_path
    ):
        """F11: a model the fan-out lost to the gather deadline never gets an
        entry in ``results`` at all — main() seeds it into ``failed_models``
        instead, and this function must report that rather than silently
        dropping it (or, worse, re-deriving nothing for it from ``results``).
        """
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        results = {"model-a": (True, [], None)}
        failed_models = {
            "model-b": "no result (cancelled at the fan-out deadline)",
        }

        report = cro.generate_review_report(
            plan_path, results, changed_files=[], failed_models=failed_models,
        )

        assert "## Failed Reviews" in report
        failed_section = report.split("## Failed Reviews", 1)[1]
        assert "model-b" in failed_section
        assert "no result (cancelled at the fan-out deadline)" in failed_section
        assert "model-a" not in failed_section

    def test_a_genuinely_failed_model_reports_even_with_no_callers_map(
        self, tmp_path
    ):
        """No ``failed_models`` passed — e.g. a fixture exercising this
        function directly — falls back to deriving one from ``results`` on
        ``success`` alone, not the old ``success and issues``.
        """
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        results = {"model-a": (False, None, "boom")}

        report = cro.generate_review_report(plan_path, results, changed_files=[])

        assert "## Failed Reviews" in report
        assert "model-a" in report
        assert "boom" in report


class TestTheFanoutReportsProgressWhileItRuns:
    """F6: `run_all_reviews` never advanced `progress.models_done` — it was
    written `0` once before dispatch and recomputed only after the last model
    returned, so a mid-run poll read `models 0/4` beside three finished
    reviewers (measured 2026-09-16: 646 s of an 833 s run). The counter is
    the base's `fanout_slot_counter` (shared with `run_all_models`, tested in
    test_review_plan_orchestrator.py), which reports through the base's own
    `_detach` — the seam patched here.
    """

    def test_each_finished_slot_advances_the_counter(self, tmp_path):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        calls = []

        def record(phase_dir, **updates):
            calls.append(dict(updates))

        async def fake_review(*args, **kwargs):
            return True, [], None

        with patch.object(cro, "run_single_review", new=fake_review), \
             patch.object(cro, "PROVIDER_STAGGER_DELAY", 0.0), \
             patch.object(base._detach, "report_progress", record):
            results = asyncio.run(cro.run_all_reviews(
                ["cursor-agent:auto", "claude-code:sonnet", "codex:gpt-5.5"],
                plan_path, tmp_path, 30, 3,
            ))

        assert len(results) == 3
        counted = [c["models_done"] for c in calls if "models_done" in c]
        # One call per finished slot — no opening zero here, unlike the base's
        # fan-out, because this lane's own `0` is written by `main()` before
        # `run_all_reviews` is ever called, not by this function itself.
        assert counted == [1, 2, 3]
        assert all(c["models_total"] == 3 for c in calls if "models_total" in c)

    def test_a_resumed_run_counts_the_skipped_slots_too(self, tmp_path):
        """A model answered from an existing `{model}.json` has finished its
        slot exactly as surely as one that was dispatched; a resume that
        skips every model must not read `models 0/2` for the whole of it."""
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        phase_dir.mkdir(parents=True, exist_ok=True)
        (phase_dir / "cursor-agent_auto.json").write_text("[]", encoding="utf-8")
        calls = []

        def record(phase_dir_arg, **updates):
            calls.append(dict(updates))

        async def fake_review(*args, **kwargs):
            return True, [], None

        with patch.object(cro, "run_single_review", new=fake_review), \
             patch.object(cro, "PROVIDER_STAGGER_DELAY", 0.0), \
             patch.object(base._detach, "report_progress", record):
            results = asyncio.run(cro.run_all_reviews(
                ["cursor-agent:auto", "claude-code:sonnet"],
                plan_path, tmp_path, 30, 2,
            ))

        assert len(results) == 2
        counted = [c["models_done"] for c in calls if "models_done" in c]
        assert counted == [1, 2]


# ---------------------------------------------------------------------------
# D6 step 3 — the review-code lane asks the same pre-dispatch sandbox question
# ---------------------------------------------------------------------------

#: What real bubblewrap prints on a stock Ubuntu 24.04 with
#: `kernel.apparmor_restrict_unprivileged_userns=1`, replayed verbatim.
_BWRAP_BROKEN_LINE = "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted"


class TestTheCodeReviewFanoutGatesOnTheSandboxProbe:
    """`run_single_review` is its own dispatch lane, so it asks for itself.

    Same probe, same once-per-run cache and same wording as
    `review_orchestrator_base.run_single_model`: a `generate` dispatch whose
    sandbox cannot start reads neither the plan nor the diff, and answers with
    a sentence about the files it could not open — which scores as a review.
    """

    class _Probed:
        """An adapter declaring a `generate` probe that fails, as codex's does
        where bwrap cannot start. The `sh` is real; the line is replayed."""

        internal_retries = None

        def sandbox_probe_command(self, profile):
            from utils.providers.base import PROFILE_GENERATE

            if profile != PROFILE_GENERATE:
                return None
            return ["sh", "-c", f"echo '{_BWRAP_BROKEN_LINE}' >&2; exit 1"]

    def test_the_code_review_fanout_refuses_a_slot_whose_sandbox_cannot_start(
        self, tmp_path, monkeypatch, capsys
    ):
        from utils import review_orchestrator_base as rob
        from utils import subagent_executor as executor

        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# Plan\n", encoding="utf-8")
        monkeypatch.setattr(rob, "_FANOUT_SANDBOX_CACHE", {})
        monkeypatch.setattr(rob, "get_provider", lambda _name: self._Probed())
        monkeypatch.setattr(cro, "get_project_root", lambda _p: str(tmp_path))

        def no_dispatch(**kwargs):
            raise AssertionError("a slot was dispatched past a refused probe")

        monkeypatch.setattr(cro, "invoke_with_file_output", no_dispatch)

        def no_retire(*args, **kwargs):
            raise AssertionError("a refused slot claimed the phase dir")

        monkeypatch.setattr(cro, "retire_salvage_artifacts", no_retire)
        probes = []
        real = executor._run_sandbox_probe

        def counting(argv, **kwargs):
            probes.append(list(argv))
            return real(argv, **kwargs)

        monkeypatch.setattr(executor, "_run_sandbox_probe", counting)

        success, issues, error = asyncio.run(cro.run_single_review(
            "fakebwrap:m1", plan_path, tmp_path / "out",
        ))
        assert success is False
        assert issues is None
        assert error.startswith("[SANDBOX_UNAVAILABLE]")
        assert "fakebwrap's sandbox cannot start here" in error
        assert _BWRAP_BROKEN_LINE in error
        assert "multi_llm.py doctor" in error
        assert error in capsys.readouterr().out

        # `save_model_result` records this as a failed model under exactly this
        # reason, so `models_failed` and the report's failed section carry it.
        #
        # The second slot shares the fan-out's cache: one probe for the run.
        again = asyncio.run(cro.run_single_review(
            "fakebwrap:m2", plan_path, tmp_path / "out",
        ))
        assert again[0] is False and again[2] == error
        assert len(probes) == 1

    # -- the D6 override (2026-09-17): the third answer of the gate --------

    class _Attested(_Probed):
        """The probing adapter plus the operator's `external` attestation."""

        def sandbox_fallback_policy(self, profile, anchor=None):
            from utils.providers.base import PROFILE_GENERATE

            return "external" if profile == PROFILE_GENERATE else None

    BYPASS_LINE = (
        "[SANDBOX_BYPASSED] fakebwrap: bubblewrap cannot start; running --sandbox "
        "danger-full-access under providers.fakebwrap.sandbox_fallback=external "
        "(read-only phase; no OS boundary between the model's shell and this "
        "machine)"
    )

    def _attested_lane(self, tmp_path, monkeypatch, dispatch):
        from utils import review_orchestrator_base as rob

        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# Plan\n", encoding="utf-8")
        monkeypatch.setattr(rob, "_FANOUT_SANDBOX_CACHE", {})
        monkeypatch.setattr(rob, "get_provider", lambda _name: self._Attested())
        monkeypatch.setattr(cro, "get_project_root", lambda _p: str(tmp_path))
        monkeypatch.setattr(cro, "invoke_with_file_output", dispatch)
        return plan_path

    def test_the_code_review_fanout_dispatches_unsandboxed_with_the_flag_and_the_marker(
        self, tmp_path, monkeypatch, capsys
    ):
        seen = []

        def dispatch(**kwargs):
            seen.append(kwargs)
            return {
                "success": True, "data": [], "source": "stdout_fallback",
                "output_file": str(tmp_path / "out" / "o.json"),
                "details": {"sandbox": "bypassed"},
            }

        plan_path = self._attested_lane(tmp_path, monkeypatch, dispatch)
        served = {}
        success, issues, error = asyncio.run(cro.run_single_review(
            "fakebwrap:m1", plan_path, tmp_path / "out", served_models=served,
        ))
        assert (success, issues, error) == (True, [], None)
        assert len(seen) == 1 and seen[0]["sandbox_fallback"] is True
        # The provenance lands in the collector without a served model beside it.
        assert served == {"fakebwrap:m1": {"sandbox": "bypassed"}}
        out = capsys.readouterr().out
        assert out.count("[SANDBOX_BYPASSED]") == 1
        assert f"[fakebwrap:m1] {self.BYPASS_LINE}" in out

    def test_a_bypassed_code_review_slot_that_writes_is_failed_by_path(
        self, tmp_path, monkeypatch, capsys
    ):
        import os
        import subprocess

        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x"}
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, env=env)
        (tmp_path / "src.py").write_text("v1\n", encoding="utf-8")
        subprocess.run(["git", "add", "src.py"], cwd=tmp_path, check=True, env=env)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True, env=env)

        def dispatch(**kwargs):
            (tmp_path / "src.py").write_text("v2\n", encoding="utf-8")
            return {
                "success": True, "data": [{"title": "x"}], "source": "stdout_fallback",
                "output_file": str(tmp_path / "out" / "o.json"),
                "details": {"sandbox": "bypassed"},
            }

        plan_path = self._attested_lane(tmp_path, monkeypatch, dispatch)
        success, issues, error = asyncio.run(cro.run_single_review(
            "fakebwrap:m1", plan_path, tmp_path / "out",
        ))
        assert success is False and issues is None
        assert error == (
            "[SANDBOX_BYPASSED] fakebwrap wrote outside its read-only contract: src.py"
        )
        assert error in capsys.readouterr().out
        assert (tmp_path / "src.py").read_text(encoding="utf-8") == "v2\n"


class TestTheReportReconcilesItsIssueCountWithItsGroupCount:
    """`generate_review_report` numbers issues and validates groups.

    Measured 2026-09-16: a header reading "12 issues, 7 valid, 1 invalid,
    1 needs human review" over 9 groups, with the same consensus bug printed
    three times as three findings with three checkboxes, and the word "group"
    nowhere in the file — three sets of numbers, none reconcilable without
    grouped.json.
    """

    def _issue(self, title, importance="high"):
        return {
            "title": title,
            "desc": f"{title} description",
            "type": "bug",
            "importance": importance,
            "file": "index.html",
        }

    def _render(self, tmp_path, validated=True):
        import copy

        teleport_a = self._issue("OOB teleport skips resync")
        teleport_b = self._issue("Teleport leaves timing stale")
        fog = self._issue("Fog distance from control points", importance="medium")
        results = {
            "cursor-agent:composer": (True, [teleport_a, fog], None),
            "claude-code:sonnet": (True, [teleport_b], None),
        }
        groups = [
            {
                "theme": "Teleport does not resync timing",
                "file": "index.html",
                "validation_status": "valid",
                "validation_reason": "",
                "suggestions": [copy.deepcopy(teleport_a), copy.deepcopy(teleport_b)],
            },
            {
                "theme": "Fog distance",
                "file": "index.html",
                "validation_status": "needs-human-decision",
                "validation_reason": "Needs a human eye",
                "suggestions": [copy.deepcopy(fog)],
            },
        ]
        return cro.generate_review_report(
            tmp_path / "plan.md",
            results,
            ["index.html"],
            groups if validated else None,
        )

    def test_the_header_counts_issues_and_groups_separately(self, tmp_path):
        report = self._render(tmp_path)
        assert "**Total Issues:** 3" in report
        assert "**Groups:** 2" in report
        # The validation line has always counted groups; now it says so.
        assert "1 valid, 0 invalid, 1 needs human review (of 2 groups)" in report

    def test_a_consensus_finding_names_its_siblings_once_per_member(self, tmp_path):
        report = self._render(tmp_path)
        # Issues 1 and 3 are the same finding (model order: cursor, cursor, claude).
        assert "_Same finding as #3 (2 models agree); marking any marks all._" in report
        assert "_Same finding as #1 (2 models agree); marking any marks all._" in report
        # The lone finding claims no siblings.
        assert report.count("Same finding as") == 2

    def test_a_shared_verdict_is_labelled_as_the_group_s(self, tmp_path):
        report = self._render(tmp_path)
        # Two badges for one verdict — counting badges is not counting verdicts.
        assert report.count("**Validation:** Valid (group verdict)") == 2
        assert "**Validation:** ? Needs Review |" in report
        assert "? Needs Review (group verdict)" not in report

    def test_the_checkbox_contract_survives_the_extra_lines(self, tmp_path):
        """report_parser keys off `### N.` + the very next line, so Skip stays there."""
        from utils.report_parser import (
            parse_skipped_issues,
            parse_validation_overrides_issues,
            parse_issue_descriptions,
        )

        report = self._render(tmp_path).replace(
            "### 3. Teleport leaves timing stale\n- [ ] Skip",
            "### 3. Teleport leaves timing stale\n- [x] Skip",
        ).replace("- [ ] Mark invalid", "- [x] Mark invalid")
        path = tmp_path / "report.md"
        path.write_text(report, encoding="utf-8")

        assert parse_skipped_issues(str(path)) == {3}
        assert parse_validation_overrides_issues(str(path)) == {2: "invalid"}
        # The sibling line is not mistaken for an edited description.
        descriptions = parse_issue_descriptions(str(path))
        assert descriptions[1].startswith("OOB teleport skips resync description")

    def test_a_title_holding_a_newline_keeps_its_skip_box_on_the_next_line(
        self, tmp_path
    ):
        """F6: the header and the box are one unit to `report_parser`.

        A title is model output. A `\\n` in it pushed the Skip box two lines
        below the header, the `### N.` + next-line regex stopped matching, and
        the tick was lost with no warning.
        """
        from utils.report_parser import parse_skipped_issues

        issue = self._issue("Line one\nline two of the title")
        report = cro.generate_review_report(
            tmp_path / "plan.md",
            {"cursor-agent:composer": (True, [issue], None)},
            ["index.html"],
            [{"theme": "t", "file": "index.html", "validation_status": "valid",
              "validation_reason": "", "suggestions": [dict(issue)]}],
            phase_dir=None,
        )

        assert "### 1. Line one line two of the title\n- [ ] Skip" in report
        path = tmp_path / "report.md"
        path.write_text(report.replace("- [ ] Skip", "- [x] Skip"), encoding="utf-8")
        assert parse_skipped_issues(str(path)) == {1}

    def test_an_unvalidated_report_says_nothing_about_groups(self, tmp_path):
        """--skip-validation hands no groups over; the report claims none."""
        report = self._render(tmp_path, validated=False)
        assert "**Groups:**" not in report
        assert "Same finding as" not in report
        assert "**Total Issues:** 3" in report
class TestWhatTheValidatorsAreShownOfTheCodeUnderReview:
    """F7: the plan whole, plus the code at the lines the issues cite.

    The validation prompt declares `context_mode: inline`, so a batch is judged
    on what the prompt carries. It used to carry a plan the caller had already
    sliced at 30 000 chars before `excerpt_context` applied the same limit
    again, and nothing at all of the code.
    """

    @staticmethod
    def _groups():
        return [{
            "theme": "CONFIG",
            "suggestions": [
                {"file": "index.html", "line_range": [98, 99],
                 "type": "bug", "details": "accel deviates"},
                {"file": "index.html", "line_range": [103, 103],
                 "type": "bug", "details": "nearby"},
            ],
        }]

    def test_the_excerpt_carries_the_cited_lines_with_their_real_numbers(
        self, tmp_path
    ):
        (tmp_path / "index.html").write_text(
            "\n".join(f"row {i}" for i in range(1, 201)), encoding="utf-8"
        )

        excerpt = cro.build_reviewed_code_excerpt(
            self._groups(), project_root=str(tmp_path), context_lines=3
        )

        assert "### index.html" in excerpt
        assert "98: row 98" in excerpt
        assert "103: row 103" in excerpt
        # The two windows overlap once ±3 lines are added, so they are merged
        # into one block rather than repeating rows 100-102 twice.
        assert excerpt.count("100: row 100") == 1
        assert excerpt.count("```") == 2
        # Nothing outside the windows travels: this is what keeps the excerpt
        # proportional to the findings rather than to the file.
        assert "row 1\n" not in excerpt
        assert "row 200" not in excerpt

    def test_a_file_that_cannot_be_read_is_named_rather_than_dropped(
        self, tmp_path
    ):
        excerpt = cro.build_reviewed_code_excerpt(
            self._groups(), project_root=str(tmp_path)
        )

        assert "### index.html" in excerpt
        assert "not readable at review time" in excerpt

    def test_issues_with_no_location_produce_no_excerpt(self, tmp_path):
        groups = [{"suggestions": [
            {"file": "", "line_range": [1, 2]},
            {"file": "index.html", "line_range": None},
            {"file": "index.html", "line_range": ["x", "y"]},
        ]}]

        assert cro.build_reviewed_code_excerpt(
            groups, project_root=str(tmp_path)
        ) == ""

    def test_the_plan_reaches_the_validators_whole(self, tmp_path, monkeypatch):
        """The double truncation, in the units that caused it.

        The plan that produced this finding is 44 374 chars; the caller's slice
        ended mid-§4.6 and §4.7-§11 reached no validator, while four of nine
        verdicts cited exactly those sections.
        """
        from utils.validation import VALIDATION_CONTEXT_LIMIT, excerpt_context

        plan = "# Plan\n" + ("filler line\n" * 3000) + "\n## S11 Definition of Done\n"
        assert 30000 < len(plan) <= VALIDATION_CONTEXT_LIMIT

        assert "S11 Definition of Done" in excerpt_context(plan)


class TestTheExcerptStaysInsideTheRepository:
    """F1: `suggestion["file"]` is model output, and whatever it names is read.

    The lines land in every validation batch prompt and in
    `validation_tasks.json`, which lives under the plan's output dir and is
    commonly committed — so a reviewer (or a prompt injection a reviewer echoes)
    citing `../../.config/multi-llm/e2e.env` used to ship those lines to the
    validation provider.
    """

    @staticmethod
    def _repo(tmp_path):
        repo = tmp_path / "repo"
        (repo / "src").mkdir(parents=True)
        (repo / "src" / "app.py").write_text(
            "\n".join(f"row {i}" for i in range(1, 51)), encoding="utf-8"
        )
        secret = tmp_path / "outside.txt"
        secret.write_text("SECRET_ONE\nSECRET_TWO\nSECRET_THREE\n", encoding="utf-8")
        return repo, secret

    def test_a_relative_path_that_escapes_the_root_is_named_but_not_read(
        self, tmp_path
    ):
        repo, _ = self._repo(tmp_path)
        groups = [{"suggestions": [
            {"file": "../outside.txt", "line_range": [1, 3]},
        ]}]

        excerpt = cro.build_reviewed_code_excerpt(groups, project_root=str(repo))

        assert "### ../outside.txt" in excerpt
        assert "outside the repository; not shown" in excerpt
        assert "SECRET_ONE" not in excerpt

    def test_an_absolute_cited_path_is_named_but_not_read(self, tmp_path):
        repo, secret = self._repo(tmp_path)
        groups = [{"suggestions": [{"file": str(secret), "line_range": [1, 3]}]}]

        excerpt = cro.build_reviewed_code_excerpt(groups, project_root=str(repo))

        assert f"### {secret}" in excerpt
        assert "outside the repository; not shown" in excerpt
        assert "SECRET_TWO" not in excerpt

    def test_a_symlink_pointing_out_of_the_repository_is_not_followed(
        self, tmp_path
    ):
        """Both sides are resolved, so an in-tree name is not enough."""
        repo, secret = self._repo(tmp_path)
        try:
            (repo / "link.txt").symlink_to(secret)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable on this platform")
        groups = [{"suggestions": [{"file": "link.txt", "line_range": [1, 3]}]}]

        excerpt = cro.build_reviewed_code_excerpt(groups, project_root=str(repo))

        assert "outside the repository; not shown" in excerpt
        assert "SECRET_THREE" not in excerpt

    def test_a_path_inside_the_repository_is_still_read(self, tmp_path):
        repo, _ = self._repo(tmp_path)
        groups = [{"suggestions": [
            {"file": "src/app.py", "line_range": [10, 11]},
            {"file": "./src/../src/app.py", "line_range": [30, 30]},
        ]}]

        excerpt = cro.build_reviewed_code_excerpt(
            groups, project_root=str(repo), context_lines=0
        )

        assert "10: row 10" in excerpt
        assert "30: row 30" in excerpt
        assert "not shown" not in excerpt

    def test_the_budget_drops_whole_files_and_names_the_ones_it_dropped(
        self, tmp_path
    ):
        """Half a window reads as a whole one; a named absence does not."""
        repo = tmp_path / "repo"
        repo.mkdir()
        groups = []
        for name in ("a.py", "b.py", "c.py"):
            (repo / name).write_text(
                "\n".join(f"{name} line {i} " + "x" * 80 for i in range(1, 41)),
                encoding="utf-8",
            )
            groups.append({"suggestions": [
                {"file": name, "line_range": [1, 40]},
            ]})

        full = cro.build_reviewed_code_excerpt(
            groups, project_root=str(repo), context_lines=0
        )
        trimmed = cro.build_reviewed_code_excerpt(
            groups, project_root=str(repo), context_lines=0,
            budget=len(full) // 2,
        )

        assert "a.py line 1 " in trimmed
        # A dropped file keeps none of its lines and is named instead.
        assert "c.py line 1 " not in trimmed
        assert "c.py" in trimmed.rsplit("(Over this block", 1)[-1]
        assert len(trimmed) < len(full)


class TestWhatTheReviewersAreToldTheImplementationDid:
    """F8: a logged, deliberate deviation must not read as an unexplained one."""

    @staticmethod
    def _plan_with_a_record(tmp_path):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# Plan\n\naccel: 28\n", encoding="utf-8")
        out = tmp_path / "plan"
        (out / "implement").mkdir(parents=True)
        (out / "implement" / "summary.md").write_text("# Summary\n", encoding="utf-8")
        (out / "prior_changes.jsonl").write_text(
            json.dumps({"id": "T003", "phase": "implement",
                        "summary": "accel tuned 28->32 for <4s"}) + "\n"
            + json.dumps({"id": "step_4", "phase": "apply-suggestions",
                          "summary": "edited the plan"}) + "\n",
            encoding="utf-8",
        )
        (out / "state.json").write_text(
            json.dumps({
                "task_status": {"T003": "completed", "T014": "skipped",
                                "apply-batch-0": "completed"},
                "task_status_reasons": {"T014": "user chose to complete manually"},
            }),
            encoding="utf-8",
        )
        return plan_path

    def test_the_record_names_the_logged_deviation_and_the_deferred_task(
        self, tmp_path
    ):
        record = cro.build_implementation_record(self._plan_with_a_record(tmp_path))

        assert "accel tuned 28->32 for <4s" in record
        assert "T014 (skipped: user chose to complete manually)" in record
        assert "summary.md" in record
        # Another phase's log entries are not this phase's record, and a
        # write-batch id is not an implement task.
        assert "edited the plan" not in record
        assert "apply-batch-0" not in record

    def test_a_task_that_did_not_finish_is_a_gap_and_not_a_decision(
        self, tmp_path
    ):
        """F7: implement seeds every task `pending`, and crashes leave them.

        "Must not be reported as missing" is a claim about a decision somebody
        made. A failed task, or one left `pending` by an aborted run, is exactly
        the one whose missing work a code review should report — so it goes on
        its own line, with whatever reason was recorded for context.
        """
        plan_path = self._plan_with_a_record(tmp_path)
        (tmp_path / "plan" / "state.json").write_text(
            json.dumps({
                "task_status": {
                    "T001": "completed",
                    "T002": "failed",
                    "T003": "in_progress",
                    "T004": "pending",
                    "T014": "skipped",
                },
                "task_status_reasons": {
                    "T002": "write batch timed out",
                    "T014": "user chose to complete manually",
                },
            }),
            encoding="utf-8",
        )

        record = cro.build_implementation_record(plan_path)
        decided, gaps = (
            [line for line in record.splitlines()
             if "must not be reported as missing" in line],
            [line for line in record.splitlines() if "did not finish" in line],
        )

        assert len(decided) == 1 and len(gaps) == 1
        assert "T014" in decided[0]
        for task_id in ("T002", "T003", "T004"):
            assert task_id in gaps[0]
            assert task_id not in decided[0]
        # The reason a failed task recorded is still shown, as context.
        assert "write batch timed out" in gaps[0]
        assert "T001" not in record

    def test_a_pending_task_with_a_recorded_reason_is_a_decision(self, tmp_path):
        """A reason is what distinguishes "not run on purpose" from "not run"."""
        plan_path = self._plan_with_a_record(tmp_path)
        (tmp_path / "plan" / "state.json").write_text(
            json.dumps({
                "task_status": {"T007": "pending"},
                "task_status_reasons": {"T007": "deferred to the next milestone"},
            }),
            encoding="utf-8",
        )

        record = cro.build_implementation_record(plan_path)

        assert "must not be reported as missing: T007 (pending: deferred" in record
        assert "did not finish" not in record

    def test_a_plan_with_no_implement_phase_says_so_without_creating_state(
        self, tmp_path
    ):
        """A reader must not be the thing that writes the run's state.

        `StateManager` creates `state.json` when it is absent, and
        `_create_initial_state` shells out to git for `head_at_start` — so a
        summary of a phase that never ran would both write a file and fail
        wherever git cannot answer.
        """
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# Plan\n", encoding="utf-8")

        record = cro.build_implementation_record(plan_path)

        assert "Nothing on record" in record
        assert not (tmp_path / "plan" / "state.json").exists()

    def test_the_review_prompt_template_binds_the_record(self):
        template = cro.load_code_review_prompt()

        assert "{implementation_record}" in template
        assert "## Implementation Record" in template

    def test_the_validation_secondary_block_leads_with_the_record(self, tmp_path):
        plan_path = self._plan_with_a_record(tmp_path)
        (tmp_path / "index.html").write_text(
            "\n".join(f"row {i}" for i in range(1, 201)), encoding="utf-8"
        )
        groups = [{"suggestions": [{"file": "index.html", "line_range": [98, 99]}]}]

        secondary, reviewed_code = cro.build_validation_secondary_context(
            groups, plan_path, project_root=str(tmp_path)
        )

        # The record is the shorter half and the one a shared-excerpt trim must
        # not reach, so it goes first.
        assert secondary.index("Implementation record") < secondary.index(
            "Code at the cited lines"
        )
        assert "accel tuned 28->32 for <4s" in secondary
        assert "98: row 98" in secondary
        # Only the code half is hashed: the record is regenerated on every read.
        assert reviewed_code and "accel tuned" not in reviewed_code
        assert reviewed_code in secondary

    def test_the_record_cannot_crowd_the_code_out_of_the_block(
        self, tmp_path, capsys
    ):
        """F3: one budget between the halves made the *code* the half cut.

        Measured: a 150-entry implement record (~20 KB) plus 20 cited files left
        11 of the 20 with no window at all, traced only by a `[... truncated
        ...]` line inside the prompt — the exact hole the second block exists to
        close, one block over.
        """
        from utils.validation import VALIDATION_CONTEXT_LIMIT, excerpt_context

        plan_path = self._plan_with_a_record(tmp_path)
        with open(tmp_path / "plan" / "prior_changes.jsonl", "a",
                  encoding="utf-8") as handle:
            for i in range(150):
                handle.write(json.dumps({
                    "id": f"T{i:03d}", "phase": "implement",
                    "summary": f"task {i}: " + ("changed a thing " * 10),
                }) + "\n")
        groups = []
        for i in range(20):
            name = f"m{i:02d}.py"
            (tmp_path / name).write_text(
                "\n".join(f"line {n} " + "x" * 60 for n in range(1, 201)),
                encoding="utf-8",
            )
            groups.append({"suggestions": [
                {"file": name, "line_range": [100, 101]},
            ]})

        snapshot_calls = []
        real_snapshots = cro.capture_file_snapshots

        def counting_snapshots(paths):
            snapshot_calls.append(list(paths))
            return real_snapshots(paths)

        with patch.object(cro, "capture_file_snapshots", counting_snapshots):
            secondary, reviewed_code = cro.build_validation_secondary_context(
                groups, plan_path, project_root=str(tmp_path)
            )

        record_half = secondary.split("#### Code at the cited lines")[0]
        assert len(record_half) <= cro.IMPLEMENTATION_RECORD_CHARS + 200
        # The whole block now fits, so the caller's shared excerpt takes nothing
        # more off the end of the code.
        assert len(secondary) <= VALIDATION_CONTEXT_LIMIT
        assert excerpt_context(secondary) == secondary
        # Whole files are dropped, named, and never half-shown.
        assert "the windows for these files are not shown" in secondary
        assert "[... truncated ...]" not in secondary.split(
            "#### Code at the cited lines"
        )[1]
        # The hash still covers the code as it is on disk, untrimmed, so the
        # reaggregate site's own recompute agrees with it.
        assert reviewed_code == cro.build_reviewed_code_excerpt(
            groups, project_root=str(tmp_path)
        )
        assert len(reviewed_code) > VALIDATION_CONTEXT_LIMIT
        # The over-budget path used to render the excerpt twice, reading every
        # cited file twice; now it reads once and fits twice.
        assert len(snapshot_calls) == 1
        # And the stderr line names the files that lost their window — the
        # question a verdict about an unshown file raises — not just a count.
        err = capsys.readouterr().err
        line = next(ln for ln in err.splitlines() if "reviewed-code" in ln)
        assert "[VALIDATION_CONTEXT_TRUNCATED] reviewed-code" in line
        shown = secondary.split("#### Code at the cited lines")[1]
        dropped = [
            f"m{i:02d}.py" for i in range(20) if f"### m{i:02d}.py" not in shown
        ]
        assert dropped, "the fixture is meant to overflow the code budget"
        assert f"windows for {len(dropped)} of 20 cited files not shown: " in line
        for name in dropped:
            assert name in line.split("not shown: ")[1]

    def test_a_file_edited_after_fanout_moves_the_recorded_input_hash(
        self, tmp_path
    ):
        """F7's other half: `diff_base` alone cannot see an edit on its near side."""
        from utils.subagent_tasks import build_inputs

        plan_path = self._plan_with_a_record(tmp_path)
        target = tmp_path / "index.html"
        target.write_text(
            "\n".join(f"row {i}" for i in range(1, 201)), encoding="utf-8"
        )
        groups = [{"suggestions": [{"file": "index.html", "line_range": [98, 99]}]}]

        before = build_inputs(
            plan_file=str(plan_path),
            diff_base="abc123",
            reviewed_code=cro.build_reviewed_code_excerpt(
                groups, project_root=str(tmp_path)
            ),
        )
        lines = target.read_text(encoding="utf-8").split("\n")
        lines[97] = "row 98 (edited)"
        target.write_text("\n".join(lines), encoding="utf-8")
        after = build_inputs(
            plan_file=str(plan_path),
            diff_base="abc123",
            reviewed_code=cro.build_reviewed_code_excerpt(
                groups, project_root=str(tmp_path)
            ),
        )

        assert before["diff_base"] == after["diff_base"]
        assert before["reviewed_code_sha256"] != after["reviewed_code_sha256"]


class TestTheValidationTaskFileMainWrites:
    """End to end: what a review-code run really hands its validators."""

    def test_the_batch_prompts_carry_the_whole_plan_the_code_and_the_record(
        self, tmp_path
    ):
        plan_path = tmp_path / "plan.md"
        # Longer than the caller's old 30 000-char slice, so the tail of the
        # plan is present only because that slice is gone.
        plan_path.write_text(
            "# Plan\n" + ("filler line\n" * 3000) + "\n## S8 accel: 28\n",
            encoding="utf-8",
        )
        assert 30000 < len(plan_path.read_text(encoding="utf-8"))

        reviewed = tmp_path / "index.html"
        reviewed.write_text(
            "\n".join(f"row {i}" for i in range(1, 201)), encoding="utf-8"
        )
        out = tmp_path / "plan"
        (out / "implement").mkdir(parents=True)
        (out / "implement" / "summary.md").write_text("# Summary\n", encoding="utf-8")
        (out / "prior_changes.jsonl").write_text(
            json.dumps({"id": "T003", "phase": "implement",
                        "summary": "accel tuned 28->32 for <4s"}) + "\n",
            encoding="utf-8",
        )
        (out / "state.json").write_text(json.dumps({"task_status": {}}), "utf-8")

        phase_dir = tmp_path / "phase"
        phase_dir.mkdir()
        issues = [{
            "title": "accel deviates from the plan",
            "desc": "index.html sets accel: 32 but the plan says 28",
            "importance": "high", "file": "index.html", "line_range": [98, 99],
            "type": "bug", "anchor_text": "accel: 32",
            "model": "cursor-agent:auto", "source_model": "cursor-agent:auto",
        }]

        fake_state = MagicMock()
        fake_state.get = MagicMock(
            side_effect=lambda k, default=None: {
                "tracked_files": [{"path": "index.html"}],
                "head_before_implement": "abc123",
                "head_at_start": "abc123",
                "pre_existing_changes": [],
            }.get(k, default)
        )
        fake_state.state = {"phases_completed": {}}
        fake_state.is_phase_completed = MagicMock(return_value=False)

        args = _make_args(plan_path, skip_validation=False)

        with patch.object(cro, "parse_args", return_value=args), \
             patch.object(cro, "get_or_create_state", return_value=fake_state), \
             patch.object(cro, "resolve_models",
                          return_value=["cursor-agent:auto"]), \
             patch.object(cro, "is_model_valid", return_value=True), \
             patch.object(cro, "validate_git_ref",
                          side_effect=lambda r: r or "HEAD~1"), \
             patch.object(cro, "get_project_root", return_value=str(tmp_path)), \
             patch.object(cro, "intent_to_add_untracked",
                          side_effect=lambda files: _TrackingContextManager(
                              MagicMock(), list(files))), \
             patch.object(cro, "run_all_reviews", new=AsyncMock(
                 return_value={"cursor-agent:auto": (True, issues, None)})), \
             patch.object(cro, "generate_review_report",
                          return_value="# report\n"), \
             patch.object(cro, "get_phase_dir", return_value=phase_dir):
            asyncio.run(cro.main())

        tasks = json.loads(
            (phase_dir / "validation_tasks.json").read_text(encoding="utf-8")
        )
        assert tasks["inputs"]["reviewed_code_sha256"]
        assert tasks["batches"]
        for batch in tasks["batches"]:
            prompt = batch["prompt"]
            # The plan, whole: the section past the old 30 000-char cut.
            assert "## S8 accel: 28" in prompt
            # The code, at the lines the issue cites (F7).
            assert "98: row 98" in prompt
            # The implementation's own record (F8).
            assert "accel tuned 28->32 for <4s" in prompt
            assert prompt.count("## Context (") == 2

        # The read side recuts the same excerpt from grouped.json, so an
        # untouched tree resumes instead of the phase refusing its own file.
        # (`get_phase_dir` is patched above; `grouped.json` is written through
        # `get_output_paths`, which is not, so it lands under the plan dir.)
        from utils.state_manager import load_groups_payload
        from utils.subagent_tasks import content_hash

        grouped_path = out / "code-review" / "grouped.json"
        recut = cro.build_reviewed_code_excerpt(
            load_groups_payload(
                json.loads(grouped_path.read_text(encoding="utf-8"))
            ),
            project_root=str(tmp_path),
        )
        assert content_hash(recut) == tasks["inputs"]["reviewed_code_sha256"]
