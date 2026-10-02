"""Wiring tests for code_review_orchestrator.

These tests focus specifically on verifying that the review pipeline is
wrapped in the `intent_to_add_untracked` context manager so untracked files
are visible to git diff during review execution, validation prep, and
report generation.
"""

import argparse
import asyncio
import contextlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import code_review_orchestrator as cro
from utils import cli_common, progress_tracker, status_file
from utils import review_orchestrator_base as base

SKILL_ROOT = Path(__file__).parent.parent


def _answering_review(source: str = "file"):
    """A `run_single_review` stand-in that retires its own slot.

    Not a convenience: `run_single_review` **owns** both of its slot's
    transitions — the dispatch that turns the row `running`, and the terminal
    one, which is the only site that knows the `source` the completion print
    names or the error the failure print names. A stand-in that skips them
    leaves the slot live and the fan-out's `finally` sweep retires it as
    `cancelled`; that is the behaviour, not a test artifact, which is why every
    fake here transitions the way the real function does.
    """

    async def fake_review(model_spec, *args, **kwargs):
        tracker = kwargs.get("tracker")
        assert tracker is not None, "the fan-out did not hand this slot a tracker"
        tracker.running(model_spec, attempt=1)
        tracker.done(model_spec, source=source)
        return True, [], None

    return fake_review


class _FakeClock:
    """A monotonic clock the *ticker* advances, which is what makes the
    cadence assertions exact.

    The ticker's only wait is ``progress_tracker._tick_sleep``; patched to
    :meth:`sleep`, every interval it would have spent moves the clock by
    exactly that interval and yields to the loop, so a test never waits out a
    real 30 s and the number of lines is a property of the script rather than
    of the machine the suite runs on. The same three seams the tracker's own
    unit tests use (``tests/test_progress_tracker.py``).
    """

    def __init__(self, start: float = 10_000.0) -> None:
        self.now = float(start)

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.now += float(delay)
        await asyncio.sleep(0)


@contextlib.contextmanager
def _realtime_ticker(interval_s: float = 0.01):
    """Both tickers on a real (tiny) interval rather than on :class:`_FakeClock`.

    The fake clock only advances when the *ticker* is scheduled, which is the
    right seam for a cadence assertion and the wrong one for measuring whether
    the loop was **free** during a particular window: a ticker that never got a
    turn advances nothing and looks identical to one that was starved. A real
    10 ms interval measures what "the ticker kept printing while the dispatch
    blocked" actually claims. The twin of
    ``tests/test_review_plan_orchestrator._realtime_ticker``.
    """
    emitted: list = []

    async def sleep(_delay):
        await asyncio.sleep(interval_s)

    def emit(line):
        emitted.append(line)
        # Still printed, so a test that cares about ORDER against the phase's
        # own output reads one stream.
        print(line, flush=True)

    with patch.object(progress_tracker, "_tick_sleep", sleep), patch.object(
        progress_tracker, "_emit", emit
    ):
        yield emitted


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


class TestTheLaunchWriteDropsAPredecessorsMap:
    """F1, 2026-09-20 — the review-code lane's copy of the launch write.

    A phase dir is re-used run after run, and on the FOREGROUND lane nothing
    dropped the predecessor's `progress`: `claim_run` is the only writer of
    `progress: None` and only the detached worker calls it, while `write_status`
    carries the reserved objects over from disk by design. So a second review of
    the same plan opened on the last run's rows — every slot `done`, its own
    `models_succeeded` — under this run's `models 0/N done`. The launch write
    now drops it, and this pins that it does on this lane too: the fix is three
    separate copies in three orchestrators, so two lanes passing says nothing
    about the third.
    """

    def test_the_launch_write_drops_a_predecessors_map(self, tmp_path):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        phase = tmp_path / "phase"
        phase.mkdir()
        status_file.write_status(str(phase), {
            "phase": "code-review", "state": "complete",
            "models_requested": ["gemini:old-a", "pi:old-b"],
            "progress": {
                "step": "complete", "models_total": 2, "models_done": 2,
                "models_succeeded": 2,
                "models": {
                    "gemini:old-a": {"state": "done", "elapsed_s": 300},
                    "pi:old-b": {"state": "done", "elapsed_s": 240},
                },
            },
        })

        fake_state = MagicMock()
        fake_state.get = MagicMock(
            side_effect=lambda k, default=None: {
                "tracked_files": [{"path": "src/a.py"}],
                "head_before_implement": "abc123",
                "head_at_start": "abc123",
                "pre_existing_changes": [],
            }.get(k, default)
        )
        fake_state.state = {"phases_completed": {}}
        fake_state.is_phase_completed = MagicMock(return_value=False)

        # The window is between the launch `write_status` and the tracker's
        # seed: the seed replaces `models` wholesale, so a final-state assertion
        # would pass either way. `report_progress` is the first call after that
        # write, so the disk as it is entered is what a poll landing there reads.
        snapshots = []
        real = cro.report_progress

        def record(phase_dir, **updates):
            if not snapshots:
                snapshots.append(
                    status_file.read_status(str(phase_dir)).get("progress")
                )
            real(phase_dir, **updates)

        with patch.object(cro, "parse_args",
                          return_value=_make_args(plan_path)), \
             patch.object(cro, "get_or_create_state", return_value=fake_state), \
             patch.object(cro, "resolve_models",
                          return_value=["cursor-agent:auto"]), \
             patch.object(cro, "is_model_valid", return_value=True), \
             patch.object(cro, "validate_git_ref",
                          side_effect=lambda r: r or "HEAD~1"), \
             patch.object(cro, "report_progress", record), \
             patch.object(cro, "run_all_reviews", new=AsyncMock(
                 return_value={"cursor-agent:auto": (True, [], None)})), \
             patch.object(cro, "generate_review_report",
                          return_value="# report\n"), \
             patch.object(cro, "get_phase_dir", return_value=phase):
            asyncio.run(cro.main())

        assert snapshots, "the launch never reported progress"
        assert not snapshots[0], (
            f"the predecessor's progress survived the launch write: "
            f"{snapshots[0]}"
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

        def fake_ctx(files, **kwargs):
            parent.intent_to_add_untracked(files, **kwargs)
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

        def fake_ctx(files, **kwargs):
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

        def fake_ctx(files, **kwargs):
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

        def fake_ctx(files, **kwargs):
            parent.intent_to_add_untracked(files, **kwargs)
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


class TestAnInProcessSalvageMovesTheMapWithTheReport:
    """F3, 2026-09-20, the same defect on this lane.

    Under ``--exec-subagents`` the run repairs an unparseable review, folds the
    issues back into ``results`` and ``del failed_models[spec]`` — so the
    checkpoint below lists that model under ``models_completed`` and restates
    ``models_succeeded`` to include it, while the per-model map went on saying
    ``failed`` with the pre-salvage error on the row.  ``tracker.done()`` cannot
    fix it: terminal transitions are idempotent, so a second one against a
    frozen ``failed`` entry is a silent no-op.
    """

    def test_a_salvaged_model_is_done_on_the_map_not_only_in_the_counters(
        self, tmp_path
    ):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        specs = ["cursor-agent:auto", "claude-code:sonnet"]
        good, bad = specs
        issues = [{"title": "Bug", "desc": "d", "importance": "low",
                   "file": "src/a.py", "type": "bug"}]
        phase_dir = cro.get_phase_dir(plan_path, "code-review")

        fake_state = MagicMock()
        fake_state.get = MagicMock(
            side_effect=lambda k, default=None: {
                "tracked_files": [{"path": "src/a.py"}],
                "head_before_implement": "abc123",
                "head_at_start": "abc123",
                "pre_existing_changes": [],
            }.get(k, default)
        )
        fake_state.mark_phase_completed = MagicMock()
        fake_state.save = MagicMock()
        fake_state.is_phase_completed = MagicMock(return_value=False)
        fake_state.state = {"phases_completed": {}}

        args = _make_args(plan_path, models=list(specs), exec_subagents=True)

        async def fake_run_all_reviews(*_a, **kw):
            # A real fan-out's two endings, through the real tracker, so the map
            # this test starts from is the one production leaves behind.
            phase_dir.mkdir(parents=True, exist_ok=True)
            tracker = progress_tracker.FanoutProgress(
                str(phase_dir), specs, anchor=str(plan_path), interval_s=0
            )
            kw["tracker_out"].append(tracker)
            tracker.seed()
            tracker.done(good, source="file")
            tracker.failed(bad, error="Expected JSON array, got dict", source="stdout")
            return {good: (True, issues, None),
                    bad: (False, None, "Expected JSON array, got dict")}

        def fake_salvage(_args, *, phase_dir, phase, plan_path=None):
            recovered = Path(phase_dir) / f"{cro.sanitize_model_name(bad)}.json"
            recovered.write_text(json.dumps(issues), encoding="utf-8")
            return [str(recovered)]

        with patch.object(cro, "parse_args", return_value=args), \
             patch.object(cro, "get_or_create_state", return_value=fake_state), \
             patch.object(cro, "resolve_models", return_value=list(specs)), \
             patch.object(cro, "is_model_valid", return_value=True), \
             patch.object(cro, "validate_git_ref",
                          side_effect=lambda r: r or "HEAD~1"), \
             patch.object(cro, "dispatch_salvage_tasks", new=fake_salvage), \
             patch.object(cro, "run_all_reviews", new=fake_run_all_reviews):
            assert asyncio.run(cro.main()) == 0

        status = status_file.read_status(str(phase_dir))
        assert status["models_failed"] == {}
        assert sorted(status["models_completed"]) == sorted(specs)
        progress = status["progress"]
        assert progress["models_succeeded"] == 2

        entry = progress["models"][bad]
        assert entry["state"] == "done", entry
        # Not `existing_file` (2026-09-21): that word is in
        # `progress_tracker.RESUME_SOURCES` / `run_status.SKIPPED_SOURCES` and
        # renders the row **skipped**, which is a lie about a slot that
        # dispatched, ran, failed to parse and was repaired in-process.
        assert entry["source"] == "salvage", entry
        assert entry["source"] not in progress_tracker.RESUME_SOURCES
        assert "error" not in entry, entry
        # The freeze stands: this is a later reading of the same slot's outcome,
        # not a second run of it.
        assert entry["ended_at"] and "elapsed_s" in entry


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


class TestOneModelsMalformedElementCannotStrandTheOthers:
    """2026-09-20: a `--review-code --quick` fan-out of five lost all of it.

    ``cursor-agent:kimi-k3-high`` reviewed cleanly and ended its final message
    on ``[]``; the stdout fallback scraped the ``[0]`` out of its prose sentence
    "``unattended.ROWS[0].site`` correctly left as-is" instead (clause 5 of
    ``extract_json_from_text`` is the root-cause fix), so its result file held
    one bare integer. ``validate_code_review_issues`` reported it exactly right
    — ``Warning: Validation errors: ['[0]: expected object, got int']`` — and
    only warned; aggregation then died on ``issue["model"] = model_spec``
    (``TypeError: 'int' object does not support item assignment``) and the
    eleven findings the other four models had already produced were never
    written. ``--reaggregate``, the documented recovery, died at the same
    statement one function over.

    Both aggregation legs are exercised on one malformed payload here, as
    ``TestEmptyReviewReplacesPreviousFindings`` does, because they are two
    implementations of one rule: one model's malformed element costs that
    element and nothing else.
    """

    #: The other reviewer's finding, and the thing the phase used to lose.
    SURVIVING_ISSUE = {
        "title": "Bound the retry ladder",
        "desc": "The backoff loop has no deadline.",
        "importance": "high",
        "file": "src/a.py",
        "type": "bug",
    }

    @pytest.mark.parametrize("reaggregate", [False, True])
    def test_the_other_models_findings_still_reach_the_report(
        self, tmp_path, capsys, reaggregate
    ):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        phase_dir.mkdir(parents=True, exist_ok=True)
        # What the extractor wrote for the clean reviewer: an array of one int.
        (phase_dir / "cursor-agent_kimi-k3-high.json").write_text(
            "[\n  0\n]", encoding="utf-8"
        )
        (phase_dir / "claude-code_sonnet.json").write_text(
            json.dumps([self.SURVIVING_ISSUE]), encoding="utf-8"
        )

        fake_state = MagicMock()
        fake_state.get.side_effect = lambda key, default=None: {
            "tracked_files": [{"path": "src/a.py"}],
            "head_before_implement": "abc123",
            "head_at_start": "abc123",
            "pre_existing_changes": [],
        }.get(key, default)
        fake_state.is_phase_completed.return_value = False
        fake_state.state = {"phases_completed": {}}
        models = ["cursor-agent:kimi-k3-high", "claude-code:sonnet"]
        args = _make_args(
            plan_path, models=models, force=True, skip_validation=True,
            reaggregate=reaggregate,
        )
        fanout_results = {
            "cursor-agent:kimi-k3-high": (True, [0], None),
            "claude-code:sonnet": (True, [dict(self.SURVIVING_ISSUE)], None),
        }

        with patch.object(cro, "parse_args", return_value=args), \
             patch.object(cro, "get_or_create_state", return_value=fake_state), \
             patch.object(cro, "resolve_models", return_value=models), \
             patch.object(cro, "is_model_valid", return_value=True), \
             patch.object(cro, "validate_git_ref", side_effect=lambda ref: ref or "HEAD~1"), \
             patch.object(cro, "build_unsanitize_map", return_value={
                 "cursor-agent_kimi-k3-high": "cursor-agent:kimi-k3-high",
                 "claude-code_sonnet": "claude-code:sonnet",
             }), \
             patch.object(cro, "run_all_reviews",
                          new=AsyncMock(return_value=fanout_results)):
            if reaggregate:
                asyncio.run(cro.reaggregate_from_existing_files(
                    "plan", phase_dir.parent, plan_path, args
                ))
            else:
                assert asyncio.run(cro.main()) == 0

        issues = json.loads((phase_dir / "issues.json").read_text(encoding="utf-8"))
        assert [issue["title"] for issue in issues] == ["Bound the retry ladder"]
        assert issues[0]["model"] == "claude-code:sonnet"

        report = (phase_dir / "report.md").read_text(encoding="utf-8")
        assert "**Total Issues:** 1" in report
        assert "Bound the retry ladder" in report
        # The malformed reviewer still completed; it is not a failed review.
        assert "## Failed Reviews" not in report

        # Named, not swallowed: the model and the element it produced.
        warning = capsys.readouterr().err
        assert "cursor-agent:kimi-k3-high" in warning
        assert "0 (int)" in warning

    def test_generate_review_report_drops_the_element_and_keeps_the_rest(
        self, tmp_path, capsys
    ):
        """The report generator's own site, reached by every leg above."""
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        results = {
            "cursor-agent:kimi-k3-high": (True, [0], None),
            "claude-code:sonnet": (True, [dict(self.SURVIVING_ISSUE)], None),
        }

        report = cro.generate_review_report(
            plan_path, results, changed_files=["src/a.py"]
        )

        assert "**Total Issues:** 1" in report
        assert "Bound the retry ladder" in report
        models_line = next(
            line for line in report.splitlines() if line.startswith("**Models:**")
        )
        assert "claude-code:sonnet" in models_line
        assert "cursor-agent:kimi-k3-high" in models_line
        assert "cursor-agent:kimi-k3-high" in capsys.readouterr().err

    def test_a_result_file_that_is_not_a_list_at_all_is_survivable_too(
        self, tmp_path, capsys
    ):
        """An object where an array belongs iterates to its *keys* — strings,

        which fail on the identical item assignment. Same disposition.
        """
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        results = {
            "cursor-agent:kimi-k3-high": (True, {"issues": []}, None),
            "claude-code:sonnet": (True, [dict(self.SURVIVING_ISSUE)], None),
        }

        report = cro.generate_review_report(
            plan_path, results, changed_files=["src/a.py"]
        )

        assert "**Total Issues:** 1" in report
        assert "cursor-agent:kimi-k3-high" in capsys.readouterr().err


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


# ---------------------------------------------------------------------------
# Plan review-run-scope-salvage-and-failure-visibility §5 (T5.2): the lost
# reviewer is named in the header, and a salvage this attempt asked for says so
# ---------------------------------------------------------------------------

_PI_SPEC = "pi:openrouter/deepseek/deepseek-v4.1-flash"
_PI_REASON = "pi timed out after 1200s (spent $0.09 est., 28,193 output tokens)"
_PI_STEM = cro.sanitize_model_name(_PI_SPEC)
_STARTED_AT = "2026-09-25T15:58:00"


class TestTheModelsFailedHeaderLine:
    """``**Models:**`` names the models that produced results; the one that
    produced nothing used to appear only in ``## Failed Reviews`` at the very
    bottom — and on the reaggregate path, not at all."""

    ANSWERED = ["cursor-agent:grok-4.7-high", "cursor-agent:kimi-k3-high",
                "codex:gpt-6-astra:xhigh", "claude-code:fable-5-1:high"]

    def _report(self, tmp_path, *, phase_dir=None, requested=5, started_at=None):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        if phase_dir is not None and started_at is not None:
            status_file.write_status(str(phase_dir), {
                "phase": "code-review", "state": "models_complete",
                "started_at": started_at,
            })
        results = {spec: (True, [], None) for spec in self.ANSWERED}
        return cro.generate_review_report(
            plan_path, results, changed_files=["x.py"], phase_dir=phase_dir,
            failed_models={_PI_SPEC: _PI_REASON}, requested_models=requested,
        )

    def _phase_dir(self, tmp_path):
        phase_dir = tmp_path / "code-review"
        phase_dir.mkdir()
        return phase_dir

    def _retired_request(self, phase_dir, reason, timestamp, stamp="20260925-160000"):
        name = f"retired_salvage_{_PI_STEM}.{reason}.{stamp}.json"
        (phase_dir / name).write_text(json.dumps({
            "model": _PI_SPEC, "phase": "code_review", "raw_output": "…",
            "timestamp": timestamp,
        }), encoding="utf-8")

    @staticmethod
    def _failed_line(report):
        lines = report.splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith("**Models:**"))
        return lines[index + 1]

    def test_the_line_sits_directly_under_models_with_the_reason_verbatim(
        self, tmp_path
    ):
        report = self._report(tmp_path)
        assert self._failed_line(report) == (
            f"**Models failed:** 1 of 5 requested — {_PI_SPEC}: {_PI_REASON}"
        )
        # The bottom section stays.
        assert f"- **{_PI_SPEC}**: {_PI_REASON}" in report

    def test_no_failure_writes_no_line(self, tmp_path):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        report = cro.generate_review_report(
            plan_path, {"a:b": (True, [], None)}, changed_files=[],
            failed_models={}, requested_models=1,
        )
        assert "**Models failed:**" not in report

    def test_a_live_salvage_request_adds_the_clause(self, tmp_path):
        phase_dir = self._phase_dir(tmp_path)
        (phase_dir / f"salvage_{_PI_STEM}.json").write_text("{}", encoding="utf-8")
        report = self._report(tmp_path, phase_dir=phase_dir)
        assert self._failed_line(report).endswith(
            f"{_PI_SPEC}: {_PI_REASON}; salvage requested, nothing recovered"
        )

    def test_an_unrecoverable_request_from_this_attempt_adds_the_clause(
        self, tmp_path
    ):
        phase_dir = self._phase_dir(tmp_path)
        self._retired_request(phase_dir, "unrecoverable", "2026-09-25T16:20:00")
        report = self._report(tmp_path, phase_dir=phase_dir, started_at=_STARTED_AT)
        assert "; salvage requested, nothing recovered" in self._failed_line(report)

    @pytest.mark.parametrize("reason, timestamp, started_at", [
        # A previous attempt's pair, retired when this attempt began.
        ("superseded", "2026-09-25T16:20:00", _STARTED_AT),
        # A previous run's unrecoverable verdict survives the next attempt.
        ("unrecoverable", "2026-09-24T09:00:00", _STARTED_AT),
        # A checkpoint without `started_at` gets no clause from this route.
        ("unrecoverable", "2026-09-25T16:20:00", None),
    ])
    def test_no_clause_from_a_request_this_attempt_did_not_make(
        self, tmp_path, reason, timestamp, started_at
    ):
        phase_dir = self._phase_dir(tmp_path)
        self._retired_request(phase_dir, reason, timestamp)
        report = self._report(tmp_path, phase_dir=phase_dir, started_at=started_at)
        assert "salvage requested" not in report

    def test_the_batch_report_is_not_evidence(self, tmp_path):
        """The validation dispatch that follows the salvage writes the same bare
        name, so the salvage verdict is usually gone by report time."""
        phase_dir = self._phase_dir(tmp_path)
        (phase_dir / "subagent_batch_report.json").write_text(json.dumps({
            "tasks": [{"task_id": f"salvage_{_PI_STEM}",
                       "status": "salvage_unrecoverable"}],
        }), encoding="utf-8")
        report = self._report(tmp_path, phase_dir=phase_dir, started_at=_STARTED_AT)
        assert "salvage requested" not in report

    def test_the_html_notice_is_the_same_line_without_its_bold(self, tmp_path):
        phase_dir = self._phase_dir(tmp_path)
        (phase_dir / f"salvage_{_PI_STEM}.json").write_text("{}", encoding="utf-8")
        report = self._report(tmp_path, phase_dir=phase_dir)
        data = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
        assert data["failedModels"] == {_PI_SPEC: _PI_REASON}
        assert data["requestedModels"] == 5
        [notice] = [n for n in data["notices"] if n["type"] == "warning"]
        assert notice["message"] == self._failed_line(report).replace("**", "")


class TestTheFanoutReportsProgressWhileItRuns:
    """F6: `run_all_reviews` never advanced `progress.models_done` — it was
    written `0` once before dispatch and recomputed only after the last model
    returned, so a mid-run poll read `models 0/4` beside three finished
    reviewers (measured 2026-09-16: 646 s of an 833 s run). The counter was
    the base's `fanout_slot_counter` (shared with `run_all_models`, tested in
    test_review_plan_orchestrator.py), which reported through the base's own
    `_detach`; the tracker writes through that same module-level door
    (`detach.report_progress`), so it is still the seam patched here.

    That counter is **deleted** now and the per-model map has taken the key
    over, because a map-derived count beside a counter is two writers for one
    key and they can only disagree: the counter ticked at slot exit while the
    map turns terminal at the completion print, so a lane running both flickers
    between counter-order and map-order.  The consolidation changes the *call
    sequence* — a tracker that seeds the map and writes on every transition
    makes many more `report_progress` calls than one tick per slot — so these
    tests assert the property the F6 regression was actually about rather than
    the literal `[1, 2, 3]` / `[1, 2]` they used to pin: the number read off
    `models_done` never decreases, it reaches `models_total`, and on a resumed
    run it is already at the skipped count while the models that really dispatch
    are still running, instead of sitting at `0` for the length of the fan-out.
    """

    def test_models_done_never_decreases_and_reaches_the_total(self, tmp_path):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        calls = []

        def record(phase_dir, **updates):
            calls.append(dict(updates))

        with patch.object(cro, "run_single_review", new=_answering_review()), \
             patch.object(cro, "PROVIDER_STAGGER_DELAY", 0.0), \
             patch.object(base._detach, "report_progress", record):
            results = asyncio.run(cro.run_all_reviews(
                ["cursor-agent:auto", "claude-code:sonnet", "codex:gpt-5.5"],
                plan_path, tmp_path, 30, 3,
            ))

        assert len(results) == 3
        counted = [c["models_done"] for c in calls if "models_done" in c]
        # Many more writes than slots now (seed, stagger, dispatch, completion),
        # so the shape that matters is the monotonicity, not the length.
        assert counted, "the fan-out wrote no progress at all"
        assert counted == sorted(counted), f"models_done walked backwards: {counted}"
        assert counted[-1] == 3
        assert all(c["models_total"] == 3 for c in calls if "models_total" in c)
        # The other half of the pair, which the slot counter never wrote: every
        # slot produced a usable result here, so the two numbers end equal.
        succeeded = [c["models_succeeded"] for c in calls if "models_succeeded" in c]
        assert succeeded[-1] == 3

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

        with patch.object(cro, "run_single_review", new=_answering_review()), \
             patch.object(cro, "PROVIDER_STAGGER_DELAY", 0.0), \
             patch.object(base._detach, "report_progress", record):
            results = asyncio.run(cro.run_all_reviews(
                ["cursor-agent:auto", "claude-code:sonnet"],
                plan_path, tmp_path, 30, 2,
            ))

        assert len(results) == 2
        counted = [c["models_done"] for c in calls if "models_done" in c]
        assert counted == sorted(counted), f"models_done walked backwards: {counted}"
        assert counted[-1] == 2

        # D4a on the one lane where nothing dispatches at all: by the time the
        # model that really runs is dispatched, the skipped one is already
        # counted. The old failure is exactly the `0` this forbids.
        dispatching = [
            call for call in calls
            if any(
                entry["state"] == "running"
                for entry in (call.get("models") or {}).values()
            )
        ]
        assert dispatching, "no write caught a model in flight"
        assert dispatching[0]["models_done"] == 1
        assert dispatching[0]["models_succeeded"] == 1


class TestThisLanesOwnFanoutCarriesTheMapAndTheTicker:
    """review-code reaches neither shared helper.

    `run_all_reviews` is a duplicate fan-out — its own `run_with_semaphore`,
    its own stagger loop, its own `asyncio.gather` — so instrumenting
    `run_models_concurrent` reaches nothing here.  These assert the three
    properties on *this* function's own code path: the map is complete at the
    first poll rather than filling in as models start, a `[PROGRESS]` ticker
    really ran while it was fanning out, and nothing is left live once it
    returns.
    """

    def test_the_map_is_complete_at_the_first_write(self, tmp_path):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        specs = ["cursor-agent:auto", "claude-code:sonnet", "codex:gpt-5.5"]
        calls = []
        real_report_progress = base._detach.report_progress

        def record(phase_dir, **updates):
            calls.append(dict(updates))
            real_report_progress(phase_dir, **updates)

        with patch.object(cro, "run_single_review", new=_answering_review()), \
             patch.object(cro, "PROVIDER_STAGGER_DELAY", 0.0), \
             patch.object(base._detach, "report_progress", record):
            asyncio.run(cro.run_all_reviews(specs, plan_path, tmp_path, 30, 3))

        seeded = calls[0]["models"]
        assert sorted(seeded) == sorted(specs), (
            "a poller's first read has to see every requested model"
        )
        assert {entry["state"] for entry in seeded.values()} == {"queued"}
        for entry in seeded.values():
            # A queued row renders `0m00s/30m`, not a blank denominator.
            assert entry["timeout_s"] == 30
            assert entry["attempt"] == 0
            assert entry["source"] is None
            assert entry["started_at"]

        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        stored = status_file.read_status(str(phase_dir))["progress"]["models"]
        assert {entry["state"] for entry in stored.values()} == {"done"}

    def test_a_ticker_runs_on_this_functions_own_code_path(self, tmp_path):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        clock = _FakeClock()
        emitted = []

        async def fake_review(model_spec, *args, **kwargs):
            kwargs["tracker"].running(model_spec, attempt=1)
            # Three suspension points, so the ticker task the fan-out started
            # is scheduled whatever order the loop picks — and so at least one
            # line is rendered while a row really is `running`.
            for _ in range(3):
                await asyncio.sleep(0)
            kwargs["tracker"].done(model_spec, source="file")
            return True, [], None

        with patch.object(cro, "run_single_review", new=fake_review), \
             patch.object(cro, "PROVIDER_STAGGER_DELAY", 0.0), \
             patch.object(progress_tracker, "_monotonic", clock.monotonic), \
             patch.object(progress_tracker, "_tick_sleep", clock.sleep), \
             patch.object(progress_tracker, "_emit", emitted.append):
            asyncio.run(cro.run_all_reviews(
                ["cursor-agent:auto", "claude-code:sonnet"],
                plan_path, tmp_path, 30, 2,
            ))

        assert any("done (" in line for line in emitted), (
            "run_all_reviews never ran a ticker of its own: " + repr(emitted)
        )
        assert all(line.startswith("[PROGRESS] ") for line in emitted)
        # The last line is written after the teardown sweep, so it states the
        # run's real totals rather than whatever the last interval caught.
        assert emitted[-1].startswith("[PROGRESS] 2/2 done (2 ok, 0 failed)")

    def test_nothing_is_left_live_when_the_fanout_returns(self, tmp_path):
        """The mixed run: one model answers, one fails outright."""
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")

        async def fake_review(model_spec, *args, **kwargs):
            tracker = kwargs.get("tracker")
            if model_spec == "claude-code:sonnet":
                tracker.failed(model_spec, error="boom", source="unknown")
                return False, None, "boom"
            tracker.done(model_spec, source="file")
            return True, [], None

        with patch.object(cro, "run_single_review", new=fake_review), \
             patch.object(cro, "PROVIDER_STAGGER_DELAY", 0.0):
            asyncio.run(cro.run_all_reviews(
                ["cursor-agent:auto", "claude-code:sonnet"],
                plan_path, tmp_path, 30, 2,
            ))

        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        progress = status_file.read_status(str(phase_dir))["progress"]
        states = {spec: e["state"] for spec, e in progress["models"].items()}
        assert states == {
            "cursor-agent:auto": "done", "claude-code:sonnet": "failed",
        }
        assert progress["models_done"] == 2
        assert progress["models_succeeded"] == 1
        assert progress["models"]["claude-code:sonnet"]["error"] == "boom"
        # `error` is terminal-only and never lands on a `done` row, so its
        # presence is itself the signal that something went wrong there.
        assert "error" not in progress["models"]["cursor-agent:auto"]


class TestASlotWaitingOnTheSemaphoreReadsQueuedOnThisFanout:
    """2026-09-21, this lane's copy: the `max_parallel` wait read `staggering`.

    `run_all_reviews` is its own fan-out, so the cure landed in
    `run_models_concurrent` reaches nothing here. `staggering` fires *before*
    both semaphores are acquired and `running` only inside
    `run_single_review`, so a slot whose stagger sleep had already elapsed and
    which was parked on `max_parallel` kept advertising a stagger it had
    spent — while index 0, which never staggers, rendered the same wait as
    `queued`. One wait, two words, on the one lane that still had it.
    """

    MODELS = ["cursor-agent:auto", "claude-code:sonnet"]

    def test_the_parked_slot_reads_queued_and_not_staggering(self, tmp_path):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        head, tail = self.MODELS
        dispatched = asyncio.Event()
        release = asyncio.Event()
        seen = {}
        trackers: list = []

        async def fake_review(model_spec, *args, **kwargs):
            tracker = kwargs["tracker"]
            tracker.running(model_spec, attempt=1)
            if model_spec == head:
                dispatched.set()
                await release.wait()
            tracker.done(model_spec, source="file")
            return True, [], None

        async def drive():
            task = asyncio.ensure_future(cro.run_all_reviews(
                list(self.MODELS), plan_path, tmp_path, 30, 1,
                tracker_out=trackers,
            ))
            # `max_parallel=1`, so while the head slot holds the semaphore the
            # tail slot can only be waiting for it.
            await dispatched.wait()
            tracker = trackers[0]
            deadline = time.monotonic() + 2.0
            state = tracker.snapshot()[tail]["state"]
            while state == "staggering" and time.monotonic() < deadline:
                await asyncio.sleep(0.005)
                state = tracker.snapshot()[tail]["state"]
            seen["tail"] = state
            seen["head"] = tracker.snapshot()[head]["state"]
            release.set()
            await task

        with patch.object(cro, "run_single_review", new=fake_review), \
             patch.object(cro, "PROVIDER_STAGGER_DELAY", 0.0):
            asyncio.run(drive())

        assert seen["head"] == "running", seen
        assert seen["tail"] == "queued", (
            "a slot parked on `max_parallel` must not advertise a stagger it "
            f"has already spent: {seen}"
        )
        # And the ending is unchanged: both slots still retire.
        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        states = {
            spec: entry["state"] for spec, entry in
            status_file.read_status(str(phase_dir))["progress"]["models"].items()
        }
        assert states == {head: "done", tail: "done"}, states


class TestTheStaggeringRowCarriesItsCountdownOnThisFanout:
    """2026-09-21, this lane's copy: `run_all_reviews` names the sleep it takes.

    The base's `run_models_concurrent` reaches nothing here, so the one
    `tracker.staggering` call this fan-out makes has to pass `delay_s` itself
    — without it the row reads `staggering` with no `starts in Ns`, on the
    one lane a user watching a `--review-code` launch would see.
    """

    MODELS = ["cursor-agent:auto", "claude-code:sonnet", "codex:gpt-5.5"]

    def test_every_staggered_slot_is_stamped_with_its_own_delay(self, tmp_path):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        seen = {}
        real = progress_tracker.FanoutProgress.staggering

        def spy(self, spec, **kwargs):
            real(self, spec, **kwargs)
            # The row as a poll landing on this write would read it.
            seen[spec] = (kwargs, dict(self.snapshot()[spec]))

        async def fake_review(model_spec, *args, **kwargs):
            tracker = kwargs["tracker"]
            tracker.running(model_spec, attempt=1)
            tracker.done(model_spec, source="file")
            return True, [], None

        with patch.object(cro, "run_single_review", new=fake_review), \
             patch.object(cro, "PROVIDER_STAGGER_DELAY", 0.02), \
             patch.object(progress_tracker.FanoutProgress, "staggering", spy):
            asyncio.run(cro.run_all_reviews(
                list(self.MODELS), plan_path, tmp_path, 30, 3,
            ))

        head, *tail = self.MODELS
        assert head not in seen, "index 0 never staggers"
        assert list(seen) == tail
        for index, spec in enumerate(tail, start=1):
            kwargs, row = seen[spec]
            # This lane sleeps the bare product; it does not route the delay
            # through `get_backoff_delay`, and the row must say what it slept.
            assert kwargs == {"delay_s": pytest.approx(index * 0.02)}, (spec, kwargs)
            assert row["state"] == "staggering"
            assert "stagger_until" in row, (spec, row)
        # And once dispatched, no row still carries a countdown.
        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        rows = status_file.read_status(str(phase_dir))["progress"]["models"]
        assert not [spec for spec, e in rows.items() if "stagger_until" in e]


class TestEveryEndingRetiresEverySlotOnThisFanout:
    """The sweep is in the `finally`, not on the timeout branch, and this is
    the fan-out that proves why.

    It used to prove it the hard way: `tasks` held **bare coroutines**, so the
    first `task.done()` in the `except asyncio.TimeoutError` handler raised
    `AttributeError` and the deadline branch could not finish — and the sweep,
    hanging off the `finally` instead, retired every slot anyway.  F2
    (2026-09-20) made them real tasks, because harvesting the slots that had
    already answered needs a handle on them, so that branch now runs to the end
    and `run_all_reviews` returns the partial results instead of raising out of
    the orchestrator.  The sweep's placement is unchanged and still right: the
    `finally` runs whichever way the fan-out ends.
    """

    def test_the_deadline_path_retires_every_slot_with_the_reports_wording(
        self, tmp_path
    ):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        specs = ["cursor-agent:auto", "claude-code:sonnet"]

        async def never_returns(*args, **kwargs):
            await asyncio.Event().wait()

        # `timeout=0` makes `total_timeout` 0 with the stagger patched out, so
        # `asyncio.wait_for` fires its deadline on the first suspension.
        with patch.object(cro, "run_single_review", new=never_returns), \
             patch.object(cro, "PROVIDER_STAGGER_DELAY", 0.0):
            results = asyncio.run(
                cro.run_all_reviews(specs, plan_path, tmp_path, 0, 2)
            )

        # F2, 2026-09-20: this call used to be wrapped in
        # `pytest.raises(AttributeError)` — the deadline branch crashed on
        # `task.done()` over a bare coroutine and the whole fan-out raised out
        # of `main()`, losing every review a slot HAD produced.  It returns now;
        # nothing finished here, so the harvest finds nothing to keep.
        assert results == {}

        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        progress = status_file.read_status(str(phase_dir))["progress"]
        models = progress["models"]
        assert not [
            spec for spec, entry in models.items()
            if entry["state"] in ("queued", "staggering", "running", "retrying")
        ], "a slot outlived the fan-out"

        # One cause, named the same way in the table and in the report's
        # failure list: the map's failed set is exactly what F11's backfill
        # produced for the same run, error string included.
        failed_models = {}
        base.backfill_deadline_losses(specs, {}, failed_models)
        assert failed_models == {
            spec: progress_tracker.DEADLINE_ERROR for spec in specs
        }
        assert {
            spec for spec, entry in models.items() if entry["state"] == "failed"
        } == set(failed_models)
        assert {spec: entry["error"] for spec, entry in models.items()} == failed_models
        assert progress["models_done"] == 2
        assert progress["models_succeeded"] == 0

    def test_the_deadline_keeps_the_reviews_of_the_slots_that_finished(
        self, tmp_path
    ):
        """F2, 2026-09-20: the mixed ending, on this lane.

        `asyncio.wait_for` raises instead of assigning `completed`, so the
        deadline branch returned nothing at all — including for the reviewer
        that had already answered and whose `<model>.json` `save_model_result`
        had written. `backfill_deadline_losses` walks `results`, so that model
        was named a deadline loss in the report and in `models_failed` while the
        map (written by `run_single_review`'s own completion site) called it
        `done`, and the `models_succeeded` restatement counted it as lost.
        """
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        specs = ["cursor-agent:auto", "claude-code:sonnet"]
        issues = [{"title": "Bug", "desc": "d", "importance": "low",
                   "file": "src/a.py", "type": "bug"}]

        async def one_answers_one_straggles(model_spec, *args, **kwargs):
            if model_spec == specs[1]:
                await asyncio.Event().wait()
            kwargs["tracker"].done(model_spec, source="file")
            return True, issues, None

        with patch.object(cro, "run_single_review", new=one_answers_one_straggles), \
             patch.object(cro, "PROVIDER_STAGGER_DELAY", 0.0):
            results = asyncio.run(
                cro.run_all_reviews(specs, plan_path, tmp_path, 0.05, 2)
            )

        assert list(results) == [specs[0]]
        assert results[specs[0]] == (True, issues, None)

        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        models = status_file.read_status(str(phase_dir))["progress"]["models"]
        assert models[specs[0]]["state"] == "done"
        assert models[specs[1]]["state"] == "failed"

        # The report's failure list and the map name the same set — which is
        # what the fix buys: F11 backfills only the model that produced nothing.
        failed_models = {}
        base.backfill_deadline_losses(specs, results, failed_models)
        assert failed_models == {specs[1]: progress_tracker.DEADLINE_ERROR}
        assert {
            spec for spec, entry in models.items() if entry["state"] == "failed"
        } == set(failed_models)

    def test_a_slot_that_raises_records_its_own_ending_not_a_cancellation(
        self, tmp_path
    ):
        """F5, 2026-09-20: two lanes, one event, two different answers.

        `run_all_reviews` had no `except Exception` around `run_single_review`,
        so a slot that raised past the transition sites inside it reached no
        terminal transition of its own and was retired by the teardown sweep as
        `cancelled` / `source: null` — the wording reserved for a slot whose ending
        runs no transition site at all (a `KeyboardInterrupt` between slots, a
        task the gather drops; NOT `--abort`, which runs no `finally` here at
        all and is retired on disk instead).  The base lane's
        `run_models_concurrent` records `source: "exception"` and the
        exception's own text for the identical event, and the base lane is the
        one that is right: the exception IS the ending, and it is the only place
        the spec behind a task the gather drops is still known.  The sibling
        assertion lives in
        `tests/test_review_plan_orchestrator.py::TestEveryEndingRetiresEverySlot`.
        """
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        specs = ["cursor-agent:auto", "claude-code:sonnet"]

        async def one_explodes(model_spec, *args, **kwargs):
            if model_spec == specs[1]:
                raise RuntimeError("provider exploded")
            kwargs["tracker"].done(model_spec, source="file")
            return True, [], None

        with patch.object(cro, "run_single_review", new=one_explodes), \
             patch.object(cro, "PROVIDER_STAGGER_DELAY", 0.0):
            results = asyncio.run(
                cro.run_all_reviews(specs, plan_path, tmp_path, 30, 2)
            )

        # The raising slot never reaches `results` — it is printed as `Review
        # exception:` and dropped — which is exactly why the map has to carry
        # its reason.
        assert list(results) == [specs[0]]
        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        models = status_file.read_status(str(phase_dir))["progress"]["models"]
        loser = models[specs[1]]
        assert loser["state"] == "failed"
        assert loser["error"] == "provider exploded", loser
        assert loser["source"] == "exception", loser
        # Not the sweep's wording, which is what it used to get.
        assert loser["error"] != progress_tracker.CANCELLED_ERROR
        assert models[specs[0]]["state"] == "done"

    def test_the_cancel_all_path_uses_the_other_wording(self, tmp_path):
        """A cancellation that reaches no transition site is `cancelled`, not
        the deadline string. Those two are the only cancellation wordings; a
        third invented at a transition site would split one cause across two
        labels.

        Renamed from `test_an_abort_uses_the_other_wording` (F1, 2026-09-20):
        the body cancels the fan-out task, which is asyncio shutdown's
        cancel-all, and `--abort` never reaches Python here at all. The real
        `--abort` ending is pinned by
        `test_detach.py::TestAbortRetiresTheMapOnDisk`."""
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        dispatched = asyncio.Event()

        async def hang(*args, **kwargs):
            dispatched.set()
            await asyncio.Event().wait()

        async def scenario():
            task = asyncio.ensure_future(cro.run_all_reviews(
                ["cursor-agent:auto", "claude-code:sonnet"],
                plan_path, tmp_path, 30, 2,
            ))
            await dispatched.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        with patch.object(cro, "run_single_review", new=hang), \
             patch.object(cro, "PROVIDER_STAGGER_DELAY", 0.0):
            asyncio.run(scenario())

        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        models = status_file.read_status(str(phase_dir))["progress"]["models"]
        assert {entry["state"] for entry in models.values()} == {"failed"}
        assert {entry["error"] for entry in models.values()} == {
            progress_tracker.CANCELLED_ERROR
        }


class TestAResumedRunDoesNotReportSkippedModelsAsPending:
    """Acceptance criterion 2, on the lane whose skip is inline.

    `run_all_reviews`' resume branch answers from `{model}.json` and returns a
    bare `(True, issues, None)` tuple carrying no `source` at all, so the
    transition site supplies `existing_file` itself.  Left `queued` — the
    default if the skip predicate writes nothing — `models_done` and the
    rendered table would lie about a run that has nothing left to do, and
    `claim_run`'s wholesale drop of the predecessor's `progress` means there is
    no earlier entry to fall back on.
    """

    def test_every_skipped_model_is_done_and_counted(self, tmp_path):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        specs = ["cursor-agent:auto", "claude-code:sonnet"]
        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        phase_dir.mkdir(parents=True, exist_ok=True)
        (phase_dir / "cursor-agent_auto.json").write_text("[]", encoding="utf-8")
        (phase_dir / "claude-code_sonnet.json").write_text("[]", encoding="utf-8")

        async def never_dispatched(*args, **kwargs):
            raise AssertionError("a resumed slot must not dispatch")

        with patch.object(cro, "run_single_review", new=never_dispatched), \
             patch.object(cro, "PROVIDER_STAGGER_DELAY", 0.0):
            results = asyncio.run(
                cro.run_all_reviews(specs, plan_path, tmp_path, 30, 2)
            )

        assert sorted(results) == sorted(specs)
        progress = status_file.read_status(str(phase_dir))["progress"]
        for spec in specs:
            entry = progress["models"][spec]
            assert entry["state"] == "done"
            # There is no `skipped` state: the distinction lives in `source`,
            # which is what a renderer reads to label the row.
            assert entry["source"] == "existing_file"
            assert entry["source"] in progress_tracker.RESUME_SOURCES
            assert entry["attempt"] == 0
            assert entry["elapsed_s"] == 0
        # It has a usable result, so it counts on *both* numbers.
        assert progress["models_done"] == 2
        assert progress["models_succeeded"] == 2
        assert progress["models_total"] == 2


class TestThePhaseTickerCoversTheRestOfThePhase:
    """T5: `run_all_reviews` returns the moment its last slot does, and this
    phase then keeps working for minutes — grouping, the reviewed-code and plan
    excerpts, validation-batch preparation, and under `--exec-subagents` the
    whole validation dispatch and merge.  Without a second ticker that window
    is where the log goes quiet, now at the *end* of the run instead of
    throughout it.  It reuses `progress.subagents`, gets no second map, and
    never touches `models`.
    """

    def test_it_starts_after_the_fanout_and_ends_on_the_terminal_step(
        self, tmp_path
    ):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        clock = _FakeClock()
        emitted = []
        snapshots = {}

        fake_state = MagicMock()
        fake_state.get = MagicMock(
            side_effect=lambda k, default=None: {
                "tracked_files": [{"path": "src/a.py"}],
                "head_before_implement": "abc123",
                "head_at_start": "abc123",
                "pre_existing_changes": [],
            }.get(k, default)
        )
        fake_state.mark_phase_completed = MagicMock()
        fake_state.save = MagicMock()
        fake_state.is_phase_completed = MagicMock(return_value=False)
        fake_state.state = {"phases_completed": {}}

        args = _make_args(plan_path, models=["cursor-agent:auto"])
        phase_dir = cro.get_phase_dir(plan_path, "code-review")

        async def fake_review(model_spec, *a, **kw):
            kw["tracker"].running(model_spec, attempt=1)
            for _ in range(3):
                await asyncio.sleep(0)
            kw["tracker"].done(model_spec, source="file")
            return True, [], None

        real_fanout = cro.run_all_reviews

        async def watched_fanout(*a, **kw):
            results = await real_fanout(*a, **kw)
            # The map exactly as the fan-out's sweep left it.
            snapshots["after_fanout"] = json.dumps(
                status_file.read_status(str(phase_dir))["progress"]["models"],
                sort_keys=True,
            )
            snapshots["lines_at_fanout_end"] = len(emitted)
            return results

        with patch.object(cro, "parse_args", return_value=args), \
             patch.object(cro, "get_or_create_state", return_value=fake_state), \
             patch.object(cro, "resolve_models", return_value=["cursor-agent:auto"]), \
             patch.object(cro, "is_model_valid", return_value=True), \
             patch.object(cro, "validate_git_ref", side_effect=lambda r: r or "HEAD~1"), \
             patch.object(cro, "run_single_review", new=fake_review), \
             patch.object(cro, "run_all_reviews", new=watched_fanout), \
             patch.object(cro, "PROVIDER_STAGGER_DELAY", 0.0), \
             patch.object(progress_tracker, "_monotonic", clock.monotonic), \
             patch.object(progress_tracker, "_tick_sleep", clock.sleep), \
             patch.object(progress_tracker, "_emit", emitted.append):
            assert asyncio.run(cro.main()) == 0

        fanout_lines = emitted[: snapshots["lines_at_fanout_end"]]
        phase_lines = emitted[snapshots["lines_at_fanout_end"] :]
        assert any("done (" in line for line in fanout_lines)
        # Rule 1: the fan-out's ticker dies before the caller's per-model
        # summary, so no `step:` line can interleave with it.
        assert not any("step:" in line for line in fanout_lines)
        # Exactly one, and that is honest rather than a weak assertion:
        # everything between the fan-out and the terminal step is synchronous,
        # so an interval that expires while it runs cannot be serviced. The
        # line that matters is the teardown one, emitted after the phase wrote
        # `progress.step`, so the log's last [PROGRESS] says how the phase
        # ended rather than trailing off at whatever the fan-out last printed.
        assert phase_lines, "the phase after the fan-out narrated nothing"
        assert phase_lines[-1].startswith("[PROGRESS] step: complete")

        # It prints and does not write: the terminal map is byte-identical
        # across the whole of the rest of the phase, so a table rendered while
        # validation runs still shows each model's real duration rather than a
        # clock that kept ticking after the model finished. (The counters
        # restatement beside it therefore carries no `models` key either.)
        after = json.dumps(
            status_file.read_status(str(phase_dir))["progress"]["models"],
            sort_keys=True,
        )
        assert after == snapshots["after_fanout"]

        progress = status_file.read_status(str(phase_dir))["progress"]
        assert progress["step"] == "complete"
        assert progress["models_done"] == 1
        assert progress["models_succeeded"] == 1

    def test_lines_keep_arriving_through_the_in_process_salvage(self, tmp_path):
        """T5, 2026-09-21: this lane's salvage window had no ticker at all.

        In-process salvage (`--exec-subagents`, a model whose output would not
        parse) runs *after* `run_all_reviews` returns — so the fan-out's ticker
        has already been cancelled and awaited by lifetime rule 1 — and
        *before* the phase ticker starts at grouping. `dispatch_salvage_tasks`
        blocks the calling thread the way `dispatch_subagent_tasks` does (one
        `subagent_executor.run_task_file` per request), so a salvage that took
        minutes emitted no `[PROGRESS]` line at all while `progress.step` still
        read `fanout`: the silent gap between the last model and the report
        that T5 forbids. Same cure as the validation window, pinned the same
        way — the blocking stub itself reports how many lines landed while it
        was blocking, which is a statement about the production call site and
        not about the stub.
        """
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        specs = ["cursor-agent:auto", "claude-code:sonnet"]
        good, bad = specs
        issues = [{"title": "Bug", "desc": "d", "importance": "low",
                   "file": "src/a.py", "type": "bug"}]
        phase_dir = cro.get_phase_dir(plan_path, "code-review")

        fake_state = MagicMock()
        fake_state.get = MagicMock(
            side_effect=lambda k, default=None: {
                "tracked_files": [{"path": "src/a.py"}],
                "head_before_implement": "abc123",
                "head_at_start": "abc123",
                "pre_existing_changes": [],
            }.get(k, default)
        )
        fake_state.mark_phase_completed = MagicMock()
        fake_state.save = MagicMock()
        fake_state.is_phase_completed = MagicMock(return_value=False)
        fake_state.state = {"phases_completed": {}}

        args = _make_args(plan_path, models=list(specs), exec_subagents=True)

        async def fake_run_all_reviews(*_a, **kw):
            phase_dir.mkdir(parents=True, exist_ok=True)
            tracker = progress_tracker.FanoutProgress(
                str(phase_dir), specs, anchor=str(plan_path), interval_s=0
            )
            kw["tracker_out"].append(tracker)
            tracker.seed()
            tracker.done(good, source="file")
            tracker.failed(bad, error="Expected JSON array, got dict",
                           source="stdout")
            return {good: (True, issues, None),
                    bad: (False, None, "Expected JSON array, got dict")}

        def phase_lines():
            return [ln for ln in emitted if ln.startswith("[PROGRESS] step: ")]

        window = {}

        def blocking_salvage(_args, *, phase_dir, phase, plan_path=None):
            """`dispatch_salvage_tasks`' shape: it blocks, it does not await."""
            recovered = Path(phase_dir) / f"{cro.sanitize_model_name(bad)}.json"
            recovered.write_text(json.dumps(issues), encoding="utf-8")
            # Read here, not afterwards: the point of the step is that a poll
            # taken DURING the window names the work, rather than still reading
            # the step the fan-out left behind.
            window["step"] = status_file.read_status(phase_dir)["progress"]["step"]
            window["before"] = len(phase_lines())
            deadline = time.monotonic() + 0.2
            while time.monotonic() < deadline:
                time.sleep(0.005)
            window["after"] = len(phase_lines())
            return [str(recovered)]

        with _realtime_ticker() as emitted, \
             patch.object(cro, "parse_args", return_value=args), \
             patch.object(cro, "get_or_create_state", return_value=fake_state), \
             patch.object(cro, "resolve_models", return_value=list(specs)), \
             patch.object(cro, "is_model_valid", return_value=True), \
             patch.object(cro, "validate_git_ref",
                          side_effect=lambda r: r or "HEAD~1"), \
             patch.object(cro, "dispatch_salvage_tasks", new=blocking_salvage), \
             patch.object(cro, "run_all_reviews", new=fake_run_all_reviews):
            assert asyncio.run(cro.main()) == 0

        assert window, "the in-process salvage was never reached"
        assert window["step"] == "salvage", (
            f"a poll during the salvage window still read `{window['step']}`"
        )
        # 0.2 s of blocking against a 10 ms interval: the honest floor is many
        # lines and the defect's floor is zero, so two leaves the margin a
        # loaded machine needs without weakening the discrimination.
        assert window["after"] - window["before"] >= 2, (
            f"the ticker was starved for the whole salvage window: "
            f"{window} in {emitted}"
        )
        assert any(
            ln.startswith("[PROGRESS] step: salvage") for ln in phase_lines()
        ), phase_lines()
        # The window closes on the phase's own terminal step, so the two
        # tickers leave no gap between them either.
        assert phase_lines()[-1].startswith("[PROGRESS] step: complete"), (
            phase_lines()
        )
        # And the run still ends the way it did: the salvaged model is
        # completed, on the report's side and on the map's.
        status = status_file.read_status(str(phase_dir))
        assert status["models_failed"] == {}
        assert sorted(status["models_completed"]) == sorted(specs)
        assert status["progress"]["models"][bad]["state"] == "done"

    def test_the_reaggregate_path_narrates_too(self, tmp_path):
        """`--reaggregate` has no fan-out at all, so it is the phase ticker or
        nothing — and its whole body is synchronous, which is the case that
        catches a ticker out: `start_ticker()` only *schedules* the task, and a
        task cancelled before the loop ever ran it is closed without executing
        its `finally`, so the teardown line that carries the terminal step is
        lost and nothing is printed at all. Asserting on the emitted lines is
        the only assertion that can tell those apart — the checkpoint looks
        identical either way.
        """
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
        emitted = []

        fake_state = MagicMock()
        fake_state.get = MagicMock(
            side_effect=lambda k, default=None: {
                "tracked_files": [{"path": "src/a.py"}],
                "head_before_implement": "abc123",
                "head_at_start": "abc123",
            }.get(k, default)
        )
        fake_state.mark_phase_completed = MagicMock()
        fake_state.save = MagicMock()
        # The reaggregate branch refuses a review older than implement
        # (2026-10-02); a bare MagicMock answers that probe truthily.
        fake_state.phase_completion_is_stale = MagicMock(return_value=False)

        args = _make_args(plan_path, reaggregate=True, skip_validation=True)

        with patch.object(cro, "parse_args", return_value=args), \
             patch.object(cro, "get_or_create_state", return_value=fake_state), \
             patch.object(cro, "validate_git_ref", side_effect=lambda r: r or "HEAD~1"), \
             patch.object(cro, "build_unsanitize_map", return_value={}), \
             patch.object(cro, "maybe_detach", return_value=None), \
             patch.object(progress_tracker, "_emit", emitted.append):
            asyncio.run(cro.main())

        assert emitted, "the reaggregate phase narrated nothing at all"
        assert emitted[-1].startswith("[PROGRESS] step: complete")
        # No fan-out ran, so nothing here may claim per-model progress.
        assert not any("done (" in line for line in emitted)


class TestTheHostValidationHandOffIsTerminal:
    """2026-09-21: neither review-code hand-off said the phase had ended.

    `review_orchestrator_base` writes `state: validation_pending` and
    `progress.step: validation_pending` where it hands validation to the host;
    both of this lane's copies — the fan-out's and `--reaggregate`'s — only
    called `record_next` and returned, leaving `state: models_complete` beside
    `progress.step: grouping`. Neither word is in
    `run_status.CHECKPOINT_TERMINAL_STATES` / `CHECKPOINT_TERMINAL_STEPS`, so
    `_checkpoint_in_flight` read every host-validated code review as still
    running: a foreground one rendered `running in foreground (…)` until
    `--reaggregate` completed, and `next` on a finished detached one waited out
    the whole `poll_s` before exiting 0. The host-driven return IS this
    process's ending, and the checkpoint has to say so.
    """

    ISSUES = [{
        "title": "accel deviates from the plan",
        "desc": "src/a.py sets accel: 32 but the plan says 28",
        "importance": "high", "file": "src/a.py", "line_range": [1, 2],
        "type": "bug",
    }]

    @staticmethod
    def _fake_state():
        fake_state = MagicMock()
        fake_state.get = MagicMock(
            side_effect=lambda k, default=None: {
                "tracked_files": [{"path": "src/a.py"}],
                "head_before_implement": "abc123",
                "head_at_start": "abc123",
                "pre_existing_changes": [],
            }.get(k, default)
        )
        fake_state.mark_phase_completed = MagicMock()
        fake_state.save = MagicMock()
        fake_state.is_phase_completed = MagicMock(return_value=False)
        fake_state.state = {"phases_completed": {}}
        return fake_state

    @staticmethod
    def _tree(tmp_path):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n\n## accel: 28\n", encoding="utf-8")
        (tmp_path / "src").mkdir(exist_ok=True)
        (tmp_path / "src" / "a.py").write_text("accel = 32\n", encoding="utf-8")
        return plan_path

    def test_the_fanout_path_writes_both_words(self, tmp_path):
        plan_path = self._tree(tmp_path)
        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        args = _make_args(plan_path, skip_validation=False)
        emitted = []

        with patch.object(cro, "parse_args", return_value=args), \
             patch.object(cro, "get_or_create_state",
                          return_value=self._fake_state()), \
             patch.object(cro, "resolve_models",
                          return_value=["cursor-agent:auto"]), \
             patch.object(cro, "is_model_valid", return_value=True), \
             patch.object(cro, "validate_git_ref",
                          side_effect=lambda r: r or "HEAD~1"), \
             patch.object(cro, "get_project_root", return_value=str(tmp_path)), \
             patch.object(cro, "run_all_reviews", new=AsyncMock(
                 return_value={"cursor-agent:auto": (True, self.ISSUES, None)})), \
             patch.object(progress_tracker, "_emit", emitted.append):
            asyncio.run(cro.main())

        # The hand-off really happened rather than the run finishing.
        assert (phase_dir / "validation_tasks.json").exists()
        assert not (phase_dir / "report.md").exists()

        status = status_file.read_status(str(phase_dir))
        assert status["state"] == "validation_pending", status.get("state")
        assert status["progress"]["step"] == "validation_pending", status["progress"]
        # Both words are the ones the checkpoint readers treat as an ending,
        # which is the whole point of writing them.
        from utils import run_status
        assert status["state"] in run_status.CHECKPOINT_TERMINAL_STATES
        assert status["progress"]["step"] in run_status.CHECKPOINT_TERMINAL_STEPS
        # The phase ticker's teardown line lands after the write, so the log's
        # last [PROGRESS] names the ending too.
        assert emitted[-1].startswith("[PROGRESS] step: validation_pending"), emitted

    def test_the_reaggregate_path_writes_both_words(self, tmp_path):
        plan_path = self._tree(tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        phase_dir = tmp_path / "phase"
        phase_dir.mkdir()
        (phase_dir / "cursor-agent_auto.json").write_text(
            json.dumps(self.ISSUES), encoding="utf-8"
        )
        args = _make_args(plan_path, reaggregate=True, skip_validation=False)

        with patch.object(cro, "get_or_create_state",
                          return_value=self._fake_state()), \
             patch.object(cro, "get_phase_dir", return_value=phase_dir), \
             patch.object(cro, "validate_git_ref",
                          side_effect=lambda r: r or "HEAD~1"), \
             patch.object(cro, "get_project_root", return_value=str(tmp_path)), \
             patch.object(cro, "build_unsanitize_map", return_value={}):
            asyncio.run(cro.reaggregate_from_existing_files(
                "plan", out_dir, plan_path, args
            ))

        # It stopped at the marker: no report was rebuilt.
        assert (phase_dir / "validation_tasks.json").exists()
        status = status_file.read_status(str(phase_dir))
        assert status["state"] == "validation_pending", status.get("state")
        assert status["progress"]["step"] == "validation_pending", status["progress"]


class TestTheValidationStretchIsNamedBeforeItIsDispatched:
    """The word is written where the stretch begins, not where it may end.

    `review_orchestrator_base` writes `state: validation_pending`
    unconditionally at both of its own materialization sites, right after
    `validation_tasks.json` lands and *before* `dispatch_subagent_tasks`. This
    lane wrote it only inside the host-driven `else` — so under
    `--exec-subagents`, which does not return there, the whole validation
    dispatch ran under `state: models_complete` (the fan-out's last word) beside
    `progress.step: validating`. That pair is the same disagreement the base
    lane fixed on 2026-09-16 (`phase state: grouping_complete` / `progress:
    validating` for the whole of a measured 62 s stretch), and
    `utils/status_file.py` calls the ladder it breaks the in-code contract.

    Not a `--wait` bug — `models_complete` is non-terminal, so
    `_checkpoint_in_flight` kept answering True — which is why it survived the
    hand-off fix above. What it costs is the reading: the `phase state:` line is
    what rule 2 tells a polling agent to read before believing an instant exit
    0, and a dir left at `models_complete` by a crashed or aborted validation
    dispatch names a stretch that had already finished.

    Observed *at dispatch time* rather than after the call, because the write
    under test is transient: on the host-driven path the hand-off overwrites it
    with the same word a few lines later, and on the executor path `complete`
    overwrites it at the end of the phase.
    """

    ISSUES = TestTheHostValidationHandOffIsTerminal.ISSUES

    @staticmethod
    def _spy(phase_dir, seen):
        """Stand in for `dispatch_subagent_tasks`, recording what it can see."""
        def _dispatch(*_args, **_kwargs):
            seen.append(status_file.read_status(str(phase_dir)).get("state"))
            return False  # host-driven: the caller returns at its hand-off
        return _dispatch

    def test_the_fanout_path_names_the_stretch_before_dispatching(self, tmp_path):
        plan_path = TestTheHostValidationHandOffIsTerminal._tree(tmp_path)
        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        args = _make_args(plan_path, skip_validation=False)
        seen = []

        with patch.object(cro, "parse_args", return_value=args), \
             patch.object(cro, "get_or_create_state",
                          return_value=TestTheHostValidationHandOffIsTerminal
                          ._fake_state()), \
             patch.object(cro, "resolve_models",
                          return_value=["cursor-agent:auto"]), \
             patch.object(cro, "is_model_valid", return_value=True), \
             patch.object(cro, "validate_git_ref",
                          side_effect=lambda r: r or "HEAD~1"), \
             patch.object(cro, "get_project_root", return_value=str(tmp_path)), \
             patch.object(cro, "run_all_reviews", new=AsyncMock(
                 return_value={"cursor-agent:auto": (True, self.ISSUES, None)})), \
             patch.object(cro, "dispatch_subagent_tasks",
                          new=self._spy(phase_dir, seen)):
            asyncio.run(cro.main())

        assert (phase_dir / "validation_tasks.json").exists()
        assert seen == ["validation_pending"], (
            "the validation batch was dispatched under "
            f"{seen!r}; the base lane's dispatch runs under "
            "`validation_pending`, and a `--exec-subagents` run never reaches "
            "the host hand-off that would otherwise write it"
        )

    def test_the_reaggregate_path_names_the_stretch_before_dispatching(
        self, tmp_path
    ):
        plan_path = TestTheHostValidationHandOffIsTerminal._tree(tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        phase_dir = tmp_path / "phase"
        phase_dir.mkdir()
        (phase_dir / "cursor-agent_auto.json").write_text(
            json.dumps(self.ISSUES), encoding="utf-8"
        )
        args = _make_args(plan_path, reaggregate=True, skip_validation=False)
        seen = []

        with patch.object(cro, "get_or_create_state",
                          return_value=TestTheHostValidationHandOffIsTerminal
                          ._fake_state()), \
             patch.object(cro, "get_phase_dir", return_value=phase_dir), \
             patch.object(cro, "validate_git_ref",
                          side_effect=lambda r: r or "HEAD~1"), \
             patch.object(cro, "get_project_root", return_value=str(tmp_path)), \
             patch.object(cro, "build_unsanitize_map", return_value={}), \
             patch.object(cro, "dispatch_subagent_tasks",
                          new=self._spy(phase_dir, seen)):
            asyncio.run(cro.reaggregate_from_existing_files(
                "plan", out_dir, plan_path, args
            ))

        assert (phase_dir / "validation_tasks.json").exists()
        assert seen == ["validation_pending"], (
            "the re-dispatched validation batch ran under "
            f"{seen!r} rather than the phase machine's own word for it"
        )

    def test_both_lanes_write_it_at_the_same_point_in_their_source(self):
        """Parity with the base lane, asserted structurally rather than by run.

        The two orchestrators materialize `validation_tasks.json` at four sites
        between them, and the invariant is one sentence: every one of them
        writes `state: validation_pending` before it reaches
        `dispatch_subagent_tasks`. A behavioural test covers this lane's two;
        this covers all four, so a fifth site added to either file is caught by
        the same rule rather than by whoever remembers.
        """
        marker = '{"state": "validation_pending"}'
        dispatch = "dispatch_subagent_tasks,"
        for rel in ("code_review_orchestrator.py",
                    "utils/review_orchestrator_base.py"):
            body = (SKILL_ROOT / rel).read_text(encoding="utf-8")
            sites = [
                i for i in range(len(body))
                if body.startswith("json.dump(batched_tasks", i)
            ]
            assert len(sites) == 2, (
                f"{rel} materializes a validation document at {len(sites)} "
                "sites; the parity rule below is written for two"
            )
            for start in sites:
                nxt = body.index(dispatch, start)
                assert marker in body[start:nxt], (
                    f"{rel} dispatches a validation batch (offset {nxt}) without "
                    "having written `state: validation_pending` since it "
                    f"materialized the document (offset {start}). The stretch "
                    "starts at the write, not at the host hand-off that may "
                    "never be reached."
                )


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
        where bwrap cannot start. The `sh` is real; the line is replayed.

        Its fallback ladder is the uncontained rung alone — codex's rung 2,
        which needs no probe — so the lanes below exercise exactly the
        pre-ladder bypass; the landlock rung has its own tests."""

        internal_retries = None

        def sandbox_probe_command(self, profile):
            from utils.providers.base import PROFILE_GENERATE

            if profile != PROFILE_GENERATE:
                return None
            return ["sh", "-c", f"echo '{_BWRAP_BROKEN_LINE}' >&2; exit 1"]

        def sandbox_fallback_rungs(self, profile):
            from utils.providers.base import PROFILE_GENERATE
            from utils.providers.codex import UNSANDBOXED_RUNG

            return (UNSANDBOXED_RUNG,) if profile == PROFILE_GENERATE else ()

    def test_the_code_review_fanout_refuses_a_slot_whose_sandbox_cannot_start(
        self, tmp_path, monkeypatch, capsys
    ):
        from utils import review_orchestrator_base as rob
        from utils import subagent_executor as executor

        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# Plan\n", encoding="utf-8")
        monkeypatch.setattr("utils.subagent_executor._SANDBOX_PROBE_CACHE", {})
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

    class _AutoFallback(_Probed):
        """The probing adapter declaring the fallback hook at its shipped
        default, `auto` — what codex answers on a fresh install."""

        def sandbox_fallback_policy(self, profile, anchor=None):
            from utils.providers.base import PROFILE_GENERATE

            return "auto" if profile == PROFILE_GENERATE else None

    class _FallbackOff(_Probed):
        """The same adapter under the operator's `sandbox_fallback: off`."""

        def sandbox_fallback_policy(self, profile, anchor=None):
            from utils.providers.base import PROFILE_GENERATE

            return "off" if profile == PROFILE_GENERATE else None

    BYPASS_LINE = (
        "[SANDBOX_BYPASSED] fakebwrap: bubblewrap cannot start; running --sandbox "
        "danger-full-access under providers.fakebwrap.sandbox_fallback=auto "
        "(read-only phase; no OS boundary between the model's shell and this "
        "machine)"
    )

    def _fallback_lane(self, tmp_path, monkeypatch, dispatch, adapter=None):
        from utils import review_orchestrator_base as rob

        adapter = adapter or self._AutoFallback
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# Plan\n", encoding="utf-8")
        monkeypatch.setattr("utils.subagent_executor._SANDBOX_PROBE_CACHE", {})
        monkeypatch.setattr(rob, "get_provider", lambda _name: adapter())
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

        plan_path = self._fallback_lane(tmp_path, monkeypatch, dispatch)
        served = {}
        success, issues, error = asyncio.run(cro.run_single_review(
            "fakebwrap:m1", plan_path, tmp_path / "out", served_models=served,
        ))
        assert (success, issues, error) == (True, [], None)
        assert len(seen) == 1 and seen[0]["sandbox_fallback"] == "unsandboxed"
        # The provenance lands in the collector without a served model beside it.
        assert served == {"fakebwrap:m1": {"sandbox": "bypassed"}}
        out = capsys.readouterr().out
        assert out.count("[SANDBOX_BYPASSED]") == 1
        assert f"[fakebwrap:m1] {self.BYPASS_LINE}" in out

    def test_a_landlock_code_review_slot_records_its_contained_rung(
        self, tmp_path, monkeypatch
    ):
        # A contained rung arrives as its rung id; the review-code collector
        # must record it as `fallback-landlock`, not drop it (2026-09-23).
        def dispatch(**kwargs):
            return {
                "success": True, "data": [], "source": "stdout_fallback",
                "output_file": str(tmp_path / "out" / "o.json"),
                "details": {"sandbox": "landlock"},
            }

        plan_path = self._fallback_lane(tmp_path, monkeypatch, dispatch)
        served = {}
        success, _issues, _error = asyncio.run(cro.run_single_review(
            "fakebwrap:m1", plan_path, tmp_path / "out", served_models=served,
        ))
        assert success is True
        assert served == {"fakebwrap:m1": {"sandbox": "fallback-landlock"}}

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

        plan_path = self._fallback_lane(tmp_path, monkeypatch, dispatch)
        success, issues, error = asyncio.run(cro.run_single_review(
            "fakebwrap:m1", plan_path, tmp_path / "out",
        ))
        assert success is False and issues is None
        assert error == (
"[SANDBOX_BYPASSED] fakebwrap: the working tree changed during this "
            "unsandboxed slot's window (src.py) while 0 other boundary-less "
            "slots in this fan-out, or any other process writing to this "
            "repository, could also have written it; the check cannot tell "
            "who made the change, so this slot fails closed and its output is "
            "discarded"
        )
        assert error in capsys.readouterr().out
        assert (tmp_path / "src.py").read_text(encoding="utf-8") == "v2\n"

    def test_a_clean_audited_code_review_slot_keeps_its_findings_beside_a_foreign_change(
        self, tmp_path, monkeypatch, capsys
    ):
        """Plan §4.3, with option B's switch on: the tree changed during the
        window, but the slot's own audited stream shows only reads, so the
        change is reported as not attributed and the review counts. With the
        switch off the same slot fails closed as above."""
        import os
        import subprocess

        from utils import readonly_audit

        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x"}
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, env=env)
        (tmp_path / "src.py").write_text("v1\n", encoding="utf-8")
        subprocess.run(["git", "add", "src.py"], cwd=tmp_path, check=True, env=env)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True, env=env)
        clean = {"clean": True, "reason": "", "actions": 4}

        def dispatch(**kwargs):
            (tmp_path / "src.py").write_text("edited by another session\n",
                                             encoding="utf-8")
            return {
                "success": True, "data": [], "source": "stdout_fallback",
                "output_file": str(tmp_path / "out" / "o.json"),
                "details": {"sandbox": "bypassed", "readonly_audit": clean},
            }

        plan_path = self._fallback_lane(tmp_path, monkeypatch, dispatch)
        monkeypatch.setattr(readonly_audit, "STREAM_AUDIT_EXEMPTION_ENABLED", False)
        success, _issues, error = asyncio.run(cro.run_single_review(
            "fakebwrap:m1", plan_path, tmp_path / "out",
        ))
        assert success is False and "fails closed" in error

        monkeypatch.setattr(readonly_audit, "STREAM_AUDIT_EXEMPTION_ENABLED", True)
        (tmp_path / "src.py").write_text("v1\n", encoding="utf-8")
        capsys.readouterr()
        success, _issues, error = asyncio.run(cro.run_single_review(
            "fakebwrap:m1", plan_path, tmp_path / "out",
        ))
        assert success is True, error
        out = capsys.readouterr().out
        assert "not attributed" in out and "(src.py)" in out

    def test_off_refuses_the_code_review_slot_exactly_as_before(
        self, tmp_path, monkeypatch, capsys
    ):
        """The operator's `off`: the refusal above, from an adapter that does
        declare the hook — no flag, no marker, nothing dispatched."""
        def no_dispatch(**kwargs):
            raise AssertionError("a slot was dispatched past a refused probe")

        plan_path = self._fallback_lane(
            tmp_path, monkeypatch, no_dispatch, adapter=self._FallbackOff,
        )
        success, issues, error = asyncio.run(cro.run_single_review(
            "fakebwrap:m1", plan_path, tmp_path / "out",
        ))
        assert success is False and issues is None
        assert error.startswith("[SANDBOX_UNAVAILABLE]")
        assert "[SANDBOX_BYPASSED]" not in capsys.readouterr().out


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
        assert (
            "1 valid, 0 invalid, 1 needs human review, 0 validation failed "
            "(of 2 groups)"
        ) in report

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
                          side_effect=lambda files, **_: _TrackingContextManager(
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


#: The first sentence of the instruction T7.4 appends to `extra_instructions`.
DROPPED_LEAD = (
    "These files were dropped from the reviewed-code block only for its budget."
)


class TestTheValidatorIsToldToOpenTheFilesItLost:
    """T7.4/T7.5 (review-run-scope-salvage-and-failure-visibility §7.2).

    Over its budget the reviewed-code block drops whole files, and the only
    trace used to be a passive "(… not shown: <path>…)" note appended *last* to
    the block, the first text `build_context_source`'s tail cut removes. The
    audited run reported "windows for 3 of 12 cited files not shown" and no
    validator was told to go and read them, though review-code validators run
    `agent-readonly` and can. The instruction now travels in
    `extra_instructions`, which no context budget cuts, and it names only the
    dropped paths that resolve inside the project: the paths are model output.
    """

    WIDTH = 60

    @classmethod
    def _write(cls, path, pad=0):
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"line {n} " + "x" * cls.WIDTH for n in range(1, 201)]
        lines[99] += "y" * pad  # line 100, inside every cited window
        path.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _plan(root):
        root.mkdir(parents=True, exist_ok=True)
        plan_path = root / "plan.md"
        plan_path.write_text("# plan\n\n## accel: 28\n", encoding="utf-8")
        return plan_path

    @staticmethod
    def _groups(paths):
        return [{"suggestions": [{"file": p, "line_range": [100, 101]}]}
                for p in paths]

    @staticmethod
    def _issues(paths):
        # The path only in `file`, so a count of its mentions in a prompt is a
        # count of the records that cite it.
        return [{
            "title": f"finding {i} about {Path(p).name}",
            "desc": f"line 100 of this file does the wrong thing ({i})",
            "importance": "medium", "file": p, "line_range": [100, 101],
            "type": "bug",
        } for i, p in enumerate(paths)]

    def _fill(self, repo, plan_path, first):
        """Pad ``first`` so the block holding its window alone ends 3 chars
        short of `VALIDATION_CONTEXT_LIMIT`: once it is kept, every later
        section is dropped however small, and the block's own "not shown" note
        after it is what a tail cut removes."""
        from utils.validation import VALIDATION_CONTEXT_LIMIT

        self._write(repo / first)
        alone, _ = cro.build_validation_secondary_context(
            self._groups([first]), plan_path, project_root=str(repo)
        )
        self._write(repo / first, pad=VALIDATION_CONTEXT_LIMIT - 3 - len(alone))
        alone, _ = cro.build_validation_secondary_context(
            self._groups([first]), plan_path, project_root=str(repo)
        )
        assert len(alone) == VALIDATION_CONTEXT_LIMIT - 3

    @staticmethod
    def _fake_state(paths):
        fake_state = MagicMock()
        fake_state.get = MagicMock(
            side_effect=lambda k, default=None: {
                "tracked_files": [{"path": p} for p in paths],
                "head_before_implement": "abc123",
                "head_at_start": "abc123",
                "pre_existing_changes": [],
            }.get(k, default)
        )
        fake_state.mark_phase_completed = MagicMock()
        fake_state.save = MagicMock()
        fake_state.is_phase_completed = MagicMock(return_value=False)
        fake_state.state = {"phases_completed": {}}
        return fake_state

    def _fanout(self, plan_path, root, cited):
        """`main`'s validation site, stopped at the host hand-off."""
        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        args = _make_args(plan_path, skip_validation=False)
        with patch.object(cro, "parse_args", return_value=args), \
             patch.object(cro, "get_or_create_state",
                          return_value=self._fake_state(cited)), \
             patch.object(cro, "resolve_models",
                          return_value=["cursor-agent:auto"]), \
             patch.object(cro, "is_model_valid", return_value=True), \
             patch.object(cro, "validate_git_ref",
                          side_effect=lambda r: r or "HEAD~1"), \
             patch.object(cro, "get_project_root", return_value=str(root)), \
             patch.object(cro, "run_all_reviews", new=AsyncMock(
                 return_value={"cursor-agent:auto": (
                     True, self._issues(cited), None)})):
            asyncio.run(cro.main())
        return phase_dir

    def _reaggregate(self, plan_path, root, cited, phase_dir):
        """`reaggregate_from_existing_files`' validation site."""
        phase_dir.mkdir(parents=True)
        (phase_dir / "cursor-agent_auto.json").write_text(
            json.dumps(self._issues(cited)), encoding="utf-8"
        )
        out_dir = phase_dir.parent / "out"
        out_dir.mkdir()
        args = _make_args(plan_path, reaggregate=True, skip_validation=False)
        with patch.object(cro, "get_or_create_state",
                          return_value=self._fake_state(cited)), \
             patch.object(cro, "get_phase_dir", return_value=phase_dir), \
             patch.object(cro, "validate_git_ref",
                          side_effect=lambda r: r or "HEAD~1"), \
             patch.object(cro, "get_project_root", return_value=str(root)), \
             patch.object(cro, "build_unsanitize_map", return_value={}):
            asyncio.run(cro.reaggregate_from_existing_files(
                "plan", out_dir, plan_path, args
            ))
        return phase_dir

    @staticmethod
    def _prompts(phase_dir):
        tasks = json.loads(
            (phase_dir / "validation_tasks.json").read_text(encoding="utf-8")
        )
        assert tasks["batches"]
        return [batch["prompt"] for batch in tasks["batches"]]

    @staticmethod
    def _instruction(prompt):
        lines = [ln for ln in prompt.splitlines() if ln.startswith(DROPPED_LEAD)]
        assert len(lines) == 1, lines
        return lines[0]

    # -- T7.4 ------------------------------------------------------------

    def test_a_drop_names_the_files_to_open_outside_the_block(self, tmp_path):
        from utils.subagent_prompts import CODE_REVIEW_VALIDATION_GUIDANCE

        plan_path = self._plan(tmp_path)
        names = [f"m{i:02d}.py" for i in range(20)]
        for name in names:
            self._write(tmp_path / name)
        cut_report = {}

        secondary, _ = cro.build_validation_secondary_context(
            self._groups(names), plan_path, project_root=str(tmp_path),
            cut_report=cut_report,
        )

        dropped = cut_report["dropped_files"]
        assert dropped, "the fixture is meant to overflow the code budget"
        expected = [str((tmp_path / name).resolve()) for name in dropped]
        assert cut_report["open_paths"] == expected
        extra = cro.validation_extra_instructions(cut_report)
        # After the guidance, never instead of it.
        assert extra.startswith(CODE_REVIEW_VALIDATION_GUIDANCE)
        assert DROPPED_LEAD in extra
        assert "Open each at the lines the findings cite before judging them" in extra
        for path in expected:
            assert f"`{path}`" in extra
            # Absolute, and only outside the block: the block keeps its own
            # relative note, which is what a tail cut removes first.
            assert path not in secondary
        assert DROPPED_LEAD not in secondary

    def test_no_drop_adds_no_instruction(self, tmp_path):
        from utils.subagent_prompts import CODE_REVIEW_VALIDATION_GUIDANCE

        plan_path = self._plan(tmp_path)
        for name in ("a.py", "b.py"):
            self._write(tmp_path / name)
        cut_report = {}

        cro.build_validation_secondary_context(
            self._groups(["a.py", "b.py"]), plan_path,
            project_root=str(tmp_path), cut_report=cut_report,
        )

        assert cut_report["dropped_files"] == []
        assert cut_report["open_paths"] == []
        assert cro.validation_extra_instructions(cut_report) == (
            CODE_REVIEW_VALIDATION_GUIDANCE
        )

    def test_with_no_project_root_the_cwd_is_the_boundary(
        self, tmp_path, monkeypatch
    ):
        """The same root rule `_reviewed_code_sections` reads under."""
        plan_path = self._plan(tmp_path)
        names = [f"m{i:02d}.py" for i in range(20)]
        for name in names:
            self._write(tmp_path / name)
        monkeypatch.chdir(tmp_path)
        cut_report = {}

        cro.build_validation_secondary_context(
            self._groups(names), plan_path, project_root=None,
            cut_report=cut_report,
        )

        assert cut_report["dropped_files"]
        assert cut_report["open_paths"] == [
            str((tmp_path / name).resolve()) for name in cut_report["dropped_files"]
        ]

    def test_an_escaping_path_among_the_dropped_is_named_nowhere(self, tmp_path):
        """An absolute path, a `../` escape and an in-tree symlink out of the
        tree, each forced among the dropped: none reaches the instruction or
        the block, in its cited or its resolved form. An `agent-readonly`
        prompt naming one would send the validator to it, and codex reads
        outside its cwd."""
        repo = tmp_path / "repo"
        plan_path = self._plan(repo)
        self._fill(repo, plan_path, "m00.py")
        names = ["m00.py", "m01.py", "m02.py"]
        for name in names[1:]:
            self._write(repo / name)
        outside = tmp_path / "outside.py"
        self._write(outside)
        secret = tmp_path / "secret_target.py"
        self._write(secret)
        (repo / "zz_link.py").symlink_to(secret)
        escaping = ["/etc/passwd", "../outside.py", "zz_link.py"]

        # Sorted by path, `/…` and `../…` always come first, and the first
        # section is always kept; move the escaping ones last, behind the
        # filled `m00.py`, so a budget drop reaches them, which is the case the
        # containment exists for.
        real_sections = cro._reviewed_code_sections

        def escaping_last(*args, **kwargs):
            return sorted(real_sections(*args, **kwargs),
                          key=lambda section: section[0] in escaping)

        cut_report = {}
        with patch.object(cro, "_reviewed_code_sections", escaping_last):
            secondary, _ = cro.build_validation_secondary_context(
                self._groups(names + escaping), plan_path,
                project_root=str(repo), cut_report=cut_report,
            )

        assert set(escaping) <= set(cut_report["dropped_files"])
        extra = cro.validation_extra_instructions(cut_report)
        named = escaping + [str(outside), str(outside.resolve()),
                            str(secret), str(secret.resolve())]
        for text in (secondary, extra):
            for path in named:
                assert path not in text, (path, text[-400:])
        # The contained drops are still named, absolute, in the instruction,
        # and by their cited spelling in the block's own note.
        assert cut_report["open_paths"] == [
            str((repo / name).resolve()) for name in names[1:]
        ]
        for path in cut_report["open_paths"]:
            assert f"`{path}`" in extra
        assert ("the windows for these files are not shown: m01.py, m02.py.)"
                in secondary)

    def test_both_validation_sites_materialize_the_same_instruction(
        self, tmp_path
    ):
        plan_path = self._plan(tmp_path)
        names = [f"src/m{i:02d}.py" for i in range(20)]
        for name in names:
            self._write(tmp_path / name)

        fanout_dir = self._fanout(plan_path, tmp_path, names)
        reaggregate_dir = self._reaggregate(
            plan_path, tmp_path, names, tmp_path / "reagg" / "code-review"
        )

        fanout = {self._instruction(p) for p in self._prompts(fanout_dir)}
        reaggregate = {self._instruction(p) for p in self._prompts(reaggregate_dir)}
        assert len(fanout) == 1
        assert fanout == reaggregate
        (instruction,) = fanout
        assert f"`{(tmp_path / 'src').resolve()}/" in instruction

    # -- T7.5 ------------------------------------------------------------

    def test_a_block_filled_to_the_limit_still_carries_every_path_to_open(
        self, tmp_path
    ):
        """The reviewed-code block ends a few characters under
        `VALIDATION_CONTEXT_LIMIT`, so the block's own "not shown" note is the
        text the tail cut removes, and many long paths are dropped, one of them
        an in-tree symlink out of the tree. Every materialized
        `validation_batch_N.prompt.txt` still names each contained dropped
        file, absolute, and never the escaping one."""
        repo = tmp_path / "repo"
        plan_path = self._plan(repo)
        long_dir = "pkg/" + "/".join(
            ["a_rather_long_directory_name_for_the_budget_test"] * 3
        )
        longs = [f"{long_dir}/module_with_a_long_name_{i:02d}.py" for i in range(12)]
        for path in longs:
            self._write(repo / path)
        secret = tmp_path / "secret_outside.py"
        self._write(secret)
        link = f"{long_dir}/zz_escape_link.py"
        (repo / link).symlink_to(secret)

        # The first file (by path) fills the block: every later one is dropped.
        big = "a_big.py"
        self._fill(repo, plan_path, big)

        phase_dir = self._fanout(plan_path, repo, [big] + longs + [link])

        sidecars = sorted(phase_dir.glob("validation_batch_*.prompt.txt"))
        assert sidecars
        contained = [str((repo / path).resolve()) for path in longs]
        target = str(secret.resolve())
        without_the_link = 0
        for sidecar in sidecars:
            prompt = sidecar.read_text(encoding="utf-8")
            block = prompt.split(
                f"## Context ({cro.REVIEWED_CODE_CONTEXT_LABEL})"
            )[1].split("## Suggestion groups to validate")[0]
            # The precondition: the block really was filled, and its own note
            # (the only in-block list of the dropped files) was cut away.
            assert "[... truncated ...]" in block
            assert not any(path in block for path in longs + [link])

            instruction = self._instruction(prompt)
            for path in contained:
                assert f"`{path}`" in instruction
            assert target not in prompt
            assert link not in instruction
            # The link's only mentions are its own finding's `file` field, the
            # record the validator is judging.
            assert prompt.count(link) == prompt.count(f'"file": "{link}"')
            without_the_link += link not in prompt
        if len(sidecars) > 1:
            assert without_the_link >= 1


# ---------------------------------------------------------------------------
# Every reviewer is given the resolved scope (review-scope plan §2 D2.1, T2.1)
# and `--files`/`--files-from` set it exactly (decision 11.1, T2.6)
# ---------------------------------------------------------------------------


class _NoProbeAdapter:
    """An adapter with no sandbox probe: the slot dispatches straight away."""

    internal_retries = None

    def sandbox_probe_command(self, profile):
        return None

    def sandbox_fallback_rungs(self, profile):
        return ()


class _Blank(dict):
    def __missing__(self, key):
        return ""


def _rendered(dispatch_kwargs) -> str:
    """The review prompt as the template renders it from `prompt_context`.

    Keys `invoke_with_file_output` adds itself (the output path and friends)
    render blank: the file list is what these tests read.
    """
    import string

    return string.Formatter().vformat(
        dispatch_kwargs["prompt_template"], (), _Blank(dispatch_kwargs["prompt_context"])
    )


def _run_fanout_capturing_prompts(tmp_path, monkeypatch, args_overrides, *, diff=None,
                                  setup_state=None, stdin=None):
    """Run `main()` over a real plan-local `state.json`; return the dispatches.

    The real `run_all_reviews` → `run_single_review` path runs; only the
    provider call (`invoke_with_file_output`) is replaced, so what it receives
    is exactly what a reviewer would have been sent.
    """
    from utils import review_orchestrator_base as rob
    from utils.state_manager import get_or_create_state

    plan_path = tmp_path / "plan.md"
    if not plan_path.exists():
        plan_path.write_text("# Plan\n", encoding="utf-8")
    if setup_state is not None:
        state = get_or_create_state(plan_path)
        setup_state(state)
        state.save()

    seen = []

    def dispatch(**kwargs):
        seen.append(kwargs)
        return {
            "success": True, "data": [], "source": "file",
            "output_file": str(tmp_path / "o.json"), "details": {},
        }

    monkeypatch.setattr(rob, "get_provider", lambda _name: _NoProbeAdapter())
    monkeypatch.setattr("utils.subagent_executor._SANDBOX_PROBE_CACHE", {})
    monkeypatch.setattr(cro, "get_project_root", lambda _p: str(tmp_path))
    monkeypatch.setattr(cro, "invoke_with_file_output", dispatch)
    monkeypatch.setattr(cro, "PROVIDER_STAGGER_DELAY", 0.0)
    if stdin is not None:
        import io

        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))

    args = _make_args(plan_path, **{"files": [], "files_from": None, **args_overrides})
    with patch.object(cro, "parse_args", return_value=args), \
         patch.object(cro, "resolve_models", return_value=["cursor-agent:auto"]), \
         patch.object(cro, "is_model_valid", return_value=True), \
         patch.object(cro, "validate_git_ref", side_effect=lambda r: r or "HEAD~1"), \
         patch.object(cro, "get_files_changed_since_ref",
                      side_effect=(lambda _ref: list(diff)) if diff is not None
                      else AssertionError("the git-diff fallback was consulted")), \
         patch.object(cro, "list_worktree_untracked_files", return_value=[]), \
         patch.object(cro, "intent_to_add_untracked",
                      side_effect=lambda files, **_: contextlib.nullcontext()), \
         patch.object(cro, "generate_review_report", return_value="# report\n"):
        asyncio.run(cro.main())
    return plan_path, seen


def _listed(prompt: str) -> list:
    """The paths the prompt's file list names, in order."""
    return [
        line.split("`")[1] for line in prompt.splitlines()
        if line.startswith("- `") and line.count("`") >= 2
    ]


def _state_json(plan_path):
    from utils.state_manager import get_or_create_state

    return json.loads(get_or_create_state(plan_path).state_file.read_text(encoding="utf-8"))


class TestReviewersAreGivenTheResolvedScope:
    def test_a_fallback_runs_prompt_lists_the_diffs_paths_source_first(
        self, tmp_path, monkeypatch, capsys
    ):
        """The bug: main resolved the fallback scope, then handed every reviewer
        the EMPTY `tracked_files`, so each prompt carried the placeholder."""
        plan_path, seen = _run_fanout_capturing_prompts(
            tmp_path, monkeypatch, {"base_ref": "abc123"},
            diff=["tests/test_a.py", "utils/z.py"],
        )

        assert len(seen) == 1
        prompt = _rendered(seen[0])
        assert _listed(prompt)[:2] == ["utils/z.py", "tests/test_a.py"]
        assert "No tracked files - using git diff" not in prompt
        assert "(from task:" not in prompt
        scope_label = seen[0]["prompt_context"]["scope_source"]
        assert "`git diff --name-only abc123`" in scope_label
        assert "[SCOPE_FALLBACK]" in capsys.readouterr().out

    def test_a_tracked_scope_keeps_its_task_attribution(self, tmp_path, monkeypatch):
        def setup(state):
            state.set("tracked_files", [
                {"path": "tests/test_b.py", "action": "modified", "task_id": "T002"},
                {"path": "src/b.py", "action": "modified", "task_id": "T001"},
            ])

        _plan, seen = _run_fanout_capturing_prompts(
            tmp_path, monkeypatch, {"base_ref": "abc123"}, setup_state=setup,
        )

        prompt = _rendered(seen[0])
        assert "- `src/b.py` (from task: T001)" in prompt
        assert "- `tests/test_b.py` (from task: T002)" in prompt
        assert prompt.index("src/b.py") < prompt.index("tests/test_b.py")
        assert seen[0]["prompt_context"]["scope_source"] == (
            "recorded by the implement phase (tracked_files)"
        )

    def test_the_shipped_template_formats_with_the_context_this_lane_builds(
        self, tmp_path, monkeypatch
    ):
        """`scope_source` is an extra key until the template binds it: `format`
        ignores it, and no placeholder of the template goes missing."""
        _plan, seen = _run_fanout_capturing_prompts(
            tmp_path, monkeypatch, {"base_ref": "abc123"}, diff=["src/a.py"],
        )
        context = dict(seen[0]["prompt_context"])
        context.update(output_json_path="/o.json", output_location="-",
                       working_limits="-")
        seen[0]["prompt_template"].format(**context)


class TestFilesFlagSetsTheExactScope:
    def _tracked(self, state):
        state.set("tracked_files", [
            {"path": "src/other.py", "action": "modified", "task_id": "T001"},
        ])

    def test_files_over_tracked_files_reviews_exactly_those_paths(
        self, tmp_path, monkeypatch, capsys
    ):
        plan_path, seen = _run_fanout_capturing_prompts(
            tmp_path, monkeypatch, {"files": ["b.py", "a.py"]},
            setup_state=self._tracked,
        )

        prompt = _rendered(seen[0])
        assert _listed(prompt) == ["a.py", "b.py"]
        assert seen[0]["prompt_context"]["scope_source"].startswith("given with --files")
        # Recorded at fan-out start, for a later --reaggregate.
        assert _state_json(plan_path)["review_scope_files"] == ["b.py", "a.py"]
        from utils.state_manager import get_or_create_state

        scope = cro.resolve_review_scope(get_or_create_state(plan_path), "abc123")
        assert scope.source == "files_flag"
        assert sorted(scope.paths) == ["a.py", "b.py"]
        assert "Recorded the --files scope (2 files)" in capsys.readouterr().out

    def test_files_from_with_blank_lines_gives_the_same_scope(
        self, tmp_path, monkeypatch
    ):
        listing = tmp_path / "scope.txt"
        listing.write_text("b.py\n\n   \na.py\n", encoding="utf-8")

        plan_path, seen = _run_fanout_capturing_prompts(
            tmp_path, monkeypatch, {"files_from": str(listing)},
            setup_state=self._tracked,
        )

        assert _listed(_rendered(seen[0])) == ["a.py", "b.py"]
        assert _state_json(plan_path)["review_scope_files"] == ["b.py", "a.py"]

    def test_files_from_stdin(self, tmp_path, monkeypatch):
        _plan, seen = _run_fanout_capturing_prompts(
            tmp_path, monkeypatch, {"files_from": "-"}, stdin="lib/x.py\n",
            setup_state=self._tracked,
        )
        assert _listed(_rendered(seen[0])) == ["lib/x.py"]

    def test_an_absolute_path_inside_the_repository_is_made_relative(
        self, tmp_path, monkeypatch
    ):
        _plan, seen = _run_fanout_capturing_prompts(
            tmp_path, monkeypatch, {"files": [str(tmp_path / "pkg" / "m.py")]},
        )
        assert _listed(_rendered(seen[0])) == ["pkg/m.py"]

    @pytest.mark.parametrize("outside", ["/definitely/elsewhere/x.py", "../escape.py"])
    def test_a_path_outside_the_repository_exits_1_naming_it(
        self, tmp_path, monkeypatch, capsys, outside
    ):
        with pytest.raises(SystemExit) as exc:
            _run_fanout_capturing_prompts(
                tmp_path, monkeypatch, {"files": ["a.py", outside]},
            )
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "outside the repository" in out
        assert outside in out

    def test_files_from_stdin_under_detach_is_refused(self, tmp_path, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_fanout_capturing_prompts(
                tmp_path, monkeypatch, {"files_from": "-", "detach": True},
                stdin="a.py\n",
            )
        assert exc.value.code == 1
        assert "--detach worker does not inherit" in capsys.readouterr().out

    def test_a_later_fanout_without_files_removes_the_record(
        self, tmp_path, monkeypatch, capsys
    ):
        plan_path, _seen = _run_fanout_capturing_prompts(
            tmp_path, monkeypatch, {"files": ["a.py"]}, setup_state=self._tracked,
        )
        assert _state_json(plan_path)["review_scope_files"] == ["a.py"]

        _plan, seen = _run_fanout_capturing_prompts(
            tmp_path, monkeypatch, {"force": True, "rerun_all": True},
        )

        assert "review_scope_files" not in _state_json(plan_path)
        # ...so this run, and every reaggregate after it, is back on tracked_files.
        assert _listed(_rendered(seen[0])) == ["src/other.py"]
        assert "Cleared review_scope_files from state.json" in capsys.readouterr().out

    @pytest.mark.parametrize("flags, named", [
        ({"files": ["x.py"]}, "--files"),
        ({"files_from": "-"}, "--files-from"),
        ({"files": ["x.py"], "files_from": "list.txt"}, "--files or --files-from"),
        ({}, None),
    ])
    def test_a_reaggregate_says_it_does_not_re_resolve_the_flags(
        self, tmp_path, monkeypatch, capsys, flags, named
    ):
        """Every printed reaggregate command replays the fan-out's own flags, so
        they are accepted — and said to be unused, never dropped silently."""
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# Plan\n", encoding="utf-8")
        cro.get_phase_dir(plan_path, "code-review")
        reaggregated = []

        async def fake_reaggregate(*args, **kwargs):
            reaggregated.append(args)

        class _NoStdin:
            def read(self, *_a):
                raise AssertionError("--reaggregate read --files-from - from stdin")

        monkeypatch.setattr(cro, "reaggregate_from_existing_files", fake_reaggregate)
        monkeypatch.setattr(sys, "stdin", _NoStdin())
        args = _make_args(plan_path, reaggregate=True,
                          **{"files": [], "files_from": None, **flags})
        with patch.object(cro, "parse_args", return_value=args):
            asyncio.run(cro.main())

        out = capsys.readouterr().out
        assert len(reaggregated) == 1
        lines = [text for text in out.splitlines()
                 if text.startswith("[SCOPE_FLAGS_IGNORED]")]
        if named is None:
            assert lines == []
            return
        assert len(lines) == 1
        assert f"does not re-resolve {named};" in lines[0]
        assert "review_scope_files" in lines[0]

    def test_the_parser_accepts_both_flags(self):
        args = cro.parse_args([
            "--plan-file", "p.md", "--files", "a.py", "--files", "b.py",
            "--files-from", "list.txt",
        ])
        assert args.files == ["a.py", "b.py"]
        assert args.files_from == "list.txt"
        plain = cro.parse_args(["--plan-file", "p.md"])
        assert plain.files == [] and plain.files_from is None


# ---------------------------------------------------------------------------
# The scope manifest and per-file diffs every reviewer is pointed at
# (review-scope plan §2: D2.2/T2.2, D2.3/T2.3, D3.2/T3.2)
# ---------------------------------------------------------------------------
#
# These drive the real `main()` → `run_all_reviews` → `run_single_review` path
# in a real temp repository, with only the provider call replaced: the diff set
# is written by the real `write_review_diff`, and what the stub receives is
# exactly what a reviewer would have been sent.

import os as _os
import re
import subprocess as _subprocess
from types import SimpleNamespace as _NS

from utils import git_utils as _git_utils
from utils import llm_client as _llm_client
from utils.git_utils import (
    REVIEW_SCOPE_FILENAME as _MANIFEST,
    read_review_scope as _read_review_scope,
)

_HERMETIC_GIT = {
    "GIT_CONFIG_GLOBAL": _os.devnull,
    "GIT_CONFIG_SYSTEM": _os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
}


def _git(cwd, *args):
    return _subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def _put(root, rel, text):
    path = Path(root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def scope_repo(tmp_path, monkeypatch):
    """A repository whose implementation changed two files since `base`."""
    for key, value in _HERMETIC_GIT.items():
        monkeypatch.setenv(key, value)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "root")
    _put(repo, "src/app.py", "def a():\n    return 1\n")
    _put(repo, "tests/test_app.py", "def test_a():\n    assert True\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD").strip()
    _put(repo, "src/app.py", "def a():\n    return 2\n")
    _put(repo, "tests/test_app.py", "def test_a():\n    assert a() == 2\n")
    plan = repo / "plans" / "feature.md"
    _put(repo, "plans/feature.md", "# Plan\n\nMake a() return 2.\n")
    monkeypatch.chdir(repo)
    return _NS(repo=repo, plan=plan, base=base)


def _record_scope(plan, base, paths=("src/app.py", "tests/test_app.py")):
    from utils.state_manager import get_or_create_state

    state = get_or_create_state(plan)
    state.set("tracked_files", [
        {"path": path, "action": "modified", "task_id": "T001"} for path in paths
    ])
    state.set("head_before_implement", base)
    state.save()


class _Capability:
    """An adapter as `render_diff_access`/`render_output_location` read it."""

    supports_file_output = False

    def __init__(self, read_scope, shell):
        self.generate_read_scope = read_scope
        self.runs_shell_in_generate = shell


def _phase(plan):
    return cro.get_phase_dir(plan, "code-review")


def _scope_fanout(plan, monkeypatch, models, *, respond=None, capabilities=None,
                  real_report=False, **arg_overrides):
    """Run `main()`; return ``(dispatches, exit code)``.

    Each dispatch records the kwargs `invoke_with_file_output` received plus
    what was on disk at that moment (the manifest, the ledger), which is what
    "written before dispatch" means.
    """
    from utils import review_orchestrator_base as rob

    seen = []

    def dispatch(**kwargs):
        phase = _phase(plan)
        manifest = phase / _MANIFEST
        seen.append({
            "kwargs": kwargs,
            "manifest": manifest.read_text(encoding="utf-8") if manifest.is_file() else None,
            "diff_files": sorted(p.name for p in (phase / "diff").glob("*.diff")),
            "ledger": cro.read_scope_ledger(phase),
        })
        if respond is not None:
            return respond(kwargs)
        return {"success": True, "data": [], "source": "file",
                "output_file": "", "details": {}}

    monkeypatch.setattr(rob, "get_provider", lambda _name: _NoProbeAdapter())
    monkeypatch.setattr("utils.subagent_executor._SANDBOX_PROBE_CACHE", {})
    monkeypatch.setattr(cro, "invoke_with_file_output", dispatch)
    monkeypatch.setattr(cro, "PROVIDER_STAGGER_DELAY", 0.0)
    if capabilities is not None:
        real = _llm_client.get_provider
        monkeypatch.setattr(
            _llm_client, "get_provider",
            lambda name: capabilities.get(name) or real(name),
        )
    args = _make_args(plan, models=list(models), files=[], files_from=None,
                      **arg_overrides)
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(cro, "parse_args", return_value=args))
        stack.enter_context(patch.object(cro, "resolve_models", return_value=list(models)))
        stack.enter_context(patch.object(cro, "is_model_valid", return_value=True))
        if not real_report:
            stack.enter_context(
                patch.object(cro, "generate_review_report", return_value="# report\n"))
        try:
            code = asyncio.run(cro.main())
        except SystemExit as exc:
            code = exc.code
    return seen, code


def _full_prompt(dispatch, plan) -> str:
    """The prompt exactly as `invoke_with_file_output` would format it."""
    kwargs = dispatch["kwargs"]
    context = dict(kwargs["prompt_context"])
    context["output_json_path"] = "<unused>"
    context["output_location"] = _llm_client.render_output_location(
        "<unused>", kwargs["model_spec"], anchor=str(plan)
    )
    context["working_limits"] = _llm_client.render_working_limits(kwargs["timeout"])
    return kwargs["prompt_template"].format(**context)


def _stem(spec):
    return cro.sanitize_model_name(spec)


class TestTheDiffSetIsWrittenBeforeDispatch:
    def test_a_reviewer_finds_this_runs_set_even_over_an_earlier_one(
        self, scope_repo, monkeypatch
    ):
        _record_scope(scope_repo.plan, scope_repo.base)
        phase = _phase(scope_repo.plan)
        (phase / "diff").mkdir(parents=True)
        (phase / "diff" / "9-old.diff").write_text("old\n", encoding="utf-8")
        (phase / _MANIFEST).write_text(
            "# Review scope\n\n- Scope id: `0000000000aa`\n\nDiff coverage: complete\n"
            "`diff/9-old.diff`\n",
            encoding="utf-8",
        )

        seen, _ = _scope_fanout(scope_repo.plan, monkeypatch, ["cursor-agent:auto"])

        assert len(seen) == 1
        header = _read_review_scope_text(seen[0]["manifest"])
        assert header != "0000000000aa"
        assert seen[0]["ledger"] == {_stem("cursor-agent:auto"): header}
        assert seen[0]["diff_files"] == ["1-src-app.py.diff", "2-tests-test_app.py.diff"]
        assert "9-old.diff" not in seen[0]["diff_files"]

    def test_the_prompt_names_both_paths_and_calls_the_list_a_head(
        self, scope_repo, monkeypatch
    ):
        _record_scope(scope_repo.plan, scope_repo.base)

        seen, _ = _scope_fanout(scope_repo.plan, monkeypatch, ["cursor-agent:auto"])

        prompt = _full_prompt(seen[0], scope_repo.plan)
        phase = _phase(scope_repo.plan)
        assert str(phase / _MANIFEST) in prompt
        assert str(phase / "diff") in prompt
        assert "This list is only the head of the review scope." in prompt
        assert f"The full scope is the scope manifest, {phase / _MANIFEST}" in prompt
        assert "Focus your review on ONLY these files" not in prompt
        assert "Where this list came from: recorded by the implement phase" in prompt

    def test_a_cut_list_names_the_manifest_and_a_short_one_does_not(
        self, scope_repo, monkeypatch
    ):
        many = [f"src/pkg_{i:03d}/" + "m" * 40 + ".py" for i in range(150)]
        _record_scope(scope_repo.plan, scope_repo.base, paths=many)

        seen, _ = _scope_fanout(scope_repo.plan, monkeypatch, ["cursor-agent:auto"])

        listed = seen[0]["kwargs"]["prompt_context"]["tracked_files_list"]
        manifest = _phase(scope_repo.plan) / _MANIFEST
        assert "more files not shown" in listed
        assert (f"Every path not shown is still in scope; the full list is "
                f"{manifest}.") in listed

        _record_scope(scope_repo.plan, scope_repo.base)
        seen, _ = _scope_fanout(scope_repo.plan, monkeypatch, ["cursor-agent:auto"],
                                force=True, rerun_all=True)
        assert "Every path not shown" not in (
            seen[0]["kwargs"]["prompt_context"]["tracked_files_list"]
        )

    @pytest.mark.parametrize("spec, capability", [
        ("cursor-agent:auto", None),                     # paths
        ("qwen-code:qwen3-coder", None),                  # inline
        ("opencode:m", _Capability("any", None)),         # paths, unmeasured
    ])
    def test_the_shipped_template_formats_with_this_lanes_context(
        self, scope_repo, monkeypatch, spec, capability
    ):
        _record_scope(scope_repo.plan, scope_repo.base)
        capabilities = {spec.split(":")[0]: capability} if capability else None

        seen, _ = _scope_fanout(scope_repo.plan, monkeypatch, [spec],
                                capabilities=capabilities)

        # `str.format` raises KeyError for a placeholder with no key, which is
        # what would fail every slot with PROMPT_FORMAT_ERROR.
        _full_prompt(seen[0], scope_repo.plan)

    def test_the_static_template_names_no_shell_command(self):
        template = cro.load_code_review_prompt()
        for command in ("git diff", "git show", "git log", "cat "):
            assert command not in template
        assert template.startswith(
            "You are reviewing code changes that implement a feature plan."
        )
        for placeholder in ("{scope_source}", "{scope_file}", "{diff_dir}",
                            "{diff_access}", "{working_limits}"):
            assert placeholder in template
        assert template.index("{working_limits}") < template.index("## Output Format")


def _read_review_scope_text(manifest_text):
    """The scope id a manifest's header carries."""
    match = re.search(r"^- Scope id: `([0-9a-f]+)`", manifest_text or "", re.M)
    return match.group(1) if match else None


class TestAResumeReusesOnlyResultsOfThisScope:
    SPECS = ["cursor-agent:auto", "gemini:gemini-2.5-flash"]

    def _first_run(self, scope_repo, monkeypatch):
        _record_scope(scope_repo.plan, scope_repo.base)
        seen, _ = _scope_fanout(scope_repo.plan, monkeypatch, self.SPECS)
        assert len(seen) == 2
        return _read_review_scope_text(seen[0]["manifest"])

    def test_an_unchanged_scope_reuses_every_result_and_touches_nothing(
        self, scope_repo, monkeypatch, capsys
    ):
        first_id = self._first_run(scope_repo, monkeypatch)
        phase = _phase(scope_repo.plan)
        manifest = phase / _MANIFEST
        ledger = phase / cro.REVIEW_SCOPE_LEDGER
        before = (manifest.read_bytes(), manifest.stat().st_mtime_ns, ledger.read_bytes())
        capsys.readouterr()

        seen, _ = _scope_fanout(scope_repo.plan, monkeypatch, self.SPECS, force=True)

        out = capsys.readouterr().out
        assert seen == []
        assert out.count("[SKIP]") == 2
        assert "[SCOPE_CHANGED]" not in out
        assert (manifest.read_bytes(), manifest.stat().st_mtime_ns,
                ledger.read_bytes()) == before
        assert set(cro.read_scope_ledger(phase).values()) == {first_id}

    def test_a_changed_scope_re_runs_and_retires_the_old_results(
        self, scope_repo, monkeypatch, capsys
    ):
        first_id = self._first_run(scope_repo, monkeypatch)
        _put(scope_repo.repo, "src/app.py", "def a():\n    return 3\n")
        capsys.readouterr()

        seen, _ = _scope_fanout(scope_repo.plan, monkeypatch, self.SPECS, force=True)

        out = capsys.readouterr().out
        phase = _phase(scope_repo.plan)
        new_id = _read_review_scope_text((phase / _MANIFEST).read_text(encoding="utf-8"))
        assert new_id != first_id
        assert len(seen) == 2
        for spec in self.SPECS:
            display = spec
            assert (f"[SCOPE_CHANGED] {display}: result reviewed scope {first_id}, "
                    f"this run's is {new_id}; re-running") in out
            assert (phase / f"{_stem(spec)}.json.stale-{first_id}").exists()
            assert (phase / f"{_stem(spec)}.json").exists()
        assert cro.read_scope_ledger(phase) == {_stem(s): new_id for s in self.SPECS}

    def test_a_re_run_that_fails_leaves_no_old_scope_result_behind(
        self, scope_repo, monkeypatch
    ):
        first_id = self._first_run(scope_repo, monkeypatch)
        _put(scope_repo.repo, "src/app.py", "def a():\n    return 3\n")
        failing = self.SPECS[1]

        def respond(kwargs):
            if kwargs["model_spec"] == failing:
                return {"success": False, "error": "quota", "details": {}}
            return {"success": True, "data": [], "source": "file",
                    "output_file": "", "details": {}}

        _scope_fanout(scope_repo.plan, monkeypatch, self.SPECS, respond=respond,
                      force=True)

        phase = _phase(scope_repo.plan)
        assert not (phase / f"{_stem(failing)}.json").exists()
        assert (phase / f"{_stem(failing)}.json.stale-{first_id}").exists()

    def test_a_sandbox_refused_re_run_still_retires_the_old_scope_result(
        self, scope_repo, monkeypatch
    ):
        """A refused slot starts no attempt, but a result from another scope is
        no review of this change either: left in place, the next
        `--reaggregate` would load it and move the model out of
        `models_failed`."""
        from utils import review_orchestrator_base as rob

        first_id = self._first_run(scope_repo, monkeypatch)
        _put(scope_repo.repo, "src/app.py", "def a():\n    return 3\n")
        refused = self.SPECS[1]

        def gate(provider_name, _cwd):
            if provider_name == refused.split(":")[0]:
                return rob.SandboxVerdict(
                    rob.SANDBOX_REFUSE, "[SANDBOX_UNAVAILABLE] simulated"
                )
            return rob.SandboxVerdict(rob.SANDBOX_DISPATCH, None)

        monkeypatch.setattr(cro, "sandbox_refusal", gate)

        seen, _ = _scope_fanout(scope_repo.plan, monkeypatch, self.SPECS, force=True)

        phase = _phase(scope_repo.plan)
        assert [d["kwargs"]["model_spec"] for d in seen] == [self.SPECS[0]]
        assert not (phase / f"{_stem(refused)}.json").exists()
        assert (phase / f"{_stem(refused)}.json.stale-{first_id}").exists()
        # No attempt started, so nothing claims this scope for the refused stem.
        assert cro.read_scope_ledger(phase)[_stem(refused)] == first_id
        status = json.loads((phase / ".status.json").read_text())
        assert refused in status["models_failed"]

    def test_a_result_with_no_ledger_entry_is_unknown_and_re_runs(
        self, scope_repo, monkeypatch, capsys
    ):
        self._first_run(scope_repo, monkeypatch)
        phase = _phase(scope_repo.plan)
        (phase / cro.REVIEW_SCOPE_LEDGER).unlink()
        capsys.readouterr()

        seen, _ = _scope_fanout(scope_repo.plan, monkeypatch, self.SPECS, force=True)

        out = capsys.readouterr().out
        assert len(seen) == 2
        assert "result reviewed scope unknown" in out
        for spec in self.SPECS:
            assert (phase / f"{_stem(spec)}.json.stale-unknown").exists()


class TestTheReportRecordsTheDiffCoverage:
    def _report(self, plan):
        phase = _phase(plan)
        return (
            (phase / "report.md").read_text(encoding="utf-8"),
            json.loads((phase / "report_data.json").read_text(encoding="utf-8")),
        )

    def test_a_complete_set_records_complete(self, scope_repo, monkeypatch):
        _record_scope(scope_repo.plan, scope_repo.base)

        _scope_fanout(scope_repo.plan, monkeypatch, ["cursor-agent:auto"],
                      real_report=True)

        report, data = self._report(scope_repo.plan)
        assert "**Diff coverage:** complete" in report
        assert "[`review-scope.md`](review-scope.md)" in report
        assert data["diffCoverage"]["coverage"] == "complete"
        assert data["diffCoverage"]["omitted"] == []
        header = _read_review_scope(_phase(scope_repo.plan) / _MANIFEST)
        assert data["reviewScope"]["scopeId"] == header.scope_id
        assert data["reviewScope"]["writtenBy"] == "fan-out"
        assert data["reviewScope"]["baseCommit"] == scope_repo.base

    def test_a_partial_set_prints_the_marker_and_names_the_omitted_path(
        self, scope_repo, monkeypatch, capsys
    ):
        _record_scope(scope_repo.plan, scope_repo.base)
        real = _git_utils._git_diff_output

        def flaky(args, cwd):
            if args[0] == "diff" and args[-1] == "tests/test_app.py" and "--numstat" not in args:
                return None, "fatal: simulated"
            return real(args, cwd)

        monkeypatch.setattr(_git_utils, "_git_diff_output", flaky)

        _scope_fanout(scope_repo.plan, monkeypatch, ["cursor-agent:auto"],
                      real_report=True)

        assert "[DIFF_PARTIAL] 1 of 2 source and test files" in capsys.readouterr().out
        report, data = self._report(scope_repo.plan)
        assert "**Diff coverage:** partial" in report
        assert "`tests/test_app.py` (git diff failed)" in report
        assert data["diffCoverage"]["coverage"] == "partial"
        assert data["diffCoverage"]["omitted"] == [
            {"path": "tests/test_app.py", "reason": "git diff failed"}
        ]

    def test_an_unresolvable_base_ref_still_dispatches_every_reviewer(
        self, scope_repo, monkeypatch, capsys
    ):
        _record_scope(scope_repo.plan, scope_repo.base)
        specs = ["cursor-agent:auto", "gemini:gemini-2.5-flash"]
        monkeypatch.setattr(cro, "validate_git_ref", lambda ref: ref)

        seen, code = _scope_fanout(scope_repo.plan, monkeypatch, specs,
                                   real_report=True, base_ref="no-such-ref")

        out = capsys.readouterr().out
        assert "[DIFF_FAILED]" in out
        assert "[REVIEW_UNSUPPORTED]" not in out
        assert sorted(d["kwargs"]["model_spec"] for d in seen) == sorted(specs)
        assert code == 0
        _report, data = self._report(scope_repo.plan)
        assert data["diffCoverage"]["coverage"] == "partial"


class TestAReviewInvokedFromASubdirectory:
    """The scope is repository-root-relative, so its git calls run at the root.

    Run from the caller's cwd, a `review-code` started in a subdirectory had git
    read every root-relative path relative to that subdirectory: each `git diff
    <base> -- <path>` came back empty, the untracked file was never marked
    intent-to-add, and the manifest recorded `no net change` under `complete`
    coverage. Both entry points — the fan-out and a reaggregate that has to
    regenerate the pair — now diff, classify and mark at the repository root.
    """

    _EXPECTED = [
        "1-src-app.py.diff", "2-src-new_module.py.diff", "3-tests-test_app.py.diff",
    ]

    def _subdir_scope(self, scope_repo, monkeypatch):
        _put(scope_repo.repo, "src/new_module.py", "def b():\n    return 3\n")
        _record_scope(scope_repo.plan, scope_repo.base, paths=(
            "src/app.py", "tests/test_app.py", "src/new_module.py",
        ))
        subdir = scope_repo.repo / "src"
        monkeypatch.chdir(subdir)
        return subdir

    def _assert_full_set(self, phase, scope_repo):
        manifest = (phase / _MANIFEST).read_text(encoding="utf-8")
        assert "Diff coverage: complete" in manifest
        assert "no net change" not in manifest
        assert sorted(p.name for p in (phase / "diff").glob("*.diff")) == self._EXPECTED
        new = (phase / "diff" / "2-src-new_module.py.diff").read_text(encoding="utf-8")
        assert "new file mode" in new and "+def b():" in new
        app = (phase / "diff" / "1-src-app.py.diff").read_text(encoding="utf-8")
        assert "+    return 2" in app
        # The intent-to-add entry was made at the root and undone there.
        assert "src/new_module.py" in _git(
            scope_repo.repo, "ls-files", "--others", "--exclude-standard"
        )

    def test_the_fanout_writes_every_diff(self, scope_repo, monkeypatch, capsys):
        self._subdir_scope(scope_repo, monkeypatch)

        seen, code = _scope_fanout(scope_repo.plan, monkeypatch, ["cursor-agent:auto"])

        assert code == 0
        assert "[DIFF_" not in capsys.readouterr().out
        assert seen[0]["diff_files"] == self._EXPECTED
        self._assert_full_set(_phase(scope_repo.plan), scope_repo)

    def test_a_regenerating_reaggregate_writes_every_diff(
        self, scope_repo, monkeypatch, capsys
    ):
        import shutil

        self._subdir_scope(scope_repo, monkeypatch)
        _scope_fanout(scope_repo.plan, monkeypatch, ["cursor-agent:auto"])
        phase = _phase(scope_repo.plan)
        (phase / _MANIFEST).unlink()
        shutil.rmtree(phase / "diff")
        capsys.readouterr()

        args = _make_args(scope_repo.plan, reaggregate=True, files=[], files_from=None)
        with patch.object(cro, "generate_review_report", return_value="# report\n"):
            asyncio.run(cro.reaggregate_from_existing_files(
                "plan", phase.parent, scope_repo.plan, args,
            ))

        out = capsys.readouterr().out
        assert "[REVIEW_SCOPE_REGENERATED]" in out
        assert "[DIFF_" not in out
        self._assert_full_set(phase, scope_repo)


class TestALaneThatCannotOpenTheArtifacts:
    def test_qwen_code_gets_the_change_inline_and_no_path_it_cannot_open(
        self, scope_repo, monkeypatch
    ):
        _record_scope(scope_repo.plan, scope_repo.base)

        seen, _ = _scope_fanout(scope_repo.plan, monkeypatch, ["qwen-code:qwen3-coder"])

        assert len(seen) == 1
        prompt = _full_prompt(seen[0], scope_repo.plan)
        assert "===== BEGIN review-scope.md =====" in prompt
        assert "===== BEGIN plan =====" in prompt
        assert "Make a() return 2." in prompt
        assert "+    return 2" in prompt
        assert "You cannot open files" in prompt
        assert str(scope_repo.repo) not in prompt

    def test_over_the_inline_cap_the_slot_is_not_dispatched(
        self, scope_repo, monkeypatch, capsys
    ):
        big = "".join(f"line_{i:06d} = {i}\n" for i in range(8000))
        _put(scope_repo.repo, "src/big.py", big)
        _record_scope(scope_repo.plan, scope_repo.base,
                      paths=("src/app.py", "src/big.py"))
        # `src/big.py` is new and untracked: main's own intent-to-add block is
        # what makes `git diff` show it.
        seen, _ = _scope_fanout(
            scope_repo.plan, monkeypatch,
            ["qwen-code:qwen3-coder", "cursor-agent:auto"],
        )

        out = capsys.readouterr().out
        assert [d["kwargs"]["model_spec"] for d in seen] == ["cursor-agent:auto"]
        assert ("[REVIEW_UNSUPPORTED] qwen-code:qwen3-coder: read scope none, inline "
                in out)
        assert "exceeds the 98304 B cap" in out
        status = json.loads((_phase(scope_repo.plan) / ".status.json").read_text())
        assert "qwen-code:qwen3-coder" in status["models_failed"]
        assert status["models_failed"]["qwen-code:qwen3-coder"].startswith(
            "review unsupported: read scope none"
        )

    def _block_under_the_cap_in_a_prompt_over_it(self, scope_repo, monkeypatch):
        """An inline block under the 96 KiB cap, in a prompt whose multibyte
        implementation record carries it past the argv ceiling."""
        mid = "".join(f"line_{i:06d} = {i}\n" for i in range(3500))
        _put(scope_repo.repo, "src/mid.py", mid)
        _record_scope(scope_repo.plan, scope_repo.base,
                      paths=("src/app.py", "src/mid.py"))
        # 20 000 chars, the record's own bound, at three bytes each.
        monkeypatch.setattr(cro, "build_implementation_record",
                            lambda _plan: "記" * 20000)

    def test_a_block_under_the_cap_in_a_prompt_over_argv_is_not_dispatched(
        self, scope_repo, monkeypatch, capsys
    ):
        self._block_under_the_cap_in_a_prompt_over_it(scope_repo, monkeypatch)

        seen, _ = _scope_fanout(
            scope_repo.plan, monkeypatch,
            ["qwen-code:qwen3-coder", "cursor-agent:auto"],
        )

        out = capsys.readouterr().out
        assert [d["kwargs"]["model_spec"] for d in seen] == ["cursor-agent:auto"]
        line = next(text for text in out.splitlines()
                    if text.startswith("[REVIEW_UNSUPPORTED] qwen-code:qwen3-coder"))
        inline = int(re.search(r"inline (\d+) B", line).group(1))
        assert inline <= _llm_client.INLINE_DIFF_CAP_BYTES
        assert (f"exceeds the {_llm_client.INLINE_PROMPT_CAP_BYTES} B argv cap"
                in line)
        status = json.loads((_phase(scope_repo.plan) / ".status.json").read_text())
        assert status["models_failed"]["qwen-code:qwen3-coder"].startswith(
            "review unsupported: read scope none"
        )

    def test_a_lane_whose_prompt_travels_on_stdin_is_not_measured(
        self, scope_repo, monkeypatch, capsys
    ):
        self._block_under_the_cap_in_a_prompt_over_it(scope_repo, monkeypatch)
        stdin_lane = _Capability("none", False)
        stdin_lane.prompt_transport = "stdin"

        seen, _ = _scope_fanout(
            scope_repo.plan, monkeypatch, ["qwen-code:qwen3-coder"],
            capabilities={"qwen-code": stdin_lane},
        )

        assert [d["kwargs"]["model_spec"] for d in seen] == ["qwen-code:qwen3-coder"]
        assert "[REVIEW_UNSUPPORTED]" not in capsys.readouterr().out
        assert "===== BEGIN review-scope.md =====" in _full_prompt(seen[0], scope_repo.plan)

    def test_a_path_git_failed_for_makes_the_slot_unsupported(
        self, scope_repo, monkeypatch, capsys
    ):
        _record_scope(scope_repo.plan, scope_repo.base)
        real = _git_utils._git_diff_output

        def flaky(args, cwd):
            if args[0] == "diff" and args[-1] == "src/app.py" and "--numstat" not in args:
                return None, "fatal: simulated"
            return real(args, cwd)

        monkeypatch.setattr(_git_utils, "_git_diff_output", flaky)

        seen, _ = _scope_fanout(
            scope_repo.plan, monkeypatch, ["qwen-code:qwen3-coder", "cursor-agent:auto"],
        )

        assert [d["kwargs"]["model_spec"] for d in seen] == ["cursor-agent:auto"]
        assert "diff coverage is partial" in capsys.readouterr().out
        status = json.loads((_phase(scope_repo.plan) / ".status.json").read_text())
        assert "qwen-code:qwen3-coder" in status["models_failed"]


class TestAPhaseDirOutsideTheRepository:
    def test_cwd_lanes_go_inline_and_any_lanes_get_absolute_paths(
        self, scope_repo, monkeypatch, tmp_path
    ):
        elsewhere = tmp_path / "elsewhere"
        plan = elsewhere / "feature.md"
        _put(elsewhere, "feature.md", "# Plan\n\nMake a() return 2.\n")
        _record_scope(plan, scope_repo.base)
        capabilities = {
            "cursor-agent": _Capability("cwd", False),
            "opencode": _Capability("any", None),
        }

        seen, _ = _scope_fanout(plan, monkeypatch, ["cursor-agent:auto", "opencode:m"],
                                capabilities=capabilities)

        phase = _phase(plan)
        assert (phase / _MANIFEST).is_file()
        assert sorted(p.name for p in (phase / "diff").glob("*.diff")) == [
            "1-src-app.py.diff", "2-tests-test_app.py.diff",
        ]
        by_spec = {d["kwargs"]["model_spec"]: _full_prompt(d, plan) for d in seen}
        inline = by_spec["cursor-agent:auto"]
        assert "===== BEGIN review-scope.md =====" in inline
        assert str(phase) not in inline
        paths = by_spec["opencode:m"]
        assert str(phase / _MANIFEST) in paths
        assert str(phase / "diff") in paths
        assert Path(str(phase / _MANIFEST)).is_absolute()
        assert "===== BEGIN" not in paths


class TestTheTwoRenderingsNeverDisagree:
    """`{working_limits}` and `{diff_access}` in one prompt, per shell state."""

    CAPABILITIES = {
        "pi": _Capability("cwd", True),
        "cursor-agent": _Capability("cwd", False),
        "opencode": _Capability("cwd", None),
    }
    TEST_CLAIM = (
        "Do NOT claim that a test passes or fails unless you ran it during this "
        "review and saw its result."
    )
    CROSS_CHECK = "optional cross-check"
    NO_SHELL = "You cannot run shell commands"

    def test_each_shell_state(self, scope_repo, monkeypatch):
        _record_scope(scope_repo.plan, scope_repo.base)

        seen, _ = _scope_fanout(
            scope_repo.plan, monkeypatch, ["pi:m", "cursor-agent:auto", "opencode:m"],
            capabilities=self.CAPABILITIES,
        )

        block = _llm_client.render_working_limits(1200)
        assert "git" not in block and "`" not in block
        prompts = {
            d["kwargs"]["model_spec"].split(":")[0]: _full_prompt(d, scope_repo.plan)
            for d in seen
        }
        assert set(prompts) == {"pi", "cursor-agent", "opencode"}
        for prompt in prompts.values():
            assert self.TEST_CLAIM in prompt
        shell, no_shell, unmeasured = (
            prompts["pi"], prompts["cursor-agent"], prompts["opencode"],
        )
        assert self.CROSS_CHECK in shell and self.NO_SHELL not in shell
        assert self.NO_SHELL in no_shell
        for grant in ("git diff", "git show", "git log"):
            assert grant not in no_shell
            assert grant not in unmeasured
        assert self.CROSS_CHECK not in no_shell and self.CROSS_CHECK not in unmeasured
        assert self.NO_SHELL not in unmeasured
