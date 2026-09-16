"""Wiring tests for code_review_orchestrator.

These tests focus specifically on verifying that the review pipeline is
wrapped in the `intent_to_add_untracked` context manager so untracked files
are visible to git diff during review execution, validation prep, and
report generation.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import code_review_orchestrator as cro


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

        parent.run_all_reviews = AsyncMock(return_value={})
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

    def test_an_unvalidated_report_says_nothing_about_groups(self, tmp_path):
        """--skip-validation hands no groups over; the report claims none."""
        report = self._render(tmp_path, validated=False)
        assert "**Groups:**" not in report
        assert "Same finding as" not in report
        assert "**Total Issues:** 3" in report
