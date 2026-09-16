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
