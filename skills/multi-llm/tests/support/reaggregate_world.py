"""``--reaggregate`` on prepared phase folders, for review-plan and code-review.

``_phase`` / ``_cro_phase`` lay out a phase folder holding per-model results,
``_reaggregate`` and ``_run_cro_reaggregate`` drive the reaggregation (the latter
with the report generator patched), ``_reported_results`` reads what that mock
was handed and ``_cro_report_md`` names the report it writes. No fixtures.
"""

import argparse
import asyncio
import contextlib
import json
from unittest.mock import MagicMock, patch

import code_review_orchestrator as cro
from utils import review_orchestrator_base as rob
from utils.status_file import write_status


def _suggestions(titles):
    return [
        {
            "title": title,
            "desc": f"Description for {title}",
            "importance": "MEDIUM",
            "reference": "Step 1",
            "type": "addition",
        }
        for title in titles
    ]


def _phase(tmp_path, files, models_requested=None):
    """A phase dir holding *files* (``stem -> suggestions``), ready to reaggregate.

    ``models_requested`` is written through ``write_status`` exactly as a real run
    writes its checkpoint; passing ``None`` writes no ``.status.json`` at all,
    which is the hand-copied / pre-checkpoint dir.
    """
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n\nDo the thing.\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    phase_dir = out_dir / "review-plan"
    phase_dir.mkdir(parents=True)

    for stem, suggestions in files.items():
        (phase_dir / f"{stem}.json").write_text(
            json.dumps(suggestions), encoding="utf-8"
        )

    if models_requested is not None:
        write_status(str(phase_dir), {
            "phase": "review-plan",
            "state": "models_complete",
            "started_at": "2026-01-01T00:00:00",
            "models_requested": list(models_requested),
            "models_completed": list(models_requested),
            "models_failed": {},
            "error": None,
            "plan_hash": "deadbeefcafe",
        })

    return plan, out_dir, phase_dir


def _args(**over):
    base = dict(
        skip_validation=False,
        validation_model="auto",
        timeout=None,
        force=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _reaggregate(plan, out_dir, phase_dir):
    return asyncio.run(
        rob.reaggregate_from_existing_files(
            prefix="plan",
            out_dir=str(out_dir),
            phase_dir=str(phase_dir),
            phase_name="review-plan",
            plan_path=str(plan),
            args=_args(),
            orchestrator_script="review_plan_orchestrator.py",
        )
    )


def _cro_phase(tmp_path, files, models_requested=None):
    """A code-review phase dir holding *files* (``stem -> issues``)."""
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n\nDo the thing.\n", encoding="utf-8")

    phase_dir = tmp_path / "out" / "code-review"
    phase_dir.mkdir(parents=True)

    for stem, issues in files.items():
        (phase_dir / f"{stem}.json").write_text(json.dumps(issues), encoding="utf-8")

    if models_requested is not None:
        write_status(str(phase_dir), {
            "phase": "code-review",
            "state": "models_complete",
            "started_at": "2026-01-01T00:00:00",
            "models_requested": list(models_requested),
            "models_completed": list(models_requested),
            "models_failed": {},
            "error": None,
            "plan_hash": "deadbeefcafe",
        })

    return plan, phase_dir


def _run_cro_reaggregate(
    plan, phase_dir, *, skip_validation=True, render_report=False, extra_patches=(),
    args_over=None, outcome=None, state_record=None,
):
    """Drive the code-review reaggregate; return the patched report mock.

    Everything patched here is git/state/report machinery the identity question
    does not touch — ``build_unsanitize_map`` is deliberately NOT patched, since
    the recorded checkpoint is the whole point.

    ``render_report=True`` leaves the *real* ``generate_review_report`` in place,
    which is the only way to assert on what ``report.md`` actually says about a
    model; it is off by default so the identity tests keep asserting on the
    ``results`` dict handed to the renderer rather than on rendered markdown.
    ``extra_patches`` takes already-built ``patch`` objects (a spy on
    ``check_task_file_inputs``, a ``group_similar_suggestions`` rigged to raise)
    so a caller can pin what was *called* and not only what was written.
    ``args_over`` overrides parsed flags (``fail_on_high``); ``outcome``, a
    dict, receives the coroutine's return value (``"exit"``) and the fake state
    (``"state"``) so a caller can assert on the phase's completion.
    ``state_record``, a dict, stands in for the fake state's ``state`` (the
    ``state.json`` document) so a caller can read what was written to it. The
    fake is what both modules load: the review base's
    ``reopen_phase_for_validation`` reads ``state.json`` through its own import.
    """
    args = argparse.Namespace(
        plan_file=str(plan),
        models=["claude-code:sonnet"],
        interactive=False,
        quick=False,
        timeout=None,
        max_parallel=2,
        base_ref=None,
        skip_validation=skip_validation,
        validation_model="auto",
        apply_fixes=False,
        force=False,
        rerun_all=False,
        reaggregate=True,
        report_style="pr",
    )
    for name, value in (args_over or {}).items():
        setattr(args, name, value)

    fake_state = MagicMock()
    fake_state.get = MagicMock(
        side_effect=lambda k, default=None: {
            "tracked_files": [{"path": "src/a.py"}],
            "head_at_start": "abc123",
        }.get(k, default)
    )
    if state_record is not None:
        fake_state.state = state_record
    report = MagicMock(return_value="# report\n")

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch.object(cro, "get_or_create_state", return_value=fake_state))
        stack.enter_context(
            patch.object(rob, "get_or_create_state", return_value=fake_state))
        stack.enter_context(
            patch.object(cro, "get_phase_dir", return_value=phase_dir))
        stack.enter_context(
            patch.object(cro, "validate_git_ref", side_effect=lambda r: r or "HEAD~1"))
        stack.enter_context(
            patch.object(cro, "intent_to_add_untracked",
                         side_effect=lambda files, **_: contextlib.nullcontext()))
        if not render_report:
            stack.enter_context(patch.object(cro, "generate_review_report", report))
        for extra in extra_patches:
            stack.enter_context(extra)
        returned = asyncio.run(
            cro.reaggregate_from_existing_files(
                "plan", plan.parent / "out", plan, args
            )
        )
    if outcome is not None:
        outcome.update(exit=returned, state=fake_state)
    return report


def _reported_results(report_mock):
    """The ``results`` dict code-review reaggregate hands the report generator."""
    return report_mock.call_args[0][1]


def _cro_report_md(plan):
    """`report.md` where the code-review reaggregate writes it: the plan-derived
    phase dir (`get_output_paths`), whatever `get_phase_dir` was patched to."""
    path = cro.get_output_paths(plan, "code_review", phase="code-review")
    return path.with_suffix(".md").read_text(encoding="utf-8")
