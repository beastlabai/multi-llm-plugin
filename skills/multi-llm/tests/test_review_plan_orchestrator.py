"""§Fix2: a successful ``review-plan`` run reaches the phase machine's own
terminal ``state`` — ``complete`` — not just ``grouping_complete`` or whichever
intermediate step the run's last write happened to leave behind.

One seam, following ``tests/test_e2e_failures.py``'s ``_run_cro_review``
pattern: ``review_plan_orchestrator.main()`` runs for real — the fan-out,
``save_model_result``, grouping, the report and the phase-completion
bookkeeping are all real — and only ``invoke_with_file_output`` (the one call
that would otherwise spawn a provider CLI) is replaced with a canned reply.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import review_plan_orchestrator as rpo  # noqa: E402
from utils import review_orchestrator_base as base  # noqa: E402
from utils import run_status, status_file  # noqa: E402


def _plan(tmp_path: Path) -> Path:
    plan = tmp_path / "auth-feature.md"
    plan.write_text("# Plan\n\nAdd an email field with validation.\n", encoding="utf-8")
    return plan


def _ok_reply(model_spec: str) -> dict:
    """The shape ``invoke_with_file_output`` returns on a clean file-channel success."""
    provider, _, model = model_spec.partition(":")
    return {
        "success": True,
        "data": [{
            "title": "Add input validation",
            "desc": "The plan does not validate the new email field.",
            "importance": "HIGH",
            "reference": "Overview",
            "type": "addition",
        }],
        "source": "file",
        "output_file": "unused.json",
        "details": {"provider": provider, "model": model or "auto"},
    }


def _run_review_plan(tmp_path: Path, models=("cursor-agent:auto",)):
    """Drive ``review_plan_orchestrator.main()`` end to end, foreground, one seam."""
    plan = _plan(tmp_path)
    argv = [
        "--plan-file", str(plan),
        "--models", *models,
        "--skip-validation",
        "--timeout", "5",
        "--max-parallel", "2",
    ]
    with patch.object(
        base, "invoke_with_file_output",
        side_effect=lambda **kwargs: _ok_reply(kwargs["model_spec"]),
    ), patch.object(base, "PROVIDER_STAGGER_DELAY", 0.0):
        exit_code = asyncio.run(rpo.main(argv))
    phase_dir = base.get_phase_dir(plan, "review-plan")
    return plan, phase_dir, exit_code


class TestPhaseStateReachesComplete:
    def test_a_successful_run_ends_with_state_complete(self, tmp_path):
        plan, phase_dir, exit_code = _run_review_plan(tmp_path)

        assert exit_code == 0

        status = status_file.read_status(str(phase_dir))
        # Before §Fix2 this was whichever intermediate step the run last
        # passed through (``grouping_complete`` here, since this run has a
        # suggestion to group) — never the terminal value status_file.py's
        # own module docstring documents.
        assert status.get("state") == "complete"
        # `_detach.report_progress` is a no-op outside `--detach` (by design —
        # a foreground run's .status.json gains no new keys), so `progress`
        # stays whatever a foreground run has always left it: absent. The
        # detached case (where it DOES advance to "complete", and where
        # `status` renders it) is covered directly in
        # ``tests/test_detach.py::TestStatusSurface::
        # test_complete_phase_state_renders_as_finished``.
        assert "progress" not in status or status["progress"].get("step") != "grouping"

    def test_a_run_with_nothing_to_group_also_reaches_complete(self, tmp_path):
        """The grouping/validation block is conditional on ``all_suggestions``
        being non-empty; the terminal write must not depend on having taken
        it — a model that completed and found nothing must reach ``complete``
        exactly like one that found something.
        """
        plan = _plan(tmp_path)
        argv = [
            "--plan-file", str(plan),
            "--models", "cursor-agent:auto",
            "--skip-validation",
            "--timeout", "5",
        ]
        # A completed review with zero findings: `data` is the JSON-array
        # *string* `"[]"` (truthy), never the Python list `[]` (falsy) —
        # the latter collapses to an empty `output` in run_single_model and
        # would be recorded as a failed slot instead, which is not the case
        # this test is about.
        empty_reply = {
            "success": True,
            "data": "[]",
            "source": "file",
            "output_file": "unused.json",
            "details": {"provider": "cursor-agent", "model": "auto"},
        }
        with patch.object(
            base, "invoke_with_file_output", side_effect=lambda **kwargs: dict(empty_reply)
        ), patch.object(base, "PROVIDER_STAGGER_DELAY", 0.0):
            exit_code = asyncio.run(rpo.main(argv))

        assert exit_code == 0
        phase_dir = base.get_phase_dir(plan, "review-plan")
        status = status_file.read_status(str(phase_dir))
        # Never reached grouping — no suggestions to group — so the last
        # state write before §Fix2 would have been "models_complete", not
        # "grouping_complete" as in the first test.
        assert status.get("state") == "complete"

    def test_status_renders_the_completed_run_as_finished(self, tmp_path, capsys):
        """§Fix2's second half: ``status`` (run_status.py) shows it too."""
        plan, phase_dir, exit_code = _run_review_plan(tmp_path)
        assert exit_code == 0

        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        out = capsys.readouterr().out
        # A foreground run never claims a `run` object (only --detach does),
        # so the run view reports "no run recorded" for liveness — but the
        # phase's own state, read from the same .status.json, must still say
        # the phase finished rather than naming an intermediate step.
        assert "phase state:    complete" in out
