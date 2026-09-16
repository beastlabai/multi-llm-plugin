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
from utils import cli_common, review_orchestrator_base as base  # noqa: E402
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


# ===========================================================================
# The consolidation recommendation leaves the log
# ===========================================================================

def _three_findings(model_spec: str) -> dict:
    """Three deliberately unrelated findings, so grouping yields three groups."""
    provider, _, model = model_spec.partition(":")
    return {
        "success": True,
        "data": [
            {
                "title": "Add email input validation",
                "desc": "The plan never says the email field is validated.",
                "importance": "HIGH",
                "reference": "Overview",
                "type": "addition",
            },
            {
                "title": "Pin the database migration order",
                "desc": "Migrations run in an unspecified sequence at deploy.",
                "importance": "MEDIUM",
                "reference": "Rollout",
                "type": "clarification",
            },
            {
                "title": "Name the session expiry window",
                "desc": "Logout timing is left entirely to the implementer.",
                "importance": "LOW",
                "reference": "Security",
                "type": "clarification",
            },
        ],
        "source": "file",
        "output_file": "unused.json",
        "details": {"provider": provider, "model": model or "auto"},
    }


def _run_with_validation(tmp_path: Path, threshold: int):
    """A run whose groups all validate, with the recommendation threshold given.

    Two more seams than the runs above, both standing in for work a provider
    would do: ``validate_groups`` (the LLM call) and
    ``apply_validation_to_groups`` (which joins its verdicts back). Everything
    the recommendation itself depends on — the fan-out, the grouping, the
    count — is still the real thing.
    """
    plan = _plan(tmp_path)
    argv = [
        "--plan-file", str(plan),
        "--models", "cursor-agent:auto",
        "--internal-validation",
        "--consolidation-threshold", str(threshold),
        "--timeout", "5",
    ]

    async def _validate(groups, **kwargs):
        return []

    def _apply(groups, results):
        return [dict(group, validation_status="valid") for group in groups]

    context = cli_common.activate_result(cli_common.ResultContext())
    try:
        with patch.object(
            base, "invoke_with_file_output",
            side_effect=lambda **kwargs: _three_findings(kwargs["model_spec"]),
        ), patch.object(base, "PROVIDER_STAGGER_DELAY", 0.0), \
                patch.object(base, "validate_groups", _validate), \
                patch.object(base, "apply_validation_to_groups", _apply):
            exit_code = asyncio.run(rpo.main(argv))
    finally:
        cli_common.deactivate_result()
    phase_dir = base.get_phase_dir(plan, "review-plan")
    return plan, phase_dir, exit_code, context


def _group_count(phase_dir) -> int:
    """The run's own count, as it recorded it at ``grouping_complete``."""
    return status_file.read_status(str(phase_dir))["total_groups"]


class TestConsolidationRecommendationLeavesTheLog:
    """The `[CONSOLIDATION_RECOMMENDED]` marker was a bare ``print``.

    A detached caller never reads the log, so the recommendation reached no
    host: not `.status.json`, not `output_files`, not the result line — and
    deliberately still not `next`. Measured 2026-09-16, a 29-group run printed
    the marker and the session model told the user no consolidation had been
    recommended.
    """

    def test_the_recommendation_reaches_the_checkpoint_and_the_result_line(
        self, tmp_path
    ):
        plan, phase_dir, exit_code, context = _run_with_validation(
            tmp_path, threshold=2
        )
        assert exit_code == 0
        assert _group_count(phase_dir) >= 2, "fewer groups than the threshold"

        expected = {
            "valid_groups": _group_count(phase_dir),
            "threshold": 2,
            "command": "--consolidate",
        }
        status = status_file.read_status(str(phase_dir))
        assert status.get("consolidation_recommended") == expected
        assert context.payload(0)["consolidation_recommended"] == expected
        # Never `next`: `--consolidate` is one of the dispatcher's mechanical
        # continuations, which a host may follow without asking, and following a
        # recommendation would run a phase nobody requested.
        assert context.payload(0)["next"] is None

    def test_a_run_below_the_threshold_records_nothing(self, tmp_path):
        """Absent, not ``null``: every other run's result line is unchanged."""
        _, phase_dir, exit_code, context = _run_with_validation(
            tmp_path, threshold=99
        )
        assert exit_code == 0
        status = status_file.read_status(str(phase_dir))
        assert "consolidation_recommended" not in status
        assert "consolidation_recommended" not in context.payload(0)

    def test_status_prints_the_recommendation_and_hands_it_back(
        self, tmp_path, capsys
    ):
        """The route a detached host really reads: `status --view run`."""
        plan, phase_dir, exit_code, _ = _run_with_validation(tmp_path, threshold=2)
        assert exit_code == 0
        capsys.readouterr()
        groups = _group_count(phase_dir)

        context = cli_common.activate_result(cli_common.ResultContext())
        try:
            # 2: a foreground run claims no `run` record, so the run view has no
            # liveness to report — the phase's own checkpoint is still read.
            assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
            assert context.consolidation_recommended == {
                "valid_groups": groups,
                "threshold": 2,
                "command": "--consolidate",
            }
        finally:
            cli_common.deactivate_result()
        out = capsys.readouterr().out
        assert (
            f"consolidation recommended: {groups} valid groups >= 2 "
            "— run --consolidate"
        ) in out
class TestTheFanOutReportsProgressWhileItRuns:
    """F5: `progress.models_done` was written `0` once, before the first launch,
    and recomputed only after the last model returned — so every poll during the
    fan-out (435 s of a 516 s run, measured 2026-09-16) read `models 0/4` beside
    models that had already finished."""

    #: A budget every slot can hold: `run_single_model` refuses to start a
    #: dispatch whose runner's declared worst-case envelope is larger than what
    #: is left of the slot, and this suite's 5 s default refuses codex outright.
    TIMEOUT = "120"

    def _recorded(self, tmp_path, models, *, force=False):
        """One real `review-plan` run, recording every progress report it made."""
        plan = _plan(tmp_path)
        argv = [
            "--plan-file", str(plan), "--models", *models,
            "--skip-validation", "--timeout", self.TIMEOUT,
            "--max-parallel", str(len(models)),
        ] + (["--force"] if force else [])
        calls = []
        real = base._detach.report_progress

        def record(phase_dir, **updates):
            calls.append(dict(updates))
            real(phase_dir, **updates)

        with patch.object(
            base, "invoke_with_file_output",
            side_effect=lambda **kwargs: _ok_reply(kwargs["model_spec"]),
        ), patch.object(base, "PROVIDER_STAGGER_DELAY", 0.0), patch.object(
            base._detach, "report_progress", record
        ):
            assert asyncio.run(rpo.main(argv)) == 0
        return calls

    def test_each_finished_model_advances_the_counter(self, tmp_path):
        calls = self._recorded(
            tmp_path, ("cursor-agent:auto", "codex:gpt-5.5", "opencode:auto")
        )
        counted = [c["models_done"] for c in calls if "models_done" in c]
        # The opening zero, one per finished slot, then the phase's own
        # recomputation from the models that produced a result.
        assert counted == [0, 1, 2, 3, 3]
        assert all(c["models_total"] == 3 for c in calls if "models_total" in c)

    def test_a_resumed_run_counts_the_slots_it_answers_from_disk(self, tmp_path):
        """A skipped model has finished its slot; a resume that skips every
        model must not read `models 0/2` for the whole of it."""
        models = ("cursor-agent:auto", "codex:gpt-5.5")
        self._recorded(tmp_path, models)  # writes both `{model}.json` files
        counted = [
            c["models_done"]
            for c in self._recorded(tmp_path, models, force=True)
            if "models_done" in c
        ]
        assert counted == [0, 1, 2, 2]
