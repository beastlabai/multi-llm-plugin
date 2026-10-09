#!/usr/bin/env python3
"""A re-run review reads each model's new answer, never the previous round's file.

**What was wrong** (measured 2026-10-09 with the suite's mock providers).
``invoke_with_file_output`` read the per-model output file before it looked at
stdout and never checked that the call it was serving wrote it. A review re-run
over a phase dir that still held the previous round's ``{model}.json`` —
``--review-plan --force --rerun-all``, ``--review-code --force --rerun-all``,
and the ``--improve-plan --force`` / ``--improve-code --force`` rounds that
forward both — handed every model that answered on stdout (cursor-agent,
gemini, codex, aider, qwen-code, vibe, copilot: every ``supports_file_output =
False`` adapter), or simply did not write the file this time, the previous
round's answer as its new one: the log said ``source: file`` and the old
findings were saved again. A file-writing mock got a fresh answer.

**The rule now.** The requested file is the slot's answer only if it did not
exist when the slot started, or it changed since (``requested_baseline``, a
stamp of inode, size and mtime). The baseline is the slot's, like the
misnamed-result baseline beside it: taken once before the first attempt and
handed to every retry, so a file an earlier attempt wrote before it was killed
is still this slot's, and the previous round's file is never this slot's on any
attempt. Nor anyone else's afterwards (Fable review 2026-10-09): the attempt
that finds it unchanged renames it ``<stem>.json.stale-earlier-run``, because
refused but left on disk it was read back at exit 0 by the ``--exec-subagents``
salvage recovery loop and by ``--reaggregate``; and that loop now adopts only a
file this attempt's salvage dispatch names as repaired. The unit tests of the
rule sit beside the other ``invoke_with_file_output`` tests in
``test_llm_client.py``; these pin the ladder and whole review-plan and
review-code rounds.

Usage:
    uv run --project skills/multi-llm -- pytest skills/multi-llm/tests/test_rerun_result_freshness.py -v
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness import SkillRunner  # noqa: E402
from utils import review_orchestrator_base as rob  # noqa: E402
from utils.llm_client import (  # noqa: E402
    ERROR_PARSE_ERROR,
    ERROR_TIMEOUT,
    requested_result_baseline,
)

#: A stdout-only adapter: `supports_file_output = False`, so the prompt asks it
#: for the array on stdout and its mock writes no file.
SPEC = "cursor-agent:auto"
STEM = "cursor-agent_auto"

ROUND_ONE = [{
    "title": "Bound the upload retry",
    "desc": "Step 1 retries a failed upload without saying how often.",
    "type": "clarification",
    "reference": "## Step 1",
    "importance": "MEDIUM",
}]
ROUND_TWO = [{
    "title": "Name the archive's storage class",
    "desc": "The overview uploads to the archive but never says which tier.",
    "type": "addition",
    "reference": "## Overview",
    "importance": "LOW",
}]

PLAN_BODY = """# Upload Plan

## Overview
Upload the user's files to the archive.

## Step 1
Retry a failed upload.
"""


def _titles(path: Path):
    return [item["title"] for item in json.loads(path.read_text(encoding="utf-8"))]


# ---------------------------------------------------------------------------
# 1. The review base's ladder hands every attempt the slot's baseline
# ---------------------------------------------------------------------------

def _no_sleep(_delay):
    async def _noop():
        return None
    return _noop()


def _timeout():
    return {"success": False, "error": "cursor-agent timed out after 600s",
            "error_code": ERROR_TIMEOUT, "details": {"provider": "cursor-agent"}}


def _stdout(data):
    return {"success": True, "data": data, "details": {"provider": "cursor-agent"}}


def _prose():
    return {"success": False, "error": "No valid JSON in stdout",
            "error_code": ERROR_PARSE_ERROR, "data": None,
            "details": {"provider": "cursor-agent"}}


class TestTheLadderJudgesTheFileAgainstTheSlotsStart:
    def _slot(self, tmp_path, replies, *, old_round=True):
        """One review-plan slot of two attempts; *replies* answer them in turn.

        Returns ``(result, the requested file)``. With *old_round*, the phase
        dir holds round one's file, written an hour before the slot started.
        """
        plan = tmp_path / "plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        phase_dir = rob.get_phase_dir(plan, "review-plan")
        phase_dir.mkdir(parents=True, exist_ok=True)
        requested = phase_dir / f"{STEM}.json"
        if old_round:
            requested.write_text(json.dumps(ROUND_ONE), encoding="utf-8")
            an_hour_ago = time.time() - 3600
            os.utime(requested, (an_hour_ago, an_hour_ago))
        replies = list(replies)

        def provider(*_args, **_kwargs):
            reply = replies.pop(0)
            return reply(requested) if callable(reply) else reply

        with patch("utils.llm_client.invoke_with_provider", side_effect=provider), \
                patch.object(rob, "get_subagent_retries", return_value=1), \
                patch.object(rob.asyncio, "sleep", _no_sleep):
            result = asyncio.run(rob.run_single_model(
                SPEC, str(plan), "review {output_json_path}", {},
                "review-plan", "plan_review",
                timeout=1800, out_dir=str(phase_dir.parent), prefix="plan",
                run_model_specs=[SPEC],
            ))
        assert not replies, "the ladder did not make both attempts"
        return result, requested

    def test_a_retry_answering_on_stdout_is_served_its_own_answer(self, tmp_path):
        result, requested = self._slot(tmp_path, [_timeout(), _stdout(ROUND_TWO)])

        assert result["success"] is True, result.get("error")
        assert result["source"] == "stdout_fallback"
        assert json.loads(result["output"]) == ROUND_TWO

    def test_a_retry_with_no_answer_fails_rather_than_reuse_round_one(self, tmp_path):
        result, requested = self._slot(tmp_path, [_timeout(), _prose()])

        assert result["success"] is False
        assert not result.get("output")
        # Set aside by the first attempt, so nothing after the slot reads it
        # as this round's answer (Fable review 2026-10-09).
        assert not requested.exists()
        assert _titles(requested.with_name(f"{STEM}.json.stale-earlier-run")) == [
            ROUND_ONE[0]["title"]
        ]

    def test_the_file_the_first_attempt_wrote_is_still_this_slots(self, tmp_path):
        """Per slot, not per attempt: attempt 1 rewrote the file and was killed,
        attempt 2 answered in prose. Taken per attempt, the rewritten file was
        in attempt 2's baseline and the slot failed beside its own answer."""
        def write_then_die(requested):
            requested.write_text(json.dumps(ROUND_TWO), encoding="utf-8")
            return _timeout()

        result, requested = self._slot(tmp_path, [write_then_die, _prose()])

        assert result["success"] is True, result.get("error")
        assert result["source"] == "file"
        assert json.loads(result["output"]) == ROUND_TWO

    def test_every_attempt_is_handed_the_one_baseline(self, tmp_path):
        plan = tmp_path / "plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        phase_dir = rob.get_phase_dir(plan, "review-plan")
        phase_dir.mkdir(parents=True, exist_ok=True)
        requested = phase_dir / f"{STEM}.json"
        requested.write_text(json.dumps(ROUND_ONE), encoding="utf-8")
        slot_start = requested_result_baseline(phase_dir.parent, "plan_review", SPEC)
        seen = []

        def fake_invoke(**kwargs):
            seen.append(kwargs.get("requested_baseline"))
            # Each attempt rewrites the file: a per-attempt baseline would move.
            requested.write_text(json.dumps(ROUND_TWO * len(seen)), encoding="utf-8")
            return _timeout()

        with patch.object(rob, "invoke_with_file_output", fake_invoke), \
                patch.object(rob, "get_subagent_retries", return_value=1), \
                patch.object(rob.asyncio, "sleep", _no_sleep):
            asyncio.run(rob.run_single_model(
                SPEC, str(plan), "review {output_json_path}", {},
                "review-plan", "plan_review",
                timeout=1800, out_dir=str(phase_dir.parent), prefix="plan",
            ))

        assert len(seen) == 2
        assert seen == [slot_start, slot_start]
        assert slot_start != ()


# ---------------------------------------------------------------------------
# 2. A whole `--force --rerun-all` review-plan round, through the mock binaries
# ---------------------------------------------------------------------------

class TestAForceRerunAllRoundSavesTheNewAnswer:
    def test_a_stdout_only_reviewer_s_round_two_answer_is_saved(self, tmp_path):
        runner = SkillRunner(tmp_path / "harness")
        plan = tmp_path / "upload-plan.md"
        plan.write_text(PLAN_BODY, encoding="utf-8")
        phase = plan.parent / plan.stem / "review-plan"
        result_file = phase / f"{STEM}.json"

        runner.set_config({"response": ROUND_ONE})
        first = runner.run_orchestrator(
            "review_plan", plan, "--models", SPEC, "--yes", timeout=120,
        )
        assert first.exit_code == 0, first.stdout + first.stderr
        assert _titles(result_file) == [ROUND_ONE[0]["title"]]

        runner.set_config({"response": ROUND_TWO})
        second = runner.run_orchestrator(
            "review_plan", plan, "--models", SPEC, "--yes", "--force", "--rerun-all",
            timeout=120,
        )

        assert second.exit_code == 0, second.stdout + second.stderr
        assert second.mock_was_invoked()
        assert _titles(result_file) == [ROUND_TWO[0]["title"]]
        grouped = (phase / "grouped.json").read_text(encoding="utf-8")
        assert ROUND_TWO[0]["title"] in grouped
        assert ROUND_ONE[0]["title"] not in grouped
        log = (phase / f"log_{STEM}.txt").read_text(encoding="utf-8")
        assert "Result source: stdout_fallback" in log
        assert "Result source: file" not in log


# ---------------------------------------------------------------------------
# 3. The file a failed slot refused is set aside, so nothing downstream reads it
# ---------------------------------------------------------------------------
#
# Fable review 2026-10-09: refusing the previous round's `{model}.json` inside
# the slot was not enough while the file stayed on disk. Two readers took it
# back as this round's answer at exit 0: the `--exec-subagents` salvage
# recovery loop (`{model}.json` exists and is not empty → "recovered by
# salvage", with no salvage dispatched for the model), and `--reaggregate`
# (the stem is a recorded model, so `set_aside_unrecorded_results` keeps the
# file and `record_models_present` moves the model out of `models_failed`).

#: The second reviewer, which answers round two properly, so the round has
#: something to group and the stale round-one file is the only way its
#: findings could come back.
OTHER_SPEC = "gemini:gemini-2.5-flash"
PROSE = "I reviewed the plan and found nothing I can put into the requested JSON."
SET_ASIDE = f"{STEM}.json.stale-earlier-run"


def _round_two_scenario(tmp_path: Path) -> Path:
    """cursor-agent answers in prose; gemini answers round two's array."""
    scenario = tmp_path / "round-two.yaml"
    scenario.write_text(
        "name: round-two\n"
        "prompts:\n"
        "  - pattern: \".\"\n"
        "    provider_filter: \"cursor-agent\"\n"
        f"    content: {json.dumps(PROSE)}\n"
        "  - pattern: \".\"\n"
        "    provider_filter: \"gemini\"\n"
        f"    content: {json.dumps(json.dumps(ROUND_TWO))}\n",
        encoding="utf-8",
    )
    return scenario


def _titles_in(path: Path):
    return json.dumps(json.loads(path.read_text(encoding="utf-8")))


class TestAFailedSlotLeavesNoEarlierRoundForAnyoneToRead:
    def _two_rounds(self, tmp_path, *round_two_flags):
        runner = SkillRunner(tmp_path / "harness")
        plan = tmp_path / "upload-plan.md"
        plan.write_text(PLAN_BODY, encoding="utf-8")
        phase = plan.parent / plan.stem / "review-plan"
        models = ("--models", SPEC, OTHER_SPEC)

        runner.set_config({"response": ROUND_ONE})
        first = runner.run_orchestrator(
            "review_plan", plan, *models, "--yes", "--skip-validation", timeout=120,
        )
        assert first.exit_code == 0, first.stdout + first.stderr
        assert _titles(phase / f"{STEM}.json") == [ROUND_ONE[0]["title"]]

        runner.set_scenario(_round_two_scenario(tmp_path))
        second = runner.run_orchestrator(
            "review_plan", plan, *models, "--yes", "--skip-validation",
            "--force", "--rerun-all", *round_two_flags, timeout=120,
        )
        return runner, plan, phase, second

    def _assert_round_one_is_gone(self, phase: Path):
        grouped = (phase / "grouped.json").read_text(encoding="utf-8")
        assert ROUND_TWO[0]["title"] in grouped
        assert ROUND_ONE[0]["title"] not in grouped
        status = json.loads((phase / ".status.json").read_text(encoding="utf-8"))
        assert SPEC in status["models_failed"], status.get("models_failed")
        assert SPEC not in status["models_completed"]

    def test_the_salvage_recovery_loop_does_not_adopt_it(self, tmp_path):
        """Exactly the executor leg's `--improve-plan --force` round-two argv."""
        _runner, _plan, phase, second = self._two_rounds(
            tmp_path, "--exec-subagents",
        )

        assert second.exit_code == 0, second.stdout + second.stderr
        assert f"[{SPEC}] recovered by salvage" not in second.stdout
        self._assert_round_one_is_gone(phase)
        # Set aside under its new name, never deleted, and named once.
        assert not (phase / f"{STEM}.json").exists()
        assert _titles(phase / SET_ASIDE) == [ROUND_ONE[0]["title"]]
        assert second.stdout.count(f"[RESULT_STALE] {STEM}.json") == 1
        assert SET_ASIDE in second.stdout

    def test_a_reaggregate_names_the_model_as_having_no_result(self, tmp_path):
        """The documented recovery after a partial round, on both lanes."""
        for lane, flags in (("native", ()), ("executor", ("--exec-subagents",))):
            root = tmp_path / lane
            root.mkdir()
            runner, plan, phase, second = self._two_rounds(root, *flags)
            assert second.exit_code == 0, second.stdout + second.stderr

            again = runner.run_orchestrator(
                "review_plan", plan, "--reaggregate", "--yes", "--skip-validation",
                timeout=120,
            )

            assert again.exit_code == 0, again.stdout + again.stderr
            assert f"[{SPEC}] result file present" not in again.stdout
            self._assert_round_one_is_gone(phase)
            report = (phase / "report.md").read_text(encoding="utf-8")
            assert f"**Models failed:** 1 of 2 requested — {SPEC}" in report
            assert ROUND_ONE[0]["title"] not in report
            assert _titles(phase / SET_ASIDE) == [ROUND_ONE[0]["title"]]


CODE_ROUND_ONE = [{
    "title": "Unchecked upload status",
    "desc": "The client ignores the HTTP status of the upload response.",
    "importance": "medium", "file": "src/upload/client.py", "line_range": [10, 12],
    "type": "bug", "anchor_text": "def send(path):",
}]
CODE_ROUND_TWO = [{
    "title": "Retry loop has no bound",
    "desc": "The retry loop never stops on a permanent failure.",
    "importance": "high", "file": "src/upload/retry.py", "line_range": [4, 9],
    "type": "bug", "anchor_text": "while True:",
}]


class TestTheCodeReviewSlotSetsItAsideToo:
    """The same rule on review-code's own slot, which reaches it through the
    same ``invoke_with_file_output`` (its loop was already gated on the
    salvage dispatch's repaired paths, so ``--reaggregate`` was its way in)."""

    def test_round_one_s_findings_do_not_come_back(self, tmp_path):
        from utils.state_manager import StateManager

        runner = SkillRunner(tmp_path / "harness")
        plan = tmp_path / "code-plan.md"
        plan.write_text(PLAN_BODY, encoding="utf-8")
        state = StateManager(plan)
        state.mark_phase_completed("review-plan")
        state.mark_phase_completed("implement")
        state.set("head_at_start", "abc123def")
        for issue in CODE_ROUND_ONE + CODE_ROUND_TWO:
            state.track_file(issue["file"], "created", "T001")
        state.save()
        phase = plan.parent / plan.stem / "code-review"
        models = ("--models", SPEC, OTHER_SPEC)

        runner.set_config({"response": CODE_ROUND_ONE})
        first = runner.run_orchestrator(
            "code_review", plan, *models, "--yes", "--skip-validation", timeout=120,
        )
        assert first.exit_code == 0, first.stdout + first.stderr
        assert _titles(phase / f"{STEM}.json") == [CODE_ROUND_ONE[0]["title"]]

        scenario = tmp_path / "code-round-two.yaml"
        scenario.write_text(
            "name: code-round-two\n"
            "prompts:\n"
            "  - pattern: \".\"\n"
            "    provider_filter: \"cursor-agent\"\n"
            f"    content: {json.dumps(PROSE)}\n"
            "  - pattern: \".\"\n"
            "    provider_filter: \"gemini\"\n"
            f"    content: {json.dumps(json.dumps(CODE_ROUND_TWO))}\n",
            encoding="utf-8",
        )
        runner.set_scenario(scenario)
        second = runner.run_orchestrator(
            "code_review", plan, *models, "--yes", "--skip-validation",
            "--force", "--rerun-all", "--exec-subagents", timeout=120,
        )
        assert second.exit_code == 0, second.stdout + second.stderr
        again = runner.run_orchestrator(
            "code_review", plan, "--reaggregate", "--yes", "--skip-validation",
            timeout=120,
        )

        assert again.exit_code == 0, again.stdout + again.stderr
        assert not (phase / f"{STEM}.json").exists()
        assert _titles(phase / SET_ASIDE) == [CODE_ROUND_ONE[0]["title"]]
        grouped = (phase / "grouped.json").read_text(encoding="utf-8")
        assert CODE_ROUND_TWO[0]["title"] in grouped
        assert CODE_ROUND_ONE[0]["title"] not in grouped
        status = json.loads((phase / ".status.json").read_text(encoding="utf-8"))
        assert SPEC in status["models_failed"], status.get("models_failed")


class TestTheSalvageLoopAdoptsOnlyWhatASalvageRepaired:
    """Defence in depth (Fable review 2026-10-09): the recovery loop moves a
    model out of ``failed_models`` only when this attempt's salvage dispatch
    names its result file as repaired, whatever else is on disk."""

    def test_a_result_file_no_salvage_wrote_is_not_a_recovery(self, tmp_path):
        from test_review_plan_orchestrator import TestAnInProcessSalvageMovesTheMapWithTheReport as T
        import review_plan_orchestrator as rpo
        from utils import status_file

        plan = tmp_path / "plan.md"
        plan.write_text(PLAN_BODY, encoding="utf-8")
        phase_dir = rob.get_phase_dir(plan, "review-plan")
        good, bad = T.MODELS
        helper = T()

        async def fake_run_all_models(*_args, **kwargs):
            tracker = kwargs["tracker"]
            tracker.seed()
            # Whatever put it there, it is no salvage's repair.
            (Path(phase_dir) / f"{rob.sanitize_model_name(bad)}.json").write_text(
                json.dumps(ROUND_ONE), encoding="utf-8"
            )
            results = {
                good: helper._record(good, success=True),
                bad: helper._record(bad, success=False, error="answered in prose"),
            }
            for spec, result in results.items():
                tracker.record_result(spec, result)
            return results

        argv = [
            "--plan-file", str(plan), "--models", *T.MODELS,
            "--skip-validation", "--timeout", "120", "--max-parallel", "2",
            "--exec-subagents",
        ]
        with patch.object(rob, "run_all_models", new=fake_run_all_models), \
                patch.object(rob, "dispatch_salvage_tasks", return_value=[]), \
                patch.object(rob, "PROVIDER_STAGGER_DELAY", 0.0):
            assert asyncio.run(rpo.main(argv)) == 0

        status = status_file.read_status(str(phase_dir))
        assert bad in status["models_failed"]
        assert status["models_completed"] == [good]
        assert status["progress"]["models"][bad]["state"] == "failed"
        grouped = (Path(phase_dir) / "grouped.json").read_text(encoding="utf-8")
        assert ROUND_ONE[0]["title"] not in grouped
