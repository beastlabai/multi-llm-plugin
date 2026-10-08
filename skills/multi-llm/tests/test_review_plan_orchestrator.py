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
import contextlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import review_plan_orchestrator as rpo  # noqa: E402
from utils import cli_common, review_orchestrator_base as base  # noqa: E402
from utils import progress_tracker, run_status, status_file  # noqa: E402
from utils.validation import prepare_batched_validation_tasks  # noqa: E402


#: The reserved ``progress`` keys, as ``utils/status_file.py``'s module docstring
#: enumerates them. A foreground run writes this object now (``report_progress``
#: falls back to ``status_file.update_progress`` when no run handle owns the phase
#: dir), so the foreground assertions below pin its *shape*: the keys a reader of
#: the checkpoint is allowed to meet, and never a ``run`` object beside them.
_RESERVED_PROGRESS_KEYS = {
    "step", "batch_index", "batches_total", "tasks_done", "tasks_total",
    "failed_task_ids", "models", "models_done", "models_succeeded",
    "models_total", "interval_s",
    # Written straight through ``update_status`` by the two executor mirrors,
    # so they already reach a foreground checkpoint.
    "subagents", "write_subagents", "sandbox_bypassed",
    # Landlock rung (2026-09-22): the contained-rung slots, `{spec: rung}`.
    "sandbox_fallback",
}


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
        # `report_progress` grew a foreground write path (no run handle ->
        # `status_file.update_progress` directly), so the invariant narrowed:
        # a foreground `.status.json` gains a `progress` object and still no
        # `run` object. Assert the shape rather than the absence — `progress`
        # is display-only and nothing resumes off it, while `run` is the
        # process machine only a detached worker's `claim_run` ever writes, and
        # that asymmetry is the whole of what "with --detach unset, behavior is
        # unchanged" still means.
        progress = status.get("progress") or {}
        assert progress.get("step") == "complete"
        assert progress.get("models_done") == 1
        assert progress.get("models_total") == 1
        # Every key drawn from the reserved set `utils/status_file.py`'s module
        # docstring enumerates: a foreground checkpoint is not a place for a
        # writer to invent a key nobody documented.
        assert set(progress) <= _RESERVED_PROGRESS_KEYS, progress
        assert "run" not in status, (
            "a foreground run claims no run: `progress` is the only reserved "
            "object it writes"
        )

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
    models that had already finished.

    T4 consolidated the writer: D4a's `fanout_slot_counter` is deleted and
    `progress_tracker.FanoutProgress`'s map derives the same number, because a
    map-derived count plus a counter is two writers for one key and they tick at
    different moments. So these two tests no longer pin the *call sequence*
    literally — a tracker that seeds the map and writes on every transition
    makes many more calls than the counter did — they pin the **property** those
    two regressions were actually about, which is what the literal list was ever
    standing in for:

    * the sequence read off `models_done` never decreases (a live progress line
      cannot use the success meaning: under it the number walks backwards at the
      end of a lossy run and can never reach `8/8`),
    * it reaches `models_total`,
    * and on a resumed run it reaches the skipped count with **nothing
      dispatched** — D4a itself, landing on the one lane where no attempt is
      ever made, so the number can never be read as "still waiting".

    One write is allowed to say `0`, and only one: the seed, which is written
    before the tasks exist so that a poller meets a *complete* map from its
    first read instead of one that fills in as models start. Nothing has been
    evaluated at that point, not even a resume skip. From there on the two
    numbers are derived from the map itself on every write, and the tests below
    assert that agreement directly — it is what makes "`models 0/4` beside
    models that had already finished" impossible by construction rather than by
    a counter and a map happening to tick together.
    """

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
        self.dispatched = []
        real = base._detach.report_progress

        def record(phase_dir, **updates):
            calls.append(dict(updates))
            real(phase_dir, **updates)

        def dispatch(**kwargs):
            self.dispatched.append(kwargs["model_spec"])
            return _ok_reply(kwargs["model_spec"])

        with patch.object(
            base, "invoke_with_file_output", side_effect=dispatch,
        ), patch.object(base, "PROVIDER_STAGGER_DELAY", 0.0), patch.object(
            base._detach, "report_progress", record
        ):
            assert asyncio.run(rpo.main(argv)) == 0
        return calls

    @staticmethod
    def _assert_counts_match_the_map(calls):
        """`models_done`/`models_succeeded` are derived, never accumulated.

        The failure this forbids is the one "one writer per lane" exists to
        forbid: a counter ticking at slot exit beside a map turning terminal at
        the completion print, so a poll lands between the two and reads a number
        that contradicts the rows printed beside it.
        """
        seen = 0
        for call in calls:
            models = call.get("models")
            if models is None:
                continue
            seen += 1
            terminal = [e for e in models.values() if e["state"] in ("done", "failed")]
            assert call["models_done"] == len(terminal), call
            assert call["models_succeeded"] == sum(
                1 for e in models.values() if e["state"] == "done"
            ), call
            assert call["models_total"] == len(models), call
        assert seen, "no write carried the per-model map at all"

    def test_each_finished_model_advances_the_counter(self, tmp_path):
        models = ("cursor-agent:auto", "codex:gpt-5.5", "opencode:auto")
        calls = self._recorded(tmp_path, models)
        counted = [c["models_done"] for c in calls if "models_done" in c]

        assert counted, "the fan-out reported no `models_done` at all"
        # Monotonic, which is the whole of what the old literal `[0, 1, 2, 3, 3]`
        # was asserting: `models_done` counts finished SLOTS, so it only ever
        # goes up. The lossy-run information that used to shrink it now rides on
        # the sibling key instead.
        assert counted == sorted(counted), counted
        assert counted[-1] == len(models)
        assert all(
            c["models_total"] == len(models) for c in calls if "models_total" in c
        )
        # The second counter, written by the post-fan-out restatement: every
        # model answered here, so "done" and "ok" agree.
        succeeded = [c["models_succeeded"] for c in calls if "models_succeeded" in c]
        assert succeeded and succeeded[-1] == len(models)
        self._assert_counts_match_the_map(calls)

    def test_a_resumed_run_counts_the_slots_it_answers_from_disk(self, tmp_path):
        """A skipped model has finished its slot; a resume that skips every
        model must not read `models 0/2` for the whole of it.

        The sharp end of the property: the FIRST write of a resumed run already
        reads `2/2`, because a model answered from disk is terminal the moment
        it is seeded and no attempt is ever dispatched.
        """
        models = ("cursor-agent:auto", "codex:gpt-5.5")
        self._recorded(tmp_path, models)  # writes both `{model}.json` files
        calls = self._recorded(tmp_path, models, force=True)
        counted = [c["models_done"] for c in calls if "models_done" in c]

        assert counted == sorted(counted), counted
        assert counted[-1] == len(models)
        # Nothing was dispatched — every slot was answered from disk — and the
        # count still reached `2/2`. Under the pre-D4a writer this whole run
        # reported `models 0/2`.
        assert self.dispatched == [], self.dispatched
        # Every write's counters agree with the map it carries, so no poll can
        # ever read a number that lags a model the same write shows finished.
        self._assert_counts_match_the_map(calls)


class TestAFanoutModelLostToTheGatherDeadline:
    """F11: a model whose task never returned — cancelled by
    `run_models_concurrent`'s own `asyncio.wait_for` deadline, or dropped as
    an `Exception` item there — is in neither `results` nor `failed_models`:
    without this fix it is counted completed, inflates `models_done`, and
    appears in no report section at all. Same gap
    `code_review_orchestrator.run_all_reviews`'s caller has, fixed the same
    way there.
    """

    def test_a_missing_model_is_recorded_failed_not_silently_completed(
        self, tmp_path
    ):
        plan = _plan(tmp_path)
        argv = [
            "--plan-file", str(plan),
            "--models", "cursor-agent:auto", "claude-code:sonnet",
            "--skip-validation", "--timeout", "5", "--max-parallel", "2",
        ]

        # `claude-code:sonnet` never made it into `run_all_models`'s own
        # return value -- exactly what an exception item or a total gather
        # timeout leaves behind.
        async def fake_run_all_models(*a, **kw):
            return {
                "cursor-agent:auto": {
                    "success": True,
                    "output": json.dumps([{
                        "title": "Add input validation",
                        "desc": "The plan does not validate the new email field.",
                        "importance": "HIGH", "reference": "Overview",
                        "type": "addition",
                    }]),
                    "error": None, "prompt": "", "stdout": "", "stderr": "",
                    "duration_seconds": 0.1, "model_spec": "cursor-agent:auto",
                    "source": "file",
                },
            }

        with patch.object(base, "run_all_models", new=fake_run_all_models):
            exit_code = asyncio.run(rpo.main(argv))

        assert exit_code == 0
        phase_dir = base.get_phase_dir(plan, "review-plan")
        status = status_file.read_status(str(phase_dir))
        assert status["models_failed"]["claude-code:sonnet"] == (
            "no result (cancelled at the fan-out deadline)"
        )
        assert "claude-code:sonnet" not in status["models_completed"]
        report = (phase_dir / "report.md").read_text(encoding="utf-8")
        assert "## Models Failed" in report
        assert "claude-code:sonnet" in report


def _validation_group(index: int) -> dict:
    return {
        "theme": f"Group {index}",
        "category": "test",
        "models": ["m"],
        "suggestions": [{
            "title": "t", "desc": "d", "importance": "MEDIUM", "type": "modification",
        }],
    }


class TestMergeValidationBatchesRunnerAttribution:
    """F3: ``validation.json``'s ``metadata.model`` must name who actually
    validated, not a subagent's free-hand self-report about itself — a
    captured run spelled the same runner five different ways across 11
    batches, one of them another model's name copied out of the suggestions
    it was validating.
    """

    def test_the_executors_own_dispatch_record_outranks_the_batchs_self_report(
        self, tmp_path
    ):
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()
        groups = [_validation_group(i) for i in range(2)]
        batch_metadata = prepare_batched_validation_tasks(
            groups=groups, context="ctx", output_dir=str(phase_dir),
            plan_file="/path/to/plan.md", max_per_batch=4,
        )
        batch = batch_metadata["batches"][0]
        # The self-report — deliberately wrong, to prove it loses to the
        # executor's own record below.
        batch_results = {
            "groups": [
                {"group_index": i, "status": "valid", "reason": "ok", "confidence": 0.9}
                for i in range(2)
            ],
            "metadata": {"model": "some-other-models-name", "schema_version": "2.1"},
        }
        with open(batch["output_path"], "w", encoding="utf-8") as f:
            json.dump(batch_results, f)

        # `--exec-subagents` dispatched this batch itself, so
        # `subagent_batch_report.json` carries the executor's own record of
        # what it ran — the authoritative spec `merge_validation_batches`
        # must prefer.
        (phase_dir / "subagent_batch_report.json").write_text(json.dumps({
            "report_version": 1, "invoker": "executor", "phase": "review-plan",
            "kind": "validation", "task_file": None,
            "tasks": [{
                "task_id": batch["task_id"], "status": "ok",
                "runner": "opencode:openrouter/~deepseek/deepseek-flash-latest",
            }],
        }), encoding="utf-8")

        validation_path = phase_dir / "validation.json"
        validated = base.merge_validation_batches(
            str(phase_dir), batch_metadata, groups, str(validation_path)
        )

        assert len(validated) == 2
        saved = json.loads(validation_path.read_text(encoding="utf-8"))
        assert saved["metadata"]["model"] == (
            "opencode:openrouter/~deepseek/deepseek-flash-latest"
        )
        assert saved["metadata"]["runner"] == (
            "opencode:openrouter/~deepseek/deepseek-flash-latest"
        )

    def test_falls_back_to_the_self_report_with_no_executor_record(self, tmp_path):
        """The host-native (Claude Code Task tool) path never runs the
        executor, so there is no ``subagent_batch_report.json`` to read — the
        best-effort derivation from the batch's own ``metadata.model`` is the
        only provenance available, and must still be used."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()
        groups = [_validation_group(i) for i in range(2)]
        batch_metadata = prepare_batched_validation_tasks(
            groups=groups, context="ctx", output_dir=str(phase_dir),
            plan_file="/path/to/plan.md", max_per_batch=4,
        )
        batch = batch_metadata["batches"][0]
        batch_results = {
            "groups": [
                {"group_index": i, "status": "valid", "reason": "ok", "confidence": 0.9}
                for i in range(2)
            ],
            "metadata": {"model": "claude-code:sonnet", "schema_version": "2.1"},
        }
        with open(batch["output_path"], "w", encoding="utf-8") as f:
            json.dump(batch_results, f)
        # No subagent_batch_report.json written — nothing was executor-dispatched.

        validation_path = phase_dir / "validation.json"
        base.merge_validation_batches(
            str(phase_dir), batch_metadata, groups, str(validation_path)
        )

        saved = json.loads(validation_path.read_text(encoding="utf-8"))
        assert saved["metadata"]["model"] == "claude-code:sonnet"
        assert "runner" not in saved["metadata"]


class TestTheHighPriorityGroupCountReachesTheHost:
    """The route a detached host reads: the result line and `.status.json`.
    review-code has recorded `high_priority_groups` there since F4; a plan
    review's HIGH suggestions were only ever a bare print."""

    def _run(self, tmp_path, importance):
        plan = _plan(tmp_path)
        argv = ["--plan-file", str(plan), "--models", "cursor-agent:auto",
                "--skip-validation", "--timeout", "5"]

        def reply(**kwargs):
            answer = _ok_reply(kwargs["model_spec"])
            answer["data"][0]["importance"] = importance
            return answer

        context = cli_common.activate_result(cli_common.ResultContext())
        try:
            with patch.object(base, "invoke_with_file_output", side_effect=reply), \
                    patch.object(base, "PROVIDER_STAGGER_DELAY", 0.0):
                assert asyncio.run(rpo.main(argv)) == 0
        finally:
            cli_common.deactivate_result()
        status = status_file.read_status(str(base.get_phase_dir(plan, "review-plan")))
        return status, context.payload(0)

    def test_a_high_suggestion_is_one_high_group_on_both_channels(self, tmp_path):
        status, payload = self._run(tmp_path, "HIGH")
        assert status.get("high_priority_groups") == {"count": 1}
        assert payload["high_priority_groups"] == {"count": 1}

    def test_a_review_with_no_high_suggestion_records_nothing(self, tmp_path):
        status, payload = self._run(tmp_path, "LOW")
        assert "high_priority_groups" not in status
        assert "high_priority_groups" not in payload


# ===========================================================================
# T4/T5: the per-model map, the two tickers, and every ending retiring every slot
# ===========================================================================

_LIVE_STATES = ("queued", "staggering", "running", "retrying")


@contextlib.contextmanager
def _fast_ticker(max_ticks: int = 6):
    """Run both tickers on a fake clock, bounded.

    ``progress_tracker`` exposes its sleep as a module-level seam precisely so a
    cadence test does not have to wait out a 30 s interval. Bounded because the
    fake sleep yields instead of waiting: left unbounded the ticker would spin
    for every loop iteration the fan-out takes and bury the run's own output in
    thousands of identical lines. Past the bound it parks on a real sleep it will
    be cancelled out of, which is exactly the shape of the production task.
    """
    emitted: list = []
    ticks: dict = {}

    async def fake_sleep(delay):
        # Budgeted per ticker TASK, not globally: the fan-out's ticker and the
        # phase's are two tasks with two lifetimes, and one shared counter would
        # let the first spend the second's budget and leave the phase window
        # silent — the exact failure this suite is here to catch.
        key = id(asyncio.current_task())
        ticks[key] = ticks.get(key, 0) + 1
        if ticks[key] > max_ticks:
            await asyncio.sleep(3600)
        await asyncio.sleep(0)

    def emit(line):
        emitted.append(line)
        # Still printed, so a test that cares about ORDER against the phase's
        # own output (the per-model summary) can read one captured stream.
        print(line, flush=True)

    with patch.object(progress_tracker, "_tick_sleep", fake_sleep), patch.object(
        progress_tracker, "_emit", emit
    ):
        yield emitted


@contextlib.contextmanager
def _realtime_ticker(interval_s: float = 0.01):
    """Both tickers on a real (tiny) interval rather than on a tick budget.

    ``_fast_ticker``'s fake sleep yields and then parks after six ticks per
    ticker task, which is the right shape for a cadence assertion and the wrong
    one for measuring whether the loop was *free* during a particular window:
    the phase ticker can spend its whole budget on the awaits before that window
    opens and then be parked through all of it, which is indistinguishable from
    being starved. A real 10 ms interval measures what "the ticker kept
    printing while the dispatch blocked" actually claims.
    """
    emitted: list = []

    async def sleep(_delay):
        await asyncio.sleep(interval_s)

    def emit(line):
        emitted.append(line)
        # Still printed, for the same reason `_fast_ticker` prints: a test that
        # cares about ORDER against the phase's own output reads one stream.
        print(line, flush=True)

    with patch.object(progress_tracker, "_tick_sleep", sleep), patch.object(
        progress_tracker, "_emit", emit
    ):
        yield emitted


def _progress_writes(calls):
    """Every recorded ``report_progress`` call that carried the per-model map."""
    return [c for c in calls if "models" in c]


class TestTheMapIsCompleteBeforeTheFirstSlotStarts:
    """The review-plan / review-tasks lane's own row of "all four fan-out modes".

    Four claims, and each is a bug the map was written to remove: the map is
    complete at the FIRST poll (seeded before the tasks exist, rather than
    filling in as models start, so a model missing from the map is a bug and
    never a leftover); the launch write drops any predecessor's map, on THIS
    lane and not only the detached one; a `[PROGRESS]` ticker really ran on
    *this* code path, not only on the shared helper's unit test; and the map is
    wholly terminal the moment the fan-out returns, because nothing may outlive
    it in `queued`/`staggering`/`running`/`retrying`.

    That second claim used to be credited to `claim_run`, which does drop the
    predecessor's `run` and `progress` wholesale — but only the *detached*
    worker calls it. On the foreground lane, the one Claude Code actually runs,
    nothing dropped it and `write_status` carries the reserved objects over from
    disk by design, so a re-review opened on the last run's rows (F1,
    2026-09-20).
    """

    MODELS = ("cursor-agent:auto", "codex:gpt-5.5", "opencode:auto")

    def _run(self, tmp_path):
        plan = _plan(tmp_path)
        argv = [
            "--plan-file", str(plan), "--models", *self.MODELS,
            "--skip-validation", "--timeout", "120",
            "--max-parallel", str(len(self.MODELS)),
        ]
        calls = []
        real = base._detach.report_progress

        def record(phase_dir, **updates):
            calls.append(dict(updates))
            real(phase_dir, **updates)

        with _fast_ticker() as emitted, patch.object(
            base, "invoke_with_file_output",
            side_effect=lambda **kwargs: _ok_reply(kwargs["model_spec"]),
        ), patch.object(base, "PROVIDER_STAGGER_DELAY", 0.0), patch.object(
            base._detach, "report_progress", record
        ):
            assert asyncio.run(rpo.main(argv)) == 0
        return base.get_phase_dir(plan, "review-plan"), calls, emitted

    def test_the_first_write_already_names_every_requested_model(self, tmp_path):
        _phase_dir, calls, _emitted = self._run(tmp_path)
        first = _progress_writes(calls)[0]
        assert set(first["models"]) == set(self.MODELS)
        # The seed, and only the seed, is allowed to be entirely live: nothing
        # has been dispatched or skipped yet.
        assert all(e["state"] == "queued" for e in first["models"].values())
        assert first["models_total"] == len(self.MODELS)
        # `timeout_s` is seeded from the same `get_provider_timeout` the fan-out
        # sizes its own deadline with, so a queued row renders `0m00s/2m`
        # instead of a blank denominator.
        assert all(e["timeout_s"] == 120 for e in first["models"].values())

    def test_the_launch_write_drops_a_predecessors_map(self, tmp_path):
        """A re-run of the same phase dir must not open on the last run's rows.

        Written as a pre-seeded phase dir rather than two real runs because the
        shape is what matters: whatever a predecessor left in `progress` —
        another model list, every row `done`, its own `models_succeeded` — is
        gone by the time this run reports anything. What made this worth a test
        is that the *counts* were reset here all along and the map was not, so
        the damage rendered as `models 0/3 done, 2 ok` over three rows nobody
        was running.
        """
        plan = _plan(tmp_path)
        phase = base.get_phase_dir(plan, "review-plan")
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "complete",
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

        argv = [
            "--plan-file", str(plan), "--models", *self.MODELS,
            "--skip-validation", "--timeout", "120",
            "--max-parallel", str(len(self.MODELS)),
        ]
        # The window this is about is between the launch `write_status` and the
        # tracker's `seed()`: the seed replaces `models` wholesale, so by the end
        # of the run the map is this run's either way and a final-state assertion
        # proves nothing. `report_progress` is the first thing called after the
        # launch write, so the disk as its first invocation *enters* is exactly
        # what a poll landing in that window would render.
        snapshots = []
        real = base._detach.report_progress

        def record(phase_dir, **updates):
            if not snapshots:
                snapshots.append(
                    status_file.read_status(str(phase_dir)).get("progress")
                )
            real(phase_dir, **updates)

        with _fast_ticker(), patch.object(
            base, "invoke_with_file_output",
            side_effect=lambda **kwargs: _ok_reply(kwargs["model_spec"]),
        ), patch.object(base, "PROVIDER_STAGGER_DELAY", 0.0), patch.object(
            base._detach, "report_progress", record
        ):
            assert asyncio.run(rpo.main(argv)) == 0

        assert snapshots, "the launch never reported progress"
        assert not snapshots[0], (
            f"the predecessor's progress survived the launch write: "
            f"{snapshots[0]}"
        )
        final = status_file.read_status(str(phase))["progress"]
        assert set(final["models"]) == set(self.MODELS)

    def test_a_ticker_ran_on_this_lane(self, tmp_path):
        _phase_dir, _calls, emitted = self._run(tmp_path)
        fanout_lines = [ln for ln in emitted if " done (" in ln]
        assert fanout_lines, emitted
        assert all(ln.startswith("[PROGRESS] ") for ln in emitted)
        # ASCII only: every host redirects this stream to a file, where a
        # redirected stream takes the locale codec.
        assert all(ln.encode("ascii", "strict") for ln in emitted)
        # The last fan-out line states the run's real totals rather than
        # whatever the last interval happened to catch.
        assert fanout_lines[-1].startswith(
            f"[PROGRESS] {len(self.MODELS)}/{len(self.MODELS)} done "
            f"({len(self.MODELS)} ok, 0 failed)"
        )

    def test_the_map_is_wholly_terminal_when_the_fan_out_returns(self, tmp_path):
        _phase_dir, calls, _emitted = self._run(tmp_path)
        grouping = next(i for i, c in enumerate(calls) if c.get("step") == "grouping")
        last_map = _progress_writes(calls[:grouping])[-1]["models"]
        assert set(last_map) == set(self.MODELS)
        assert not [
            spec for spec, e in last_map.items() if e["state"] in _LIVE_STATES
        ], last_map
        # A terminal entry is frozen: `ended_at` and `elapsed_s` together, so a
        # table rendered an hour later shows the slot's real duration.
        for entry in last_map.values():
            assert entry["ended_at"] and "elapsed_s" in entry
            assert "retry_after_s" not in entry


class TestAResumedRunDoesNotReportSkippedModelsAsPending:
    """Acceptance criterion 2. A model answered from disk is `done` the moment
    it is seeded — it has a usable result and no attempt is ever dispatched — so
    it counts in `models_done` *and* in `models_succeeded`, and a resumed run of
    four models that finds three on disk reads `3/4` at the first poll rather
    than `0/4`. There is deliberately no `skipped` state: the distinction lives
    in `source`, so the terminal set stays `done | failed` and a renderer that
    wants to label a row "skipped" reads `source in {existing_file, resume}`.
    """

    MODELS = ("cursor-agent:auto", "codex:gpt-5.5")

    def _run(self, tmp_path, *, force=False):
        plan = _plan(tmp_path)
        argv = [
            "--plan-file", str(plan), "--models", *self.MODELS,
            "--skip-validation", "--timeout", "120", "--max-parallel", "2",
        ] + (["--force"] if force else [])
        with patch.object(
            base, "invoke_with_file_output",
            side_effect=lambda **kwargs: _ok_reply(kwargs["model_spec"]),
        ), patch.object(base, "PROVIDER_STAGGER_DELAY", 0.0):
            assert asyncio.run(rpo.main(argv)) == 0
        return base.get_phase_dir(plan, "review-plan")

    def test_every_skipped_model_is_done_from_its_existing_file(self, tmp_path):
        self._run(tmp_path)
        phase_dir = self._run(tmp_path, force=True)

        progress = status_file.read_status(str(phase_dir))["progress"]
        models = progress["models"]
        assert set(models) == set(self.MODELS), "a requested spec has no row"
        for spec, entry in models.items():
            assert entry["state"] == "done", (spec, entry)
            assert entry["source"] == "existing_file", (spec, entry)
            assert entry["source"] in progress_tracker.RESUME_SOURCES
            assert entry["attempt"] == 0, "nothing was dispatched for a skip"
            assert "error" not in entry
        assert not [
            spec for spec, e in models.items() if e["state"] in _LIVE_STATES
        ]
        # Both counters, because a skipped slot finished AND produced a usable
        # result. Reading it as anything but `2/2, 2 ok` is D4a.
        assert progress["models_done"] == len(self.MODELS)
        assert progress["models_succeeded"] == len(self.MODELS)
        assert progress["models_total"] == len(self.MODELS)

    def test_the_rendered_line_names_no_live_row(self, tmp_path):
        """The map a resumed run leaves behind renders as a finished fan-out.

        Rebuilt through the tracker's own renderer rather than re-derived here,
        so the assertion is about the shape `run_status` and the `[PROGRESS]`
        line both read — the one place `entry_elapsed_s`'s read-time rule is
        applied.
        """
        self._run(tmp_path)
        phase_dir = self._run(tmp_path, force=True)
        models = status_file.read_status(str(phase_dir))["progress"]["models"]

        tracker = progress_tracker.FanoutProgress(
            str(phase_dir), list(self.MODELS), interval_s=0
        )
        tracker._models = {spec: dict(e) for spec, e in models.items()}
        line = tracker.render_line()
        assert line == "[PROGRESS] 2/2 done (2 ok, 0 failed)", line
        assert not any(state in line for state in _LIVE_STATES)


class TestASlotWaitingOnTheSemaphoreReadsQueued:
    """2026-09-21: the `max_parallel` wait rendered as `staggering`.

    `staggering` fires *before* both semaphores are acquired and `running`
    only inside the per-model coroutine, so a slot whose stagger sleep had
    already elapsed and which was parked on `max_parallel` kept reading
    `staggering` for as long as it waited — with five models at
    `max_parallel: 2` the ticker and the `models:` table showed three models
    "staggering" for minutes, a state every one of them had left. Index 0,
    which never staggers, rendered the same wait as `queued`, so one wait had
    two words. The slot now returns to `queued` the moment the sleep ends.
    """

    MODELS = ["cursor-agent:auto", "codex:gpt-5.5"]

    def test_the_parked_slot_reads_queued_and_not_staggering(self, tmp_path):
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()
        tracker = progress_tracker.FanoutProgress(
            str(phase_dir), self.MODELS, timeout=30, interval_s=0
        )
        head, tail = self.MODELS
        dispatched = asyncio.Event()
        release = asyncio.Event()
        seen = {}

        async def factory(model_spec, index):
            if model_spec == head:
                tracker.running(model_spec, attempt=1)
                dispatched.set()
                await release.wait()
            return {"success": True, "source": "file"}

        async def drive():
            task = asyncio.ensure_future(base.run_models_concurrent(
                list(self.MODELS), factory, 30, 1, tracker=tracker
            ))
            # `max_parallel=1`, so while the head slot holds the semaphore the
            # tail slot can only be waiting for it.
            await dispatched.wait()
            deadline = time.monotonic() + 2.0
            state = tracker.snapshot()[tail]["state"]
            while state == "staggering" and time.monotonic() < deadline:
                await asyncio.sleep(0.005)
                state = tracker.snapshot()[tail]["state"]
            seen["tail"] = state
            seen["head"] = tracker.snapshot()[head]["state"]
            release.set()
            await task

        with patch.object(base, "PROVIDER_STAGGER_DELAY", 0.0):
            asyncio.run(drive())

        assert seen["head"] == "running", seen
        assert seen["tail"] == "queued", (
            "a slot parked on `max_parallel` must not advertise a stagger it "
            f"has already spent: {seen}"
        )
        # And the ending is unchanged: both slots still retire.
        assert not [
            spec for spec, e in tracker.snapshot().items()
            if e["state"] in _LIVE_STATES
        ]


class TestTheStaggeringRowCarriesItsCountdown:
    """2026-09-21: the fan-out names the sleep it takes, so the row can.

    `run_models_concurrent` is the one caller that knows the stagger delay —
    `index * PROVIDER_STAGGER_DELAY`, through `get_backoff_delay` — and the
    tracker cannot infer it. The slot's `staggering` transition is therefore
    made with that delay, and the map, read at that instant, carries the
    `stagger_until` the `status` table renders as `starts in Ns`.
    """

    MODELS = ["cursor-agent:auto", "codex:gpt-5.5", "pi:auto"]

    def test_every_staggered_slot_is_stamped_with_its_own_delay(self, tmp_path):
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()
        tracker = progress_tracker.FanoutProgress(
            str(phase_dir), self.MODELS, timeout=30, interval_s=0
        )
        seen = {}
        real = tracker.staggering

        def spy(spec, **kwargs):
            real(spec, **kwargs)
            # The row as a poll landing on this write would read it.
            seen[spec] = (kwargs, dict(tracker.snapshot()[spec]))

        async def factory(model_spec, index):
            tracker.running(model_spec, attempt=1)
            return {"success": True, "source": "file"}

        with patch.object(base, "PROVIDER_STAGGER_DELAY", 0.02), \
                patch.object(tracker, "staggering", spy):
            asyncio.run(base.run_models_concurrent(
                list(self.MODELS), factory, 30, 3, tracker=tracker
            ))

        head, *tail = self.MODELS
        assert head not in seen, "index 0 never staggers"
        assert list(seen) == tail
        for index, spec in enumerate(tail, start=1):
            kwargs, row = seen[spec]
            expected = base.get_backoff_delay(index * 0.02)
            assert kwargs == {"delay_s": expected}, (spec, kwargs)
            assert row["state"] == "staggering"
            assert "stagger_until" in row, (spec, row)
        # And once dispatched, no row still carries a countdown.
        assert not [
            spec for spec, e in tracker.snapshot().items() if "stagger_until" in e
        ]


class TestEveryEndingRetiresEverySlot:
    """After `run_models_concurrent` returns, no entry reads `queued`,
    `staggering`, `running` or `retrying` — whichever way the fan-out ended.

    A slot cancelled by the overall deadline *does* reach its own `finally`
    (`wait_for` awaits the gather's cancellation before it raises), but it
    reaches no *transition* site: the `CancelledError` lands on the await it was
    parked on, and the recording calls sit below that. A task that raised is
    printed as `Task exception:` and dropped from `completed` without ever being
    keyed into `results`. So the map cannot be left to the slots to close out;
    the sweep in the fan-out's own `finally` walks the map instead of the tasks.

    Each ending also has to carry the wording its path owns — `DEADLINE_ERROR`
    for the overall deadline (the same string `backfill_deadline_losses` writes
    into `failed_models`, so the table and the report's failure list name one
    cause), `CANCELLED_ERROR` for an ending that runs no transition site at
    all, and a raising slot's own message for itself. A third phrasing invented
    at a transition site would split one cause across two labels.

    `--abort` is deliberately NOT in that list (F1, 2026-09-20): it is SIGTERM
    to the worker, whose handler calls `os._exit` after `finish()`, so no
    `finally` in this process runs and no in-memory sweep can reach the map.
    That ending is retired on disk by `progress_tracker.retire_live_entries`
    and is covered by `test_detach.py::TestAbortRetiresTheMapOnDisk`. What the
    cancellation rows below really drive is asyncio shutdown's cancel-all.
    """

    MODELS = ["cursor-agent:auto", "codex:gpt-5.5"]

    def _tracker(self, tmp_path):
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()
        return progress_tracker.FanoutProgress(
            str(phase_dir), self.MODELS, timeout=0.05, interval_s=0
        )

    @staticmethod
    def _assert_agrees_with_the_report(tracker, results):
        """The map's `failed` set is the failure list F11 produced for this run.

        One pass writes both facts, so a reader polling afterwards can never see
        a slot the report calls failed while the table still calls it `running`.
        """
        failed_models: dict = {}
        base.backfill_deadline_losses(
            list(TestEveryEndingRetiresEverySlot.MODELS),
            results,
            failed_models,
            tracker=tracker,
        )
        snapshot = tracker.snapshot()
        assert not [s for s, e in snapshot.items() if e["state"] in _LIVE_STATES]
        assert set(failed_models) == {
            s for s, e in snapshot.items() if e["state"] == "failed"
        }
        return snapshot

    def test_the_overall_deadline_retires_the_stragglers(self, tmp_path):
        tracker = self._tracker(tmp_path)

        async def straggle(model_spec, index):
            await asyncio.sleep(30)
            return {"success": True, "source": "file"}

        with patch.object(base, "PROVIDER_STAGGER_DELAY", 0.0):
            results = asyncio.run(base.run_models_concurrent(
                list(self.MODELS), straggle, 0.05, 4, tracker=tracker
            ))

        assert results == {}
        snapshot = self._assert_agrees_with_the_report(tracker, results)
        for spec, entry in snapshot.items():
            assert entry["state"] == "failed", (spec, entry)
            assert entry["error"] == progress_tracker.DEADLINE_ERROR, entry
            # `source` is meaningful only after a completed attempt.
            assert entry["source"] is None
            assert entry["ended_at"] and "elapsed_s" in entry

    def test_the_deadline_keeps_the_results_of_the_slots_that_finished(
        self, tmp_path
    ):
        """F2, 2026-09-20: the mixed ending — one slot answered, one straggled.

        `asyncio.wait_for` raises instead of assigning `completed`, so the
        deadline branch used to return `results == {}` **even for the model that
        had already returned successfully**. Everything downstream then read one
        run two ways: the map said `{fast: done, slow: failed/DEADLINE}` and
        `counts()` `(2, 1, 2)`, while `backfill_deadline_losses` — walking an
        empty `results` — named *both* models in `failed_models` and the
        restatement wrote `models_succeeded = 0`. The checkpoint contradicted
        its own map, and the report listed a model whose `{model}.json` was
        already on disk for the next `--reaggregate` to pick up. The two
        siblings below only ever exercise "all straggle" and "one raises";
        neither can see this.
        """
        tracker = self._tracker(tmp_path)

        async def one_answers_one_straggles(model_spec, index):
            if model_spec == self.MODELS[1]:
                await asyncio.sleep(30)
            return {"success": True, "source": "file"}

        with patch.object(base, "PROVIDER_STAGGER_DELAY", 0.0):
            results = asyncio.run(base.run_models_concurrent(
                list(self.MODELS), one_answers_one_straggles, 0.05, 4,
                tracker=tracker,
            ))

        # The finished slot's result survives the deadline that cancelled its
        # sibling — it is what `save_model_result` writes and what the report
        # counts.
        assert list(results) == [self.MODELS[0]]
        assert results[self.MODELS[0]]["success"] is True
        snapshot = self._assert_agrees_with_the_report(tracker, results)
        assert snapshot[self.MODELS[0]]["state"] == "done"
        assert snapshot[self.MODELS[0]]["source"] == "file"
        assert snapshot[self.MODELS[1]]["state"] == "failed"
        assert snapshot[self.MODELS[1]]["error"] == progress_tracker.DEADLINE_ERROR
        # Which is also the restatement's arithmetic: `models_succeeded` is
        # `len(model_specs) - len(failed_models)`, so a report that over-reports
        # the failures under-reports the successes by the same models.
        assert tracker.counts() == (2, 1, 2)

    def test_a_slot_that_raises_retires_itself_and_leaves_its_sibling_alone(
        self, tmp_path
    ):
        tracker = self._tracker(tmp_path)

        async def one_explodes(model_spec, index):
            if model_spec == self.MODELS[1]:
                raise RuntimeError("provider exploded")
            return {"success": True, "source": "file"}

        with patch.object(base, "PROVIDER_STAGGER_DELAY", 0.0):
            results = asyncio.run(base.run_models_concurrent(
                list(self.MODELS), one_explodes, 60, 4, tracker=tracker
            ))

        # The raising slot never reaches `results` — that is the F11 gap.
        assert list(results) == [self.MODELS[0]]
        snapshot = self._assert_agrees_with_the_report(tracker, results)
        assert snapshot[self.MODELS[0]]["state"] == "done"
        loser = snapshot[self.MODELS[1]]
        assert loser["state"] == "failed"
        # Its own message, not a sweep wording: the exception IS this slot's
        # ending, and the map keeps the more informative reason even though the
        # report's failure list (which has no exception in hand) says
        # `DEADLINE_ERROR` for the same spec.
        assert loser["error"] == "provider exploded"
        assert loser["source"] == "exception"

    def test_the_cancel_all_path_retires_the_rest_as_cancelled(self, tmp_path):
        """Renamed from `test_an_abort_retires_the_rest_as_cancelled`
        (F1, 2026-09-20): what this drives is a cancellation of the fan-out
        task — asyncio shutdown's cancel-all — and never `--abort`, which calls
        `os._exit` and runs no `finally` in this process. The real `--abort`
        ending is pinned by `test_detach.py::TestAbortRetiresTheMapOnDisk`.
        """
        tracker = self._tracker(tmp_path)

        async def straggle(model_spec, index):
            await asyncio.sleep(30)
            return {"success": True, "source": "file"}

        async def drive():
            task = asyncio.ensure_future(base.run_models_concurrent(
                list(self.MODELS), straggle, 60, 4, tracker=tracker
            ))
            await asyncio.sleep(0.02)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        with patch.object(base, "PROVIDER_STAGGER_DELAY", 0.0):
            asyncio.run(drive())

        snapshot = self._assert_agrees_with_the_report(tracker, {})
        for spec, entry in snapshot.items():
            assert entry["state"] == "failed", (spec, entry)
            # The other of exactly two cancellation wordings: no deadline fired,
            # so `set_sweep_reason` was never called and the default stands.
            assert entry["error"] == progress_tracker.CANCELLED_ERROR, entry


class TestAnInProcessSalvageMovesTheMapWithTheReport:
    """F3, 2026-09-20: under ``--exec-subagents`` the fan-out repairs a model's
    result file and then ``del failed_models[spec]`` — so the checkpoint it
    writes next lists that model under ``models_completed`` and counts it in
    ``models_succeeded``.  The per-model map was left exactly as the fan-out
    froze it: ``failed``, with the pre-salvage error still on the row.

    The obvious repair does not work and is the reason this needed its own
    writer: terminal transitions are **idempotent**
    (``FanoutProgress._freeze`` refuses a second one), so a ``tracker.done()``
    against an already-``failed`` entry is a silent no-op.  The flip goes
    through ``record_models_salvaged`` → ``status_file.update_progress``,
    exactly as ``record_models_present`` does it for the ``--reaggregate`` path.
    """

    MODELS = ["cursor-agent:auto", "codex:gpt-5.5"]

    SUGGESTION = {
        "title": "Add input validation",
        "desc": "The plan does not validate the new email field.",
        "importance": "HIGH", "reference": "Overview", "type": "addition",
    }

    def _record(self, model_spec, *, success, error=None):
        return {
            "success": success,
            "output": json.dumps([self.SUGGESTION]) if success else None,
            "error": error,
            "prompt": "", "stdout": "", "stderr": "",
            "duration_seconds": 0.0,
            "model_spec": model_spec,
            "source": "file" if success else "failed",
        }

    def test_a_salvaged_model_is_done_on_the_map_not_only_in_the_counters(
        self, tmp_path
    ):
        plan = _plan(tmp_path)
        phase_dir = base.get_phase_dir(plan, "review-plan")
        good, bad = self.MODELS

        async def fake_run_all_models(*_args, **kwargs):
            # A real fan-out's two endings, taken through the real tracker so
            # the map this test starts from is the one production leaves behind.
            tracker = kwargs["tracker"]
            tracker.seed()
            results = {
                good: self._record(good, success=True),
                bad: self._record(bad, success=False, error="unparseable reply"),
            }
            for spec, result in results.items():
                tracker.record_result(spec, result)
            return results

        def fake_salvage(_args, *, phase_dir, phase, plan_path=None):
            recovered = Path(phase_dir) / f"{base.sanitize_model_name(bad)}.json"
            recovered.write_text(
                json.dumps([self.SUGGESTION]), encoding="utf-8"
            )
            return [str(recovered)]

        argv = [
            "--plan-file", str(plan), "--models", *self.MODELS,
            "--skip-validation", "--timeout", "120", "--max-parallel", "2",
            "--exec-subagents",
        ]
        with patch.object(base, "run_all_models", new=fake_run_all_models), \
             patch.object(base, "dispatch_salvage_tasks", new=fake_salvage), \
             patch.object(base, "PROVIDER_STAGGER_DELAY", 0.0):
            assert asyncio.run(rpo.main(argv)) == 0

        status = status_file.read_status(str(phase_dir))
        # The report's side of the fact, which already worked.
        assert status["models_failed"] == {}
        assert sorted(status["models_completed"]) == sorted(self.MODELS)
        progress = status["progress"]
        assert progress["models_succeeded"] == 2
        assert progress["models_done"] == 2

        # The map's side, which did not: a `failed` row carrying the salvaged
        # model's pre-salvage error, one object away from `models_succeeded: 2`.
        entry = progress["models"][bad]
        assert entry["state"] == "done", entry
        # `source` is how a renderer knows where this verdict came from, and it
        # is deliberately NOT `record_models_present`'s `existing_file`
        # (2026-09-21): that word is in `run_status.SKIPPED_SOURCES`, so it
        # would have this row render **skipped** — "never dispatched" — about a
        # slot that dispatched, ran, failed to parse and was repaired.
        assert entry["source"] == "salvage", entry
        assert entry["source"] not in progress_tracker.RESUME_SOURCES
        assert "error" not in entry, entry
        # The slot really did end when it ended: a later reading of its outcome
        # is not a second run of it, so the freeze stands.
        assert entry["ended_at"] and "elapsed_s" in entry
        assert progress["models"][good]["state"] == "done"
        assert not [
            spec for spec, e in progress["models"].items()
            if e["state"] in _LIVE_STATES
        ]


# ---------------------------------------------------------------------------
# Plan review-run-scope-salvage-and-failure-visibility §5 (T5.1, T5.2): the
# base reaggregate names the models that produced nothing
# ---------------------------------------------------------------------------
#
# The base reaggregate hard-coded `failed_models_map = {}`, so a review-plan or
# review-tasks report rebuilt by `--reaggregate` never mentioned a reviewer
# that timed out. These drive `review_plan_orchestrator.main(--reaggregate)`
# over a phase dir shaped the way a fan-out leaves one.

_GOOD = "cursor-agent:auto"
_LOST = "pi:openrouter/deepseek/deepseek-v4.1-flash"
_LOST_REASON = "pi timed out after 1200s (spent $0.09 est., 28,193 output tokens)"
_LOST_STEM = base.sanitize_model_name(_LOST)
_STARTED_AT = "2026-09-25T15:58:00"


def _write_result(phase_dir: Path, spec: str) -> None:
    (phase_dir / f"{base.sanitize_model_name(spec)}.json").write_text(
        json.dumps(_ok_reply(spec)["data"]), encoding="utf-8"
    )


def _phase_left_by_a_fanout(tmp_path: Path, **fields):
    plan = _plan(tmp_path)
    phase_dir = Path(base.get_phase_dir(plan, "review-plan"))
    phase_dir.mkdir(parents=True, exist_ok=True)
    _write_result(phase_dir, _GOOD)
    status_file.write_status(str(phase_dir), {
        "phase": "review-plan",
        "state": "complete",
        "started_at": _STARTED_AT,
        "models_requested": [_GOOD, _LOST],
        "models_completed": [_GOOD],
        "models_failed": {_LOST: _LOST_REASON},
        **fields,
    })
    return plan, phase_dir


def _reaggregate_review_plan(plan: Path) -> int:
    return asyncio.run(rpo.main([
        "--plan-file", str(plan), "--skip-validation", "--timeout", "120",
        "--reaggregate",
    ]))


def _line_after_models(phase_dir: Path) -> str:
    lines = (phase_dir / "report.md").read_text(encoding="utf-8").splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("**Models:**"))
    return lines[index + 1]


class TestTheBaseReaggregateNamesTheModelsThatProducedNothing:
    def test_a_recorded_failure_with_no_result_file_is_named(self, tmp_path, capsys):
        plan, phase_dir = _phase_left_by_a_fanout(tmp_path)

        assert _reaggregate_review_plan(plan) == 0
        out = capsys.readouterr().out

        assert _line_after_models(phase_dir) == (
            f"**Models failed:** 1 of 2 requested — {_LOST}: {_LOST_REASON}"
        )
        report = (phase_dir / "report.md").read_text(encoding="utf-8")
        assert f"- **{_LOST}**: {_LOST_REASON}" in report
        assert f"[MODEL_FAILED] {_LOST}: {_LOST_REASON}" in out
        assert f"[PARTIAL_RESULTS: {_LOST}]" in out
        data = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
        assert data["failedModels"] == {_LOST: _LOST_REASON}
        assert data["requestedModels"] == 2
        # Still a finished phase.
        assert base.get_or_create_state(plan).is_phase_completed("review-plan")
        assert status_file.read_status(str(phase_dir))["state"] == "complete"

    def test_a_model_whose_file_appeared_is_not_reported(self, tmp_path, capsys):
        plan, phase_dir = _phase_left_by_a_fanout(tmp_path)
        _write_result(phase_dir, _LOST)

        assert _reaggregate_review_plan(plan) == 0
        out = capsys.readouterr().out

        assert f"[{_LOST}] result file present; recorded as completed" in out
        assert "[MODEL_FAILED]" not in out
        assert "[PARTIAL_RESULTS:" not in out
        report = (phase_dir / "report.md").read_text(encoding="utf-8")
        assert "**Models failed:**" not in report
        assert "## Models Failed" not in report

    def test_the_denominator_counts_a_model_with_neither_file_nor_failure(
        self, tmp_path, capsys
    ):
        five = [f"cursor-agent:model-{letter}" for letter in "abcde"]
        plan, phase_dir = _phase_left_by_a_fanout(
            tmp_path,
            models_requested=five,
            models_completed=five[:3],
            models_failed={five[3]: "exited with code 1"},
        )
        (phase_dir / f"{base.sanitize_model_name(_GOOD)}.json").unlink()
        for spec in five[:3]:
            _write_result(phase_dir, spec)

        assert _reaggregate_review_plan(plan) == 0
        capsys.readouterr()

        assert _line_after_models(phase_dir) == (
            f"**Models failed:** 2 of 5 requested — {five[3]}: exited with code 1; "
            f"{five[4]}: {base.UNACCOUNTED_MODEL_REASON}"
        )

    def test_the_native_lanes_live_request_carries_the_salvage_clause(
        self, tmp_path, capsys
    ):
        """The native invoker deletes `salvage_{stem}.json` only after the
        reaggregate, so a salvage that recovered nothing is still on disk."""
        plan, phase_dir = _phase_left_by_a_fanout(tmp_path)
        (phase_dir / f"salvage_{_LOST_STEM}.json").write_text(
            json.dumps({"model": _LOST, "timestamp": "2026-09-25T16:20:00"}),
            encoding="utf-8",
        )

        assert _reaggregate_review_plan(plan) == 0
        capsys.readouterr()

        assert _line_after_models(phase_dir).endswith(
            f"{_LOST}: {_LOST_REASON}; salvage requested, nothing recovered"
        )

    def _line_with_a_retired_request(self, tmp_path, reason, timestamp):
        """Under `--exec-subagents` an unrecoverable salvage leaves only
        `retired_salvage_{stem}.unrecoverable.*.json`; it counts only when it is
        this attempt's (not earlier than `started_at`), and a `.superseded.`
        pair never does."""
        plan, phase_dir = _phase_left_by_a_fanout(tmp_path)
        (phase_dir / f"retired_salvage_{_LOST_STEM}.{reason}.20260925-162000.json").write_text(
            json.dumps({"model": _LOST, "timestamp": timestamp}), encoding="utf-8"
        )
        assert _reaggregate_review_plan(plan) == 0
        line = _line_after_models(phase_dir)
        assert line.startswith(f"**Models failed:** 1 of 2 requested — {_LOST}")
        return line

    def test_the_exec_subagents_lanes_unrecoverable_request_carries_it(
        self, tmp_path, capsys
    ):
        line = self._line_with_a_retired_request(
            tmp_path, "unrecoverable", "2026-09-25T16:20:00"
        )
        assert line.endswith("; salvage requested, nothing recovered")

    def test_a_previous_runs_unrecoverable_request_does_not(self, tmp_path, capsys):
        line = self._line_with_a_retired_request(
            tmp_path, "unrecoverable", "2026-09-24T09:00:00"
        )
        assert "salvage requested" not in line

    def test_a_superseded_pair_does_not(self, tmp_path, capsys):
        line = self._line_with_a_retired_request(
            tmp_path, "superseded", "2026-09-25T16:20:00"
        )
        assert "salvage requested" not in line


class TestThePhaseTickerOutlivesTheFanOut:
    """T5: the fan-out is not the phase.

    `run_models_concurrent` returns the moment the last slot does, and the phase
    then keeps working for minutes — grouping, validation-batch preparation,
    and on the lanes that validate in-process the whole dispatch and merge. Once
    Layer 1 lands that window is the *only* remaining wait, and it is where
    Claude Code goes silent again, at the END of the run instead of throughout
    it. Lifetime rule 1 does not bend for it: the FAN-OUT's ticker still dies
    before the per-model summary, which is why this second one starts after it.
    """

    @staticmethod
    def _progress_lines(out: str):
        return [ln for ln in out.splitlines() if ln.startswith("[PROGRESS]")]

    def test_lines_keep_arriving_through_the_exec_subagents_dispatch(
        self, tmp_path, capsys
    ):
        """F6, 2026-09-20. The window this ticker exists for is the
        `--exec-subagents` validation dispatch, and the ticker was starved for
        every second of it.

        `dispatch_subagent_tasks` is **synchronous** and ends in
        `subagent_executor._run_sync`'s `pool.submit(asyncio.run, …).result()`,
        which blocks the calling thread — the thread the event loop runs on —
        until the whole batch is done. A ticker is a task on that loop, so the
        executor lane got exactly one `[PROGRESS]` line for the window, the
        teardown one, beneath a comment claiming the second ticker covered
        "the whole validation dispatch and merge". The fix is to await that call
        off the loop (`asyncio.to_thread`); this test pins it at the one place
        it can be observed, by asking the blocking stub itself how many lines
        landed while it was blocking.

        Its predecessor drove the `--internal-validation` path with an `async`
        stub that awaited, so `len(after) >= 2` was a statement about the stub
        rather than about the production dispatch path, and it passed both
        before and after the defect.
        """
        plan = _plan(tmp_path)
        argv = [
            "--plan-file", str(plan), "--models", "cursor-agent:auto",
            "--timeout", "120", "--exec-subagents",
        ]
        phase_dir = base.get_phase_dir(plan, "review-plan")
        maps = {}
        window = {}
        real = base._detach.report_progress

        def record(pd, **updates):
            real(pd, **updates)
            if updates.get("step") == "grouping":
                maps["after_fanout"] = status_file.read_status(
                    str(phase_dir)
                )["progress"]["models"]
            elif "models" not in updates and "after_fanout" in maps:
                # Every write in the phase window: a counters-only or `step=`
                # write must not carry the map, or it would replace it wholesale
                # (`progress` merges one level deep only).
                maps.setdefault("phase_writes", []).append(dict(updates))

        def phase_lines():
            return [ln for ln in emitted if ln.startswith("[PROGRESS] step: ")]

        def blocking_dispatch(_args, **kwargs):
            """`continue_past_marker`'s shape: it blocks, it does not await."""
            # Stand in for `subagent_executor._mirror_status`, which writes these
            # counts straight through `update_status` on every outcome. The
            # phase ticker READS them; it never recomputes them into a second key.
            status_file.update_status(str(phase_dir), {
                "progress": {"subagents": {
                    "kind": "validation", "total": 8, "ok": 5,
                    "counts": {"ok": 5},
                }},
            })
            window["before"] = len(phase_lines())
            deadline = time.monotonic() + 0.2
            while time.monotonic() < deadline:
                time.sleep(0.005)
            window["after"] = len(phase_lines())
            return True

        def merged(_phase_dir, _metadata, grouped, _validation_path):
            return [
                (g.to_dict() if hasattr(g, "to_dict") else dict(g))
                for g in grouped
            ]

        with _realtime_ticker() as emitted, patch.object(
            base, "invoke_with_file_output",
            side_effect=lambda **kwargs: _ok_reply(kwargs["model_spec"]),
        ), patch.object(base, "PROVIDER_STAGGER_DELAY", 0.0), patch.object(
            base, "dispatch_subagent_tasks", new=blocking_dispatch
        ), patch.object(
            base, "merge_validation_batches", new=merged
        ), patch.object(base._detach, "report_progress", record):
            assert asyncio.run(rpo.main(argv)) == 0

        # The assertion the defect fails: the dispatch really was reached, and
        # the ticker really did print while it was blocking. The stub blocks for
        # 0.2 s against a 10 ms interval, so the honest floor is many lines and
        # the defect's floor is zero — asserting two leaves the margin a loaded
        # machine needs without weakening the discrimination.
        assert window, "the --exec-subagents dispatch was never reached"
        assert window["after"] - window["before"] >= 2, (
            f"the ticker was starved for the whole dispatch window: "
            f"{window} in {emitted}"
        )
        # Those lines carry the executor mirror's own counts, read off
        # `progress.subagents` rather than recomputed into a second key.
        lines = phase_lines()
        assert any("subagents 5/8 done (5 ok)" in ln for ln in lines), lines
        # Lifetime rule 1 still holds at the other end: the last line lands at
        # the terminal step, after the per-model summary, with nothing after it.
        out = capsys.readouterr().out
        summary = max(
            i for i, ln in enumerate(out.splitlines())
            if "Saved 1 suggestions to:" in ln
        )
        after = [
            ln for i, ln in enumerate(out.splitlines())
            if i > summary and ln.startswith("[PROGRESS]")
        ]
        assert after, out
        assert lines[-1].startswith("[PROGRESS] step: complete"), lines
        # The fan-out's own ticker died before the summary, and the phase
        # ticker's writes never touch `models` — so the terminal map is what the
        # sweep left, byte for byte.
        assert all("models" not in w for w in maps.get("phase_writes", []))
        assert status_file.read_status(str(phase_dir))["progress"]["models"] == \
            maps["after_fanout"]

    def test_lines_keep_arriving_through_the_in_process_salvage(
        self, tmp_path
    ):
        """T5, 2026-09-21: the salvage window had no ticker at all.

        In-process salvage (`--exec-subagents`, a model whose output would not
        parse) runs *after* `run_all_models` returns — so the fan-out's ticker
        has already been cancelled and awaited by lifetime rule 1 — and
        *before* the phase ticker starts at grouping. `dispatch_salvage_tasks`
        blocks the same way `dispatch_subagent_tasks` does (one
        `subagent_executor.run_task_file` per request), so a salvage that took
        minutes emitted no `[PROGRESS]` line at all while `progress.step` still
        read `fanout`: exactly the silent gap between the last model and the
        report T5 forbids. Same cure, pinned the same way as the validation
        window — the blocking stub itself reports how many lines landed while
        it was blocking, which is a statement about the production call site
        and not about the stub.
        """
        plan = _plan(tmp_path)
        phase_dir = base.get_phase_dir(plan, "review-plan")
        good, bad = "cursor-agent:auto", "codex:gpt-5.5"
        suggestion = {
            "title": "Add input validation",
            "desc": "The plan does not validate the new email field.",
            "importance": "HIGH", "reference": "Overview", "type": "addition",
        }

        def _record(model_spec, *, success, error=None):
            return {
                "success": success,
                "output": json.dumps([suggestion]) if success else None,
                "error": error,
                "prompt": "", "stdout": "", "stderr": "",
                "duration_seconds": 0.0,
                "model_spec": model_spec,
                "source": "file" if success else "failed",
            }

        async def fake_run_all_models(*_args, **kwargs):
            tracker = kwargs["tracker"]
            tracker.seed()
            results = {
                good: _record(good, success=True),
                bad: _record(bad, success=False, error="unparseable reply"),
            }
            for spec, result in results.items():
                tracker.record_result(spec, result)
            return results

        def phase_lines():
            return [ln for ln in emitted if ln.startswith("[PROGRESS] step: ")]

        window = {}

        def blocking_salvage(_args, *, phase_dir, phase, plan_path=None):
            """`dispatch_salvage_tasks`' shape: it blocks, it does not await."""
            recovered = Path(phase_dir) / f"{base.sanitize_model_name(bad)}.json"
            recovered.write_text(json.dumps([suggestion]), encoding="utf-8")
            # Read here, not afterwards: the point of the step is that a poll
            # taken DURING the window names the work, rather than still
            # reading the step the fan-out left behind.
            window["step"] = status_file.read_status(phase_dir)["progress"]["step"]
            window["before"] = len(phase_lines())
            deadline = time.monotonic() + 0.2
            while time.monotonic() < deadline:
                time.sleep(0.005)
            window["after"] = len(phase_lines())
            return [str(recovered)]

        argv = [
            "--plan-file", str(plan), "--models", good, bad,
            "--skip-validation", "--timeout", "120", "--max-parallel", "2",
            "--exec-subagents",
        ]
        with _realtime_ticker() as emitted, patch.object(
            base, "run_all_models", new=fake_run_all_models
        ), patch.object(
            base, "dispatch_salvage_tasks", new=blocking_salvage
        ), patch.object(base, "PROVIDER_STAGGER_DELAY", 0.0):
            assert asyncio.run(rpo.main(argv)) == 0

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
        assert sorted(status["models_completed"]) == sorted([good, bad])
        assert status["progress"]["models"][bad]["state"] == "done"

    def test_the_validation_pending_early_return_is_a_terminal_step_too(
        self, tmp_path, capsys
    ):
        """The second terminal step, and stopping there is not a gap: on Claude
        Code the host-driven return IS the background job's completion signal,
        and the validation that follows is the agent's own subagents rather than
        the orchestrator's work to report."""
        plan = _plan(tmp_path)
        argv = [
            "--plan-file", str(plan), "--models", "cursor-agent:auto",
            "--timeout", "120",
        ]
        with _fast_ticker() as emitted, patch.object(
            base, "invoke_with_file_output",
            side_effect=lambda **kwargs: _ok_reply(kwargs["model_spec"]),
        ), patch.object(base, "PROVIDER_STAGGER_DELAY", 0.0):
            assert asyncio.run(rpo.main(argv)) == 0

        phase_dir = base.get_phase_dir(plan, "review-plan")
        status = status_file.read_status(str(phase_dir))
        assert status["state"] == "validation_pending"
        assert status["progress"]["step"] == "validation_pending"
        # This window is entirely SYNCHRONOUS — batch preparation is file I/O
        # and there is no `await` between `start_ticker()` and the `finally` —
        # so it is the path that proves the one yield after `start_ticker()` is
        # load-bearing: `ensure_future` only schedules the task, and a task
        # cancelled before its first step never runs its body, teardown line
        # included. Asserted at the `_emit` seam, because a checkpoint
        # assertion cannot tell "printed nothing" from "printed correctly".
        assert [ln for ln in emitted if ln.startswith("[PROGRESS] step: ")], emitted
        lines = self._progress_lines(capsys.readouterr().out)
        assert lines[-1].startswith("[PROGRESS] step: validation_pending"), lines

    def test_the_reaggregate_path_gets_one_too(self, tmp_path, capsys):
        """The lane with no fan-out at all.

        A `--reaggregate` invocation's whole work — regrouping, and on a phase
        that needs it a full revalidation pass — is the silent window, with no
        per-model ticker ahead of it to have covered any of it. Its window is
        synchronous under `--skip-validation`, so this is the second path that
        would emit nothing without the yield after `start_ticker()`.
        """
        plan = _plan(tmp_path)
        common = ["--plan-file", str(plan), "--skip-validation", "--timeout", "120"]
        with patch.object(
            base, "invoke_with_file_output",
            side_effect=lambda **kwargs: _ok_reply(kwargs["model_spec"]),
        ), patch.object(base, "PROVIDER_STAGGER_DELAY", 0.0):
            assert asyncio.run(rpo.main(
                common + ["--models", "cursor-agent:auto"]
            )) == 0
        phase_dir = base.get_phase_dir(plan, "review-plan")
        before = status_file.read_status(str(phase_dir))["progress"]["models"]
        capsys.readouterr()

        with _fast_ticker() as emitted:
            assert asyncio.run(rpo.main(common + ["--reaggregate"])) == 0

        phase_lines = [ln for ln in emitted if ln.startswith("[PROGRESS] step: ")]
        assert phase_lines, emitted
        lines = self._progress_lines(capsys.readouterr().out)
        assert lines[-1].startswith("[PROGRESS] step: complete"), lines
        # It prints and does not write, so the map the original run's sweep
        # froze is exactly what it was — a table rendered after a reaggregate
        # still shows each model's real duration.
        assert status_file.read_status(str(phase_dir))["progress"]["models"] == before


# ---------------------------------------------------------------------------
# The recorded prompt carries the working-limits minutes the attempt really got
# ---------------------------------------------------------------------------

class TestTheRecordedPromptNamesTheAttemptsMinutes:
    """`run_single_model` records the prompt it sent by re-rendering the
    template, so the `{working_limits}` minutes there must come from the same
    seconds `invoke_with_file_output` rendered them from — otherwise the record
    says one budget and the model was told another (plan:
    review-run-scope-salvage-and-failure-visibility D3.2). Only the provider
    call is replaced: the real template, the real `invoke_with_file_output` and
    the real ladder run.
    """

    def _run(self, tmp_path, model_spec, timeout, sent):
        from utils import llm_client

        plan = _plan(tmp_path)

        def fake_provider(prompt, spec, attempt_timeout, *args, **kwargs):
            sent.append((prompt, attempt_timeout))
            return {
                "success": True,
                "data": _ok_reply(spec)["data"],
                "details": {"provider": spec.partition(":")[0], "model": "auto"},
            }

        with patch.object(llm_client, "invoke_with_provider", fake_provider), \
             patch.object(base, "get_project_root", lambda _p: str(tmp_path)):
            return asyncio.run(base.run_single_model(
                model_spec=model_spec,
                plan_path=str(plan),
                prompt_template=rpo.load_review_prompt_template(),
                prompt_context={"plan_path": str(plan), "prefix": "auth-feature"},
                phase_name="review-plan",
                invoke_phase="plan_review",
                timeout=timeout,
                out_dir=str(tmp_path),
                prefix="auth-feature",
            ))

    def test_the_reconstructed_prompt_equals_the_sent_prompt(self, tmp_path):
        from utils.llm_client import render_working_limits

        sent = []
        result = self._run(tmp_path, "cursor-agent:auto", 1200, sent)

        assert result["success"] is True, result
        assert len(sent) == 1
        prompt, attempt_timeout = sent[0]
        # The first attempt gets the whole slot, rounded to whole seconds.
        assert attempt_timeout == 1200
        assert render_working_limits(1200) in prompt
        assert "stopped after about 20 minutes." in prompt
        assert result["prompt"] == prompt

    def test_a_slot_refused_before_dispatch_records_the_slot_timeouts_minutes(
        self, tmp_path
    ):
        """muse declares a 985 s worst-case wall, so rule 4 refuses a shorter
        slot with nothing spawned, and the placeholder is the recorded prompt.

        719.6 s pins the rounding: the placeholder uses `attempt_timeout`'s
        `round` (720 s, 12 minutes), where a truncated 719 s would read 11.
        """
        from utils.llm_client import render_working_limits

        sent = []
        result = self._run(tmp_path, "muse:muse-spark-1.2", 719.6, sent)

        assert sent == [], "rule 4 is a pre-dispatch admission test"
        assert result["source"] == "not-dispatched"
        assert "budget_exhausted" in str(result["error"])
        assert render_working_limits(720) in result["prompt"]
        assert "stopped after about 12 minutes." in result["prompt"]
