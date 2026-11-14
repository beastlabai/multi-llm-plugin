"""A cascade stage's review records its provenance, and its retries keep it.

``--improve-plan --cascade`` and ``--improve-code --cascade`` run one review
round per model. Each stage's review gets the hidden ``--cascade-stage-file
<manifest>`` (review-plan and review-code only), verifies every hash the
manifest lists before it writes anything, and records the manifest as the
``cascade`` block of its round's ``.status.json`` (``utils/cascade_provenance.py``,
``review_lane.open_round``). The flag is never replayed, so a printed retry
runs without it; a fan-out that resumes the stage's round with exactly its one
model carries the recorded block over, and any other fan-out without the flag
(``--rerun-all``, other models) records none
(docs/development/review-results.md, "Round identity").

Both lanes run here offline through their own ``main``, with only the model
slots stubbed (``tests/test_review_lane_parity.py``'s harness).
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import jsonschema
import pytest

import code_review_orchestrator as cro
import multi_llm
import review_plan_orchestrator as rpo
import review_tasks_orchestrator as rto
from tests.test_review_lane_parity import (
    MODELS, NAMES, CodeLane, PlanLane, _clean_env, _write_batch,
)
from utils import cascade_provenance as cp
from utils import cli_common, detach, review_lane, run_status, status_file
from utils import human_decisions, validation
from utils.output_handler import phase_dir_path
from utils.schema_validator import SCHEMA_DIR

HIDDEN = cp.HIDDEN_FLAG
REPLAYING_MODES = ("review-plan", "review-code")


@pytest.fixture(autouse=True)
def _no_live_invocation():
    cli_common.clear_invocation()
    yield
    cli_common.clear_invocation()


@pytest.fixture(params=[PlanLane, CodeLane], ids=["review-plan", "review-code"])
def lane(request, tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    return request.param(tmp_path, monkeypatch)


def _composite(lane) -> str:
    return "improve-plan" if lane.mode == "review-plan" else "improve-code"


def _point_at(lane, cascade_id: str, *, of: int = 2, stage: int = 1,
              state: str = "running", stage_state: str = "running") -> Path:
    """The composite's ``cascade-current.json`` naming *cascade_id*, and its
    record: *state*, stage *stage* in *stage_state*, the others pending (the
    handoff check reads these; ``utils/improve_cascade.py`` writes them)."""
    run_dir = phase_dir_path(lane.plan, _composite(lane))
    name = f"20261114T000000-{cascade_id[:8]}"
    directory = run_dir / "cascades" / name
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1, "cascade_id": cascade_id, "mode": _composite(lane),
        "state": state,
        "stages": [{"index": i, "model": MODELS[0],
                    "state": stage_state if i == stage else "pending"}
                   for i in range(1, of + 1)],
    }
    (directory / cp.RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")
    pointer = run_dir / cp.POINTER_NAME
    pointer.write_text(json.dumps({"schema_version": 1, "cascade_id": cascade_id,
                                   "dir": f"cascades/{name}"}), encoding="utf-8")
    return pointer


def _manifest(lane, *, stage: int = 1, of: int = 2,
              cascade_id: str = "c0ffee00c0ffee00", current: bool = True) -> Path:
    """A stage's frozen inputs outside the repository, the manifest written last.

    *current*: the composite's pointer names this cascade, running, so the
    review takes the handoff (``cascade_provenance.check_current``).
    """
    root = lane.repo.parent / "cascade" / cascade_id / f"stage-{stage}" / "inputs"
    root.mkdir(parents=True)
    for name in cp.LEDGER_NAMES:
        text = "" if stage == 1 else f"{name}: what stages before {stage} decided\n"
        (root / name).write_text(text, encoding="utf-8")
    if current:
        _point_at(lane, cascade_id, of=of, stage=stage)
    return cp.write_manifest(root, cascade_id=cascade_id, mode=_composite(lane), stage=stage,
                             of=of, model=MODELS[0], names=cp.LEDGER_NAMES)


def _outcomes(lane, *, failed: str = "") -> Dict[str, Any]:
    """Two models' answers; *failed* names a model whose slot fails instead."""
    findings = {MODELS[0]: lane.findings(NAMES[:3]), MODELS[1]: lane.findings(NAMES[3:])}
    if failed:
        findings[failed] = "the provider failed"
    return findings


def _stage_outcome(lane, *, failed: bool = False) -> Dict[str, Any]:
    """The stage's one model (the manifest's, ``MODELS[0]``): its answer, or a failed slot."""
    return {MODELS[0]: "the provider failed" if failed else lane.findings(NAMES[:3])}


#: Two models no stage of the test cascade has.
OTHERS = ["codex:other-a", "codex:other-b"]


def _block(lane) -> Any:
    return lane.status().get(cp.STATUS_KEY)


def _phase_dir_exists(lane) -> bool:
    """Whether the lane's phase dir exists (``lane.phase_dir`` would create it)."""
    return phase_dir_path(lane.plan, lane.phase).exists()


def _state_bytes(lane) -> bytes:
    return Path(lane.state_path).read_bytes()


def _cascade_schema_problems(block: Any) -> List[str]:
    schema = json.loads((SCHEMA_DIR / "status_file.schema.json").read_text("utf-8"))
    validator = jsonschema.Draft7Validator(schema["properties"][cp.STATUS_KEY])
    return [error.message for error in validator.iter_errors(block)]


# ---------------------------------------------------------------------------
# The hidden flag
# ---------------------------------------------------------------------------


class TestTheHiddenFlag:
    @pytest.mark.parametrize("module", [rpo, cro], ids=REPLAYING_MODES)
    def test_review_plan_and_review_code_parse_it(self, module):
        args = module.parse_args(["--plan-file", "p.md", HIDDEN, "/c/stage-1/inputs/manifest.json"])
        assert getattr(args, cp.FLAG_DEST) == "/c/stage-1/inputs/manifest.json"
        assert getattr(module.parse_args(["--plan-file", "p.md"]), cp.FLAG_DEST) is None

    @pytest.mark.parametrize("module", [rpo, cro], ids=REPLAYING_MODES)
    def test_no_help_shows_it(self, module, capsys):
        with pytest.raises(SystemExit) as stop:
            module.parse_args(["--help"])
        assert stop.value.code == 0
        out = capsys.readouterr().out
        assert "--plan-file" in out, "the help was printed"
        assert "cascade-stage" not in out and cp.FLAG_DEST not in out

    @pytest.mark.parametrize("mode", REPLAYING_MODES)
    def test_the_dispatchers_mode_help_does_not_show_it(self, mode, capsys):
        assert multi_llm.main([mode, "--help"]) == 0
        captured = capsys.readouterr()
        assert "cascade-stage" not in captured.out + captured.err

    def test_review_tasks_does_not_take_it(self, capsys):
        with pytest.raises(SystemExit) as stop:
            rto.parse_args(["--plan-file", "p.md", HIDDEN, "/m.json"])
        assert stop.value.code == 2
        assert "unrecognized arguments" in capsys.readouterr().err

    def test_the_dispatcher_reads_it_as_an_input_path(self):
        """A relative manifest takes part in the move to the repository root
        like the plan; a detached worker is handed it absolute."""
        assert HIDDEN in multi_llm._MODE_INPUT_PATH_OPTIONS
        argv = detach.build_worker_argv(
            "review-plan", ["--plan-file", "/p.md", HIDDEN, "rel/manifest.json"], run_id="r1")
        assert argv[argv.index(HIDDEN) + 1] == os.path.abspath("rel/manifest.json")
        argv = detach.build_worker_argv(
            "review-code", ["--plan-file", "/p.md", f"{HIDDEN}=rel/m.json"], run_id="r1")
        assert f"{HIDDEN}={os.path.abspath('rel/m.json')}" in argv


# ---------------------------------------------------------------------------
# Kept out of replay
# ---------------------------------------------------------------------------


class TestKeptOutOfReplay:
    @pytest.mark.parametrize("tokens", [
        [HIDDEN, "/r/plans/feature/improve-plan/cascades/x/stage-1/inputs/manifest.json"],
        [f"{HIDDEN}=/m.json"],
        ["--cascade-stage", "/m.json"],      # an argparse abbreviation
        ["--cascade-st=/m.json"],
    ])
    @pytest.mark.parametrize("mode", REPLAYING_MODES)
    def test_replay_flags_drop_it_with_its_value(self, mode, tokens):
        argv = ["--plan-file", "/p.md", *tokens, "--quick", "--host", "codex"]
        assert cli_common.replay_flags(mode, argv) == ["--quick", "--host", "codex"]

    def test_the_composites_cascade_flag_is_not_its_abbreviation(self):
        """``--cascade`` is a real option of the composites, so neither the
        replay nor the dispatcher's path scan reads it as a short
        ``--cascade-stage-file`` swallowing the next token."""
        argv = ["--plan-file", "/p.md", "--cascade", "--models", "a", "b", "--quick"]
        assert cli_common.replay_flags("improve-plan", argv) == [
            "--cascade", "--models", "a", "b", "--quick"]
        assert cli_common.flag_value(["--cascade", "--models", "a"], HIDDEN) is None
        assert not cli_common.flag_present(["--cascade"], HIDDEN)

    @pytest.mark.parametrize("mode", REPLAYING_MODES)
    def test_status_resume_lines_never_carry_it(self, mode):
        flags = cli_common.replay_flags(
            mode, ["--plan-file", "/p.md", HIDDEN, "/m.json", "--quick", "--detach"])
        assert flags == ["--quick", "--detach"]
        assert "cascade-stage" not in detach.child_env(replay_flags=flags)[detach.REPLAY_FLAGS_ENV]
        view = {"run": {"mode": mode, "replay_flags": flags}, "progress": {}}
        command = run_status._resume_command(view, Path("/tmp/plan.md"))
        assert "--quick" in command and "cascade-stage" not in command

    def test_a_retry_with_line_never_carries_it(self, lane, capsys):
        """A stage whose every model failed names its retry without the flag;
        the block is on record for the retry to carry over."""
        manifest = _manifest(lane)
        flags = ("--models", MODELS[0], "--timeout", "120", "--max-parallel", "2",
                 HIDDEN, str(manifest))
        cli_common.record_invocation(lane.mode, lane.argv(*flags))
        assert lane.fanout({MODELS[0]: "the provider failed"}, HIDDEN, str(manifest)) == 3
        captured = capsys.readouterr()
        retry = [line for line in (captured.out + captured.err).splitlines()
                 if "Retry with:" in line]
        assert retry, captured.err
        assert all("cascade-stage" not in line and "--force" in line for line in retry)
        assert _block(lane) == cp.block_from_manifest(manifest)

    @pytest.mark.parametrize("live", ["composite", "quick", "none"])
    def test_the_retry_with_line_names_the_stage_model_and_run_as_printed_keeps_the_round(
            self, lane, capsys, live):
        """A review the composite runs in its own process has no replay flags
        of its own (the invocation on record is the composite's, or none), and
        a live one may carry ``--quick``: either way the zero-result ``Retry
        with:`` line names the stage's one model, so run as printed (the
        default list here is OTHERS) its fan-out resumes the stage's round and
        keeps the block instead of discarding it."""
        manifest = _manifest(lane, stage=2)
        block = cp.block_from_manifest(manifest)
        if live == "composite":
            composite = "improve-plan" if lane.mode == "review-plan" else "improve-code"
            cli_common.record_invocation(composite, lane.argv("--cascade", "--models", *MODELS))
        elif live == "quick":
            cli_common.record_invocation(lane.mode, lane.argv("--quick", HIDDEN, str(manifest)))
        assert lane.fanout(_stage_outcome(lane, failed=True), HIDDEN, str(manifest)) == 3
        captured = capsys.readouterr()
        retry = [line.split("Retry with: ", 1)[1]
                 for line in (captured.out + captured.err).splitlines() if "Retry with: " in line]
        assert len(retry) == 1, captured.err
        tokens = shlex.split(retry[0])
        assert tokens[-3:] == ["--models", MODELS[0], "--force"], tokens
        assert "--quick" not in tokens and "cascade-stage" not in retry[0]
        flags = tokens[tokens.index(str(lane.plan)) + 1:]
        cli_common.clear_invocation()
        seen: List[List[str]] = []

        def resolve(**kwargs):
            seen.append(list(kwargs.get("cli_models") or OTHERS))
            return seen[-1]

        assert lane.run(*flags, patches=(
            patch("utils.interactive.resolve_models", side_effect=resolve),
            patch("utils.interactive.is_model_valid", return_value=True),
            lane.stub_slots(_stage_outcome(lane), ()),
        )) == 0
        assert seen == [[MODELS[0]]]
        assert _block(lane) == block, "the stage's round, block and all"


# ---------------------------------------------------------------------------
# Recorded in the round, carried across fan-outs
# ---------------------------------------------------------------------------


def _capture_opening(monkeypatch, lane) -> List[Dict[str, Any]]:
    """The checkpoint right after each opening write."""
    seen: List[Dict[str, Any]] = []
    real = review_lane.open_round

    def open_round(*args, **kwargs):
        opening = real(*args, **kwargs)
        seen.append(status_file.read_status(str(lane.phase_dir)))
        return opening

    monkeypatch.setattr(review_lane, "open_round", open_round)
    return seen


class TestTheRoundRecordsTheBlock:
    def test_the_opening_records_it_and_every_later_write_keeps_it(self, lane, monkeypatch):
        manifest = _manifest(lane)
        block = cp.block_from_manifest(manifest)
        openings = _capture_opening(monkeypatch, lane)
        assert lane.fanout(_outcomes(lane), HIDDEN, str(manifest)) == 0
        assert openings[-1][cp.STATUS_KEY] == block
        assert openings[-1]["models_requested"] == MODELS
        assert not _cascade_schema_problems(block)
        # Grouping, the validation round handed out: every write merged.
        assert _block(lane) == block
        tasks = lane.json("validation_tasks.json")
        for batch in tasks["batches"]:
            _write_batch(lane, batch)
        # A --reaggregate continues the round on record, block and all.
        assert lane.reaggregate() == 0
        assert lane.phase_completed()
        assert _block(lane) == block
        assert lane.status()["round_id"] == openings[-1]["round_id"]

    def test_a_force_resume_without_the_flag_stays_the_stages_round(self, lane):
        """The cascade's own retry: a printed retry never carries the flag, and
        a ``--force`` resume of the stage's one model (missing: its slot
        failed) carries the recorded block over to its new round id."""
        manifest = _manifest(lane, stage=2)
        assert lane.fanout(_stage_outcome(lane, failed=True), HIDDEN, str(manifest)) == 3
        first_round = lane.status()["round_id"]
        assert _block(lane) == cp.block_from_manifest(manifest)
        assert lane.fanout(_stage_outcome(lane), "--force") == 0
        assert lane.status()["round_id"] != first_round
        assert _block(lane) == cp.block_from_manifest(manifest)
        assert validation.validator_prior_rounds(str(lane.phase_dir)).startswith(
            f"{cp.VALIDATOR_LEDGER}: what stages before 2 decided")

    def test_a_force_resume_reusing_the_stages_result_stays_its_round(self, lane):
        """A ``--force`` resume of the stage's model whose result is on disk
        keeps that result, so the round is still the stage's."""
        manifest = _manifest(lane, stage=2)
        assert lane.fanout(_stage_outcome(lane), HIDDEN, str(manifest)) == 0
        assert lane.fanout(_stage_outcome(lane), "--force") == 0
        assert _block(lane) == cp.block_from_manifest(manifest)

    def test_a_resume_with_the_flag_stays_the_stages_round(self, lane):
        manifest = _manifest(lane, stage=2)
        assert lane.fanout(_stage_outcome(lane, failed=True), HIDDEN, str(manifest)) == 3
        assert lane.fanout(_stage_outcome(lane), "--force", HIDDEN, str(manifest)) == 0
        assert _block(lane) == cp.block_from_manifest(manifest)

    def test_a_hand_started_round_with_other_models_is_not_the_cascades(self, lane):
        """After a finished cascade (its last stage's round complete), a
        ``--force`` round with two other models and no ``--rerun-all`` keeps
        nothing of the stage's round: it records no block, and its validation
        and judge read no ledger, today's prompts. Removing the cascade dir
        afterwards stops nothing."""
        manifest = _manifest(lane, stage=2)
        assert lane.fanout(_stage_outcome(lane), HIDDEN, str(manifest)) == 0
        for batch in lane.json("validation_tasks.json")["batches"]:
            _write_batch(lane, batch)
        assert lane.reaggregate() == 0
        assert lane.phase_completed() and _block(lane) is not None

        findings = {OTHERS[0]: lane.findings(NAMES[3:5]), OTHERS[1]: lane.findings(NAMES[5:])}
        assert lane.fanout(findings, "--force") == 0
        status = lane.status()
        assert status["models_requested"] == OTHERS
        assert cp.STATUS_KEY not in status
        directory = str(lane.phase_dir)
        assert validation.validator_prior_rounds(directory) == ""
        assert human_decisions.cascade_judge_ledger(directory) is None
        tasks = lane.json("validation_tasks.json")
        assert validation.PRIOR_ROUNDS_INPUT_KEY not in (tasks.get("inputs") or {})
        live = [p for p in lane.phase_dir.iterdir()
                if p.is_file() and "stale" not in p.name and p.suffix in (".json", ".md")]
        assert live and not any("what stages before" in p.read_text("utf-8") for p in live)

        import shutil
        shutil.rmtree(manifest.parent.parent.parent)
        assert validation.validator_prior_rounds(directory) == ""
        assert human_decisions.cascade_judge_ledger(directory) is None

    @pytest.mark.parametrize("models", [[OTHERS[0]], [MODELS[0], OTHERS[0]]],
                             ids=["another-model", "the-stages-model-and-another"])
    def test_a_hand_started_round_in_an_unfinished_cascade_is_foreign(self, lane, models):
        """While a cascade is unfinished, a round started by hand with
        another single model (or a second model beside the stage's) records
        no block, so the cascade's binding reads it as a foreign round."""
        from utils import improve_cascade

        manifest = _manifest(lane, stage=2)
        assert lane.fanout(_stage_outcome(lane), HIDDEN, str(manifest)) == 0
        block = cp.block_from_manifest(manifest)
        mode = "improve-plan" if lane.mode == "review-plan" else "improve-code"
        assert improve_cascade._review_view(lane.plan, mode).belongs_to(
            block["cascade_id"], 2)
        findings = {spec: lane.findings(NAMES[index * 2:index * 2 + 2])
                    for index, spec in enumerate(models)}
        assert lane.fanout(findings, "--force") == 0
        assert cp.STATUS_KEY not in lane.status()
        assert not improve_cascade._review_view(lane.plan, mode).belongs_to(
            block["cascade_id"], 2)

    def test_a_hand_started_rerun_all_is_not_the_cascades(self, lane):
        manifest = _manifest(lane)
        assert lane.fanout(_outcomes(lane), HIDDEN, str(manifest)) == 0
        assert lane.fanout(_outcomes(lane), "--force", "--rerun-all") == 0
        assert cp.STATUS_KEY not in lane.status()
        # And a resume after it carries nothing back.
        assert lane.fanout(_outcomes(lane), "--force") == 0
        assert cp.STATUS_KEY not in lane.status()

    def test_the_next_stage_starts_its_own_round_over_the_last(self, lane):
        first = _manifest(lane, stage=1)
        second = _manifest(lane, stage=2)
        assert lane.fanout(_outcomes(lane), HIDDEN, str(first)) == 0
        assert lane.fanout(_outcomes(lane), "--force", "--rerun-all", HIDDEN, str(second)) == 0
        assert _block(lane) == cp.block_from_manifest(second)
        assert _block(lane)["stage"] == 2

    def test_another_stage_over_kept_results_is_refused(self, lane, capsys):
        first = _manifest(lane, stage=1)
        second = _manifest(lane, stage=2)
        assert lane.fanout(_outcomes(lane), HIDDEN, str(first)) == 0
        before = lane.status()
        state = _state_bytes(lane)
        capsys.readouterr()
        assert lane.fanout(_outcomes(lane), "--force", HIDDEN, str(second)) == 3
        err = capsys.readouterr().err
        assert "ERROR: cascade stage input" in err and "would keep its results" in err
        after = lane.status()
        assert after[cp.STATUS_KEY] == cp.block_from_manifest(first)
        assert after["round_id"] == before["round_id"], "no round was stamped"
        assert _state_bytes(lane) == state

    def test_a_relative_manifest_is_read_from_the_cwd(self, lane):
        manifest = _manifest(lane)
        relative = os.path.relpath(manifest, os.getcwd())
        assert not os.path.isabs(relative)
        assert lane.fanout(_outcomes(lane), HIDDEN, relative) == 0
        assert _block(lane)["manifest"] == str(manifest)

    def test_a_recovered_checkpoint_records_the_flags_block(self, lane):
        """A damaged checkpoint holds no block anyone can read: the fresh run
        that recovers it records the block its own flag names."""
        manifest = _manifest(lane)
        assert lane.fanout(_outcomes(lane), HIDDEN, str(manifest)) == 0
        (lane.phase_dir / status_file.STATUS_FILENAME).write_text("{", encoding="utf-8")
        lane.fanout(_outcomes(lane), "--force", HIDDEN, str(manifest))
        assert _block(lane) == cp.block_from_manifest(manifest)


# ---------------------------------------------------------------------------
# Verified before anything is written
# ---------------------------------------------------------------------------


class TestAnInputThatChangedRefuses:
    def test_a_changed_input_stops_the_review_writing_nothing(self, lane, capsys):
        manifest = _manifest(lane, stage=2)
        ledger = manifest.parent / cp.REVIEWER_LEDGER
        ledger.write_text("edited by hand\n", encoding="utf-8")
        state = _state_bytes(lane)
        assert lane.fanout(_outcomes(lane), HIDDEN, str(manifest)) == 3
        err = capsys.readouterr().err
        assert f"ERROR: cascade stage input {ledger}" in err
        assert "changed since the stage's manifest was written" in err
        assert not _phase_dir_exists(lane), "the phase dir was created"
        assert _state_bytes(lane) == state

    def test_a_changed_input_never_touches_the_round_on_record(self, lane, capsys):
        manifest = _manifest(lane, stage=2)
        assert lane.fanout(_outcomes(lane), HIDDEN, str(manifest)) == 0
        checkpoint = (lane.phase_dir / status_file.STATUS_FILENAME).read_bytes()
        (manifest.parent / cp.JUDGE_LEDGER).write_text("edited\n", encoding="utf-8")
        assert lane.fanout(_outcomes(lane), "--force", HIDDEN, str(manifest)) == 3
        assert (lane.phase_dir / status_file.STATUS_FILENAME).read_bytes() == checkpoint

    def test_a_missing_manifest_refuses(self, lane, capsys):
        missing = lane.repo.parent / "cascade" / "nowhere" / cp.MANIFEST_NAME
        assert lane.fanout(_outcomes(lane), HIDDEN, str(missing)) == 3
        assert "does not exist" in capsys.readouterr().err
        assert not _phase_dir_exists(lane)

    def test_a_reaggregate_given_a_changed_input_refuses(self, lane, capsys):
        manifest = _manifest(lane, stage=2)
        assert lane.fanout(_outcomes(lane), HIDDEN, str(manifest)) == 0
        (manifest.parent / cp.FULL_LEDGER).write_text("edited\n", encoding="utf-8")
        assert lane.reaggregate(HIDDEN, str(manifest)) == 3
        assert "ERROR: cascade stage input" in capsys.readouterr().err

    def test_the_opening_checks_the_inputs_again(self, lane, monkeypatch, capsys):
        """An input changed after the pre-flight stops the run at the opening,
        before its write: the checkpoint never records this round's models."""
        manifest = _manifest(lane, stage=2)
        real = review_lane.check_cascade_stage

        def check_then_edit(*args, **kwargs):
            real(*args, **kwargs)
            (manifest.parent / cp.REVIEWER_LEDGER).write_text("edited\n", encoding="utf-8")

        monkeypatch.setattr(review_lane, "check_cascade_stage", check_then_edit)
        assert lane.fanout(_outcomes(lane), HIDDEN, str(manifest)) == 3
        assert "changed since the stage's manifest was written" in capsys.readouterr().err
        status = lane.status()
        assert "models_requested" not in status and cp.STATUS_KEY not in status


class TestOnlyTheCurrentCascadesHandoffRuns:
    """A stage command printed for a cascade the composite's pointer no longer
    names as running (superseded by ``--force --cascade`` or a second planner,
    abandoned, finished), or for a stage already closed, never dispatches."""

    @pytest.mark.parametrize("change", [
        "superseded", "abandoned", "finished", "stage-done", "stage-skipped", "no-pointer",
    ])
    def test_a_stale_handoff_stops_the_review_writing_nothing(self, lane, capsys, change):
        manifest = _manifest(lane, stage=2)
        if change == "superseded":
            _point_at(lane, "feedface" * 4, stage=1)
        elif change in ("abandoned", "finished"):
            _point_at(lane, "c0ffee00c0ffee00", stage=2, state=change)
        elif change.startswith("stage-"):
            _point_at(lane, "c0ffee00c0ffee00", stage=2, stage_state=change[len("stage-"):])
        else:
            (phase_dir_path(lane.plan, _composite(lane)) / cp.POINTER_NAME).unlink()
        state = _state_bytes(lane)
        assert lane.fanout(_outcomes(lane), HIDDEN, str(manifest)) == 3
        err = capsys.readouterr().err
        assert "ERROR: cascade stage handoff refused" in err
        assert "nothing was run" in err
        assert not _phase_dir_exists(lane), "the phase dir was created"
        assert _state_bytes(lane) == state

    def test_the_current_running_stage_is_taken(self, lane):
        manifest = _manifest(lane, stage=2)
        assert lane.fanout(_outcomes(lane), HIDDEN, str(manifest)) == 0
        assert _block(lane) == cp.block_from_manifest(manifest)


# ---------------------------------------------------------------------------
# Verdicts are never reused across two stages' ledgers
# ---------------------------------------------------------------------------


def _live_batches(lane) -> List[str]:
    return sorted(p.name for p in lane.phase_dir.glob("validation_batch_*.json")
                  if ".stale" not in p.name)


def _validated_stage(lane, manifest: Path) -> List[str]:
    """A stage's round run, validated and reaggregated; its batch files."""
    assert lane.fanout(_stage_outcome(lane), "--force", "--rerun-all", HIDDEN,
                       str(manifest)) == 0
    for batch in lane.json("validation_tasks.json")["batches"]:
        _write_batch(lane, batch)
    assert lane.reaggregate() == 0
    batches = _live_batches(lane)
    assert batches
    return batches


class TestNoVerdictCrossesALedger:
    def test_a_later_stage_with_the_same_findings_is_validated_again(self, lane):
        """Group ids are content hashes: stage 2's findings repeating stage
        1's word for word regroup to the same hashes. Its validators judge
        them against stage 2's ledger, so stage 1's batch files (judged with
        none) retire with the whole round instead of passing as done."""
        _validated_stage(lane, _manifest(lane, stage=1))
        second = _manifest(lane, stage=2)
        assert lane.fanout(_stage_outcome(lane), "--force", "--rerun-all", HIDDEN,
                           str(second)) == 0
        assert _live_batches(lane) == []
        tasks = lane.json("validation_tasks.json")
        digest = validation.prior_rounds_digest(str(lane.phase_dir))
        assert digest and tasks["inputs"][validation.PRIOR_ROUNDS_INPUT_KEY] == digest

    def test_the_stages_own_resume_keeps_its_batches(self, lane):
        """The same stage resumed (``--force``, its one model, the block
        carried) has the same ledger: its completed batches stay, as before."""
        batches = _validated_stage(lane, _manifest(lane, stage=2))
        assert lane.fanout(_stage_outcome(lane), "--force") == 0
        assert _live_batches(lane) == batches


def _review_with_block(tmp_path: Path, *, stage: int) -> Path:
    """A review dir whose checkpoint records a stage block (stage 1: empty ledgers)."""
    root = tmp_path / "cascade" / f"stage-{stage}" / "inputs"
    root.mkdir(parents=True)
    for name in cp.LEDGER_NAMES:
        (root / name).write_text("" if stage == 1 else f"{name}: earlier\n", encoding="utf-8")
    manifest = cp.write_manifest(root, cascade_id="c0ffee00c0ffee00", mode="improve-plan",
                                 stage=stage, of=2, model=MODELS[0], names=cp.LEDGER_NAMES)
    review = tmp_path / "review-plan"
    review.mkdir()
    status_file.write_status(str(review), {
        "round_id": "a" * 16, cp.STATUS_KEY: cp.block_from_manifest(manifest)})
    return review


class TestPriorRoundsMoved:
    def test_outside_a_cascade_both_record_none(self, tmp_path):
        review = tmp_path / "review-plan"
        review.mkdir()
        assert validation.prior_rounds_digest(str(review)) is None
        assert validation.prior_rounds_moved({"inputs": {}}, str(review)) is None
        assert validation.prior_rounds_moved(None, str(review)) is None

    def test_the_first_stage_records_none(self, tmp_path):
        review = _review_with_block(tmp_path, stage=1)
        assert validation.prior_rounds_digest(str(review)) is None
        assert validation.prior_rounds_moved({"inputs": {}}, str(review)) is None

    def test_a_later_stage_needs_its_own_ledger(self, tmp_path):
        review = _review_with_block(tmp_path, stage=2)
        digest = validation.prior_rounds_digest(str(review))
        assert digest
        key = validation.PRIOR_ROUNDS_INPUT_KEY
        assert validation.prior_rounds_moved({"inputs": {key: digest}}, str(review)) is None
        assert "not reused" in validation.prior_rounds_moved({"inputs": {}}, str(review))
        assert "not reused" in validation.prior_rounds_moved(
            {"inputs": {key: "0" * 64}}, str(review))

    def test_a_validation_json_from_another_ledger_is_not_used(self, tmp_path):
        review = _review_with_block(tmp_path, stage=2)
        (review / "grouped.json").write_text('{"groups": []}', encoding="utf-8")
        (review / "validation.json").write_text('{"groups": [], "metadata": {}}',
                                                encoding="utf-8")
        decide = validation.validation_source_decision
        stale, _note = decide(review / "validation.json", review / "grouped.json",
                              task_document={"inputs": {}})
        assert stale and "not reused" in stale
        key = validation.PRIOR_ROUNDS_INPUT_KEY
        digest = validation.prior_rounds_digest(str(review))
        stale, _note = decide(review / "validation.json", review / "grouped.json",
                              task_document={"inputs": {key: digest}})
        assert stale is None
        # No task file to compare: the decision is today's.
        stale, _note = decide(review / "validation.json", review / "grouped.json",
                              task_document=None)
        assert stale is None
