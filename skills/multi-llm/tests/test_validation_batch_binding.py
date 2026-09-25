"""§7 of plans/validation-context-by-path-and-codex-0156-fallout.md.

Three defects found while redoing a run's validations by hand (2026-09-24):

- §7.1: ``--reaggregate`` kept a ``validation.json`` whose grouping still
  matched even though batch files had been rewritten under it. It is now bound
  to the exact batch bytes it merged (``metadata.batch_sha256`` +
  ``metadata.batch_manifest``); only a file without that binding keeps the
  legacy mtime test.
- §7.2: ``consolidation_recommended`` was never recomputed downward, and the
  reaggregate completion path had no recommendation block at all.
- §7.3: the merge stamped the executor's dispatch runner over every batch,
  including ones rewritten since by hand or by another host. Provenance is now
  per batch and decided by content hash.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils import cli_common, run_status, status_file  # noqa: E402
from utils import review_orchestrator_base as rob  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.subagent_tasks import batch_manifest, file_bytes_sha256  # noqa: E402
from utils.validation import (  # noqa: E402
    prepare_batched_validation_tasks,
    stale_validation_reason,
    validation_source_decision,
)


# ---------------------------------------------------------------------------
# Fixtures: a real review-plan phase dir, reaggregated for real
# ---------------------------------------------------------------------------

NAMES = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]


def _distinct(names):
    """Suggestions no two of which grouping merges (one group per name)."""
    return [
        {
            "title": name,
            "desc": f"{name}{name} {name}{name}",
            "importance": "MEDIUM",
            "reference": f"Step {name}",
            "type": f"kind-{name.lower()}",
        }
        for name in names
    ]


def _args(**over):
    base = dict(
        skip_validation=False,
        validation_model="auto",
        timeout=None,
        internal_validation=False,
        force=False,
        consolidation_threshold=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _phase(tmp_path, names=NAMES):
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n\nDo the thing.\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    phase_dir = out_dir / "review-plan"
    phase_dir.mkdir(parents=True)
    (phase_dir / "claude-code_sonnet.json").write_text(
        json.dumps(_distinct(names)), encoding="utf-8"
    )
    status_file.write_status(str(phase_dir), {
        "phase": "review-plan",
        "state": "models_complete",
        "models_requested": ["claude-code:sonnet"],
        "models_completed": ["claude-code:sonnet"],
        "models_failed": {},
    })
    return plan, out_dir, phase_dir


def _reaggregate(plan, out_dir, phase_dir, *, supports_consolidation=False, **args):
    return asyncio.run(
        rob.reaggregate_from_existing_files(
            prefix="plan",
            out_dir=str(out_dir),
            phase_dir=str(phase_dir),
            phase_name="review-plan",
            plan_path=str(plan),
            args=_args(**args),
            orchestrator_script="review_plan_orchestrator.py",
            supports_consolidation=supports_consolidation,
        )
    )


def _tasks(phase_dir):
    return json.loads((phase_dir / "validation_tasks.json").read_text(encoding="utf-8"))


def _write_batch(phase_dir, batch, status, model=None):
    verdicts = [
        {
            "group_index": global_index,
            "group_hash": batch["group_ids"][local_index],
            "status": status,
            "reason": f"{status} (group {global_index}).",
            "confidence": 0.9,
        }
        for local_index, global_index in enumerate(batch["group_indices"])
    ]
    document = {"groups": verdicts}
    if model:
        document["metadata"] = {"model": model, "schema_version": "2.1"}
    path = phase_dir / f"validation_batch_{batch['batch_index']}.json"
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def _statuses_by_batch(phase_dir, statuses):
    tasks = _tasks(phase_dir)
    for batch, status in zip(tasks["batches"], statuses):
        _write_batch(phase_dir, batch, status)
    return tasks


def _bound_round(tmp_path, capsys=None):
    """Materialize, complete every batch `valid`, merge → a bound validation.json."""
    plan, out_dir, phase_dir = _phase(tmp_path)
    _reaggregate(plan, out_dir, phase_dir)
    tasks = _tasks(phase_dir)
    assert len(tasks["batches"]) >= 2, "the fixture needs at least two batches"
    for batch in tasks["batches"]:
        _write_batch(phase_dir, batch, "valid")
    _reaggregate(plan, out_dir, phase_dir)
    validation = json.loads((phase_dir / "validation.json").read_text(encoding="utf-8"))
    if capsys is not None:
        capsys.readouterr()
    return plan, out_dir, phase_dir, tasks, validation


def _summary(phase_dir):
    report = (phase_dir / "report.md").read_text(encoding="utf-8")
    return next(ln for ln in report.splitlines() if ln.startswith("**Validation:**"))


def _stale(phase_dir):
    return sorted(p.name for p in phase_dir.glob("validation.*.stale.json"))


def _set_mtime(path, seconds):
    os.utime(path, (seconds, seconds))


# ---------------------------------------------------------------------------
# §7.1: validation.json is bound to the batch bytes it merged
# ---------------------------------------------------------------------------

class TestTheMergeRecordsTheBinding:
    def test_every_merged_batch_is_hashed_under_its_round(self, tmp_path):
        _plan, _out, phase_dir, tasks, validation = _bound_round(tmp_path)
        metadata = validation["metadata"]
        assert metadata["batch_sha256"] == {
            str(b["batch_index"]): file_bytes_sha256(
                str(phase_dir / f"validation_batch_{b['batch_index']}.json")
            )
            for b in tasks["batches"]
        }
        assert metadata["batch_manifest"] == batch_manifest(tasks)
        assert metadata["batch_manifest"]["kind"] == "validation"


class TestARewrittenBatchIsReMerged:
    def test_a_bound_file_yields_to_one_rewritten_batch(self, tmp_path, capsys):
        plan, out_dir, phase_dir, tasks, _ = _bound_round(tmp_path, capsys)
        n = len(tasks["batches"])
        assert _summary(phase_dir).startswith(f"**Validation:** {len(NAMES)} valid")
        rewritten = tasks["batches"][1]
        recorded = json.loads(
            (phase_dir / "validation.json").read_text(encoding="utf-8")
        )["metadata"]["batch_sha256"]["1"]
        path = _write_batch(phase_dir, rewritten, "invalid")

        _reaggregate(plan, out_dir, phase_dir)
        out = capsys.readouterr().out

        assert (
            f"Not using validation.json: Re-merging {n} batch files: "
            f"validation_batch_1.json changed (sha256 "
            f"{file_bytes_sha256(str(path))[:12]} != recorded {recorded[:12]})"
        ) in out
        assert len(_stale(phase_dir)) == 1
        flipped = len(rewritten["group_indices"])
        assert _summary(phase_dir) == (
            f"**Validation:** {len(NAMES) - flipped} valid, {flipped} invalid, "
            f"0 needs human review"
        )

    def test_a_rewrite_whose_mtime_was_set_back_is_still_re_merged(
        self, tmp_path, capsys
    ):
        """The `cp -p` / restore-from-backup case: old mtime, new bytes."""
        plan, out_dir, phase_dir, tasks, _ = _bound_round(tmp_path, capsys)
        validation_path = phase_dir / "validation.json"
        _set_mtime(validation_path, 1_700_000_100)
        for batch in tasks["batches"]:
            _set_mtime(phase_dir / f"validation_batch_{batch['batch_index']}.json",
                       1_700_000_000)
        path = _write_batch(phase_dir, tasks["batches"][0], "invalid")
        _set_mtime(path, 1_700_000_000)
        assert path.stat().st_mtime < validation_path.stat().st_mtime

        _reaggregate(plan, out_dir, phase_dir)
        out = capsys.readouterr().out

        assert "validation_batch_0.json changed (sha256" in out
        assert len(_stale(phase_dir)) == 1
        assert " invalid," in _summary(phase_dir)
        assert "0 invalid" not in _summary(phase_dir)

    def test_every_batch_touched_with_unchanged_bytes_keeps_the_file(
        self, tmp_path, capsys
    ):
        """The `git checkout` case: every mtime is now, no byte moved."""
        plan, out_dir, phase_dir, tasks, _ = _bound_round(tmp_path, capsys)
        validation_path = phase_dir / "validation.json"
        before = validation_path.read_bytes()
        _set_mtime(validation_path, 1_700_000_000)
        for batch in tasks["batches"]:
            _set_mtime(phase_dir / f"validation_batch_{batch['batch_index']}.json",
                       1_700_000_500)

        _reaggregate(plan, out_dir, phase_dir)
        out = capsys.readouterr().out

        n = len(tasks["batches"])
        assert "Not using validation.json" not in out
        assert f"{n} batch hashes match) for report generation" in out
        assert "Using validation.json (bound to grouping " in out
        assert _stale(phase_dir) == []
        assert validation_path.read_bytes() == before

    def test_a_listed_batch_deleted_retires_the_file(self, tmp_path, capsys):
        plan, out_dir, phase_dir, tasks, _ = _bound_round(tmp_path, capsys)
        (phase_dir / "validation_batch_1.json").unlink()

        _reaggregate(plan, out_dir, phase_dir)
        out = capsys.readouterr().out

        assert "validation_batch_1.json missing" in out
        assert len(_stale(phase_dir)) == 1
        assert not (phase_dir / "validation.json").exists()

    def test_a_batch_the_file_never_merged_retires_it(self, tmp_path):
        _plan, _out, phase_dir, _tasks_doc, _ = _bound_round(tmp_path)
        path = phase_dir / "validation.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        del data["metadata"]["batch_sha256"]["1"]
        path.write_text(json.dumps(data), encoding="utf-8")

        reason = stale_validation_reason(path, phase_dir / "grouped.json")
        assert reason and "validation_batch_1.json was never merged into it" in reason


class TestARoundIsNeverMatchedAgainstAnotherRound:
    def _revalidation_document(self, tasks):
        """Revalidation reuses `validation_batch_<n>.json` for a different subset."""
        document = json.loads(json.dumps(tasks))
        document["kind"] = "revalidation"
        for batch in document["batches"]:
            batch["kind"] = "revalidation"
            batch["task_id"] = f"revalidation-batch-{batch['batch_index']}"
            batch["group_ids"] = batch["group_ids"][:1]
            batch["group_indices"] = [0]
        return document

    def test_identical_bytes_under_a_revalidation_round_are_retired(self, tmp_path):
        _plan, _out, phase_dir, tasks, _ = _bound_round(tmp_path)
        other = tmp_path / "revalidation_tasks.json"
        other.write_text(json.dumps(self._revalidation_document(tasks)), encoding="utf-8")

        reason = stale_validation_reason(
            phase_dir / "validation.json", phase_dir / "grouped.json", other
        )
        assert reason is not None
        assert "was merged from a validation round" in reason
        assert "is a revalidation round" in reason

    def test_a_revalidation_rounds_file_of_the_same_name_is_not_re_merged(
        self, tmp_path, capsys
    ):
        """The revalidation rewrote `validation_batch_0.json` for its own subset.

        Re-merging it under the validation round's group_ids would discard its
        verdicts as foreign and revert the batch's groups to validation_failed,
        so the still-correct bound file is kept and the reason is named.
        """
        plan, out_dir, phase_dir, tasks, before = _bound_round(tmp_path, capsys)
        revalidation = self._revalidation_document(tasks)
        _write_batch(phase_dir, revalidation["batches"][0], "valid",
                     model="claude-opus-5-5")

        _reaggregate(plan, out_dir, phase_dir)
        out = capsys.readouterr().out
        assert "validation_batch_0.json belongs to another round (sha256" in out
        assert "not re-merging" in out
        assert _stale(phase_dir) == []
        after = json.loads((phase_dir / "validation.json").read_text(encoding="utf-8"))
        statuses = [g.get("status") for g in after.get("groups", [])]
        assert "validation_failed" not in statuses
        assert after == before


class TestTheLegacyFallback:
    def test_a_grouping_bound_file_without_batch_sha256_keeps_the_mtime_test(
        self, tmp_path
    ):
        _plan, _out, phase_dir, tasks, _ = _bound_round(tmp_path)
        path = phase_dir / "validation.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("batch_sha256", "batch_manifest", "runners"):
            del data["metadata"][key]
        path.write_text(json.dumps(data), encoding="utf-8")
        grouped = phase_dir / "grouped.json"

        _set_mtime(path, 1_700_000_100)
        for batch in tasks["batches"]:
            _set_mtime(phase_dir / f"validation_batch_{batch['batch_index']}.json",
                       1_700_000_000)
        reason, note = validation_source_decision(path, grouped)
        assert reason is None
        assert note.endswith("legacy: no batch_sha256, mtime test")

        _set_mtime(phase_dir / "validation_batch_0.json", 1_700_000_200)
        reason, _ = validation_source_decision(path, grouped)
        assert reason is not None
        assert "(legacy: no batch_sha256, mtime test)" in reason
        assert "validation_batch_0.json" in reason


# ---------------------------------------------------------------------------
# §7.3: per-batch provenance, decided by content hash
# ---------------------------------------------------------------------------

EXECUTOR_RUNNER = "codex:gpt-6-astra"


def _validation_group(index):
    return {
        "theme": f"Group {index}",
        "category": "test",
        "models": ["m"],
        "suggestions": [{
            "title": f"t{index}", "desc": f"d{index}", "importance": "MEDIUM",
            "type": "modification",
        }],
    }


def _executor_round(tmp_path, monkeypatch, n_groups=8):
    """Materialize a validation round and dispatch it through the real
    ``run_tasks`` with a stand-in ``dispatch_task`` that writes each batch file,
    so ``subagent_batch_report.json`` carries the executor's own provenance."""
    phase_dir = tmp_path / "review-plan"
    phase_dir.mkdir()
    groups = [_validation_group(i) for i in range(n_groups)]
    document = prepare_batched_validation_tasks(
        groups=groups, context="ctx", output_dir=str(phase_dir),
        plan_file="/path/to/plan.md", max_per_batch=4,
    )
    assert len(document["batches"]) >= 2

    def fake_dispatch(task, *, phase_dir, runner, anchor=None, cwd=None):
        path = _write_batch(Path(phase_dir), task, "valid",
                            model="self-reported-name-copied-from-a-suggestion")
        return executor.TaskOutcome(
            task_id=task["task_id"], kind=task["kind"], runner=runner,
            status=executor.STATUS_OK, output_path=str(path),
        )

    monkeypatch.setattr(executor, "dispatch_task", fake_dispatch)
    monkeypatch.setattr(
        executor, "_resolve_task_source",
        lambda task, **kwargs: (EXECUTOR_RUNNER, "env", "env"),
    )
    monkeypatch.setattr(
        executor, "_preflight_batch_runners",
        lambda resolved, *a, **k: ({EXECUTOR_RUNNER: "0.156.1"}, None),
    )
    executor.run_tasks(
        document["batches"], phase_dir=str(phase_dir), kind="validation",
        resume=False,
    )
    return phase_dir, groups, document


def _merge(phase_dir, groups, document):
    path = phase_dir / "validation.json"
    rob.merge_validation_batches(str(phase_dir), document, groups, str(path))
    return json.loads(path.read_text(encoding="utf-8"))["metadata"]


class TestTheExecutorRecordsPerBatchProvenance:
    def test_each_persisted_batch_is_recorded_with_its_hash_and_round(
        self, tmp_path, monkeypatch
    ):
        phase_dir, _groups, document = _executor_round(tmp_path, monkeypatch)
        report = json.loads(
            (phase_dir / "subagent_batch_report.json").read_text(encoding="utf-8")
        )
        manifest = batch_manifest(document)
        provenance = report["batch_provenance"]
        assert set(provenance) == {b["task_id"] for b in document["batches"]}
        for batch in document["batches"]:
            row = provenance[batch["task_id"]]
            assert row == {
                "task_id": batch["task_id"],
                "kind": "validation",
                "round": manifest["round"],
                "runner": EXECUTOR_RUNNER,
                "sha256": file_bytes_sha256(batch["output_path"]),
            }
        assert report["runner_versions"] == {EXECUTOR_RUNNER: "0.156.1"}

    def test_a_resumed_skip_keeps_its_row_only_while_the_bytes_match(
        self, tmp_path, monkeypatch
    ):
        phase_dir, _groups, document = _executor_round(tmp_path, monkeypatch)
        # Batch 1 is rewritten by hand between the two dispatches.
        _write_batch(phase_dir, document["batches"][1], "invalid",
                     model="claude-opus-5-5")
        executor.run_tasks(
            document["batches"], phase_dir=str(phase_dir), kind="validation",
            resume=True,
        )
        provenance = json.loads(
            (phase_dir / "subagent_batch_report.json").read_text(encoding="utf-8")
        )["batch_provenance"]
        assert "validation-batch-0" in provenance
        assert "validation-batch-1" not in provenance


class TestMergeAttributesEachBatchByContent:
    def test_a_hand_rewritten_batch_is_attributed_to_its_own_model(
        self, tmp_path, monkeypatch
    ):
        phase_dir, groups, document = _executor_round(tmp_path, monkeypatch)
        _write_batch(phase_dir, document["batches"][1], "valid",
                     model="claude-opus-5-5")

        metadata = _merge(phase_dir, groups, document)

        assert metadata["runners"]["1"] == "claude-opus-5-5"
        others = {k: v for k, v in metadata["runners"].items() if k != "1"}
        assert others and set(others.values()) == {EXECUTOR_RUNNER}
        assert metadata["model"] == ",".join(sorted({"claude-opus-5-5", EXECUTOR_RUNNER}))
        assert metadata["runner"] == EXECUTOR_RUNNER

    def test_a_rewrite_with_its_mtime_reset_into_the_dispatch_window_is_still_its_own(
        self, tmp_path, monkeypatch
    ):
        phase_dir, groups, document = _executor_round(tmp_path, monkeypatch)
        batch_path = phase_dir / "validation_batch_1.json"
        original = batch_path.stat()
        _write_batch(phase_dir, document["batches"][1], "valid",
                     model="claude-opus-5-5")
        os.utime(batch_path, ns=(original.st_atime_ns, original.st_mtime_ns))

        metadata = _merge(phase_dir, groups, document)

        assert metadata["runners"]["1"] == "claude-opus-5-5"

    def test_an_unchanged_batch_whose_mtime_moved_keeps_the_recorded_runner(
        self, tmp_path, monkeypatch
    ):
        phase_dir, groups, document = _executor_round(tmp_path, monkeypatch)
        for batch in document["batches"]:
            _set_mtime(Path(batch["output_path"]), 1_900_000_000)

        metadata = _merge(phase_dir, groups, document)

        assert set(metadata["runners"].values()) == {EXECUTOR_RUNNER}
        assert metadata["model"] == EXECUTOR_RUNNER
        assert metadata["runner"] == EXECUTOR_RUNNER

    def test_mixed_runners_list_each_batch_and_join_the_model(
        self, tmp_path, monkeypatch
    ):
        phase_dir, groups, document = _executor_round(tmp_path, monkeypatch)
        _write_batch(phase_dir, document["batches"][0], "valid",
                     model="opencode/nemotron-3-ultra-free")
        _write_batch(phase_dir, document["batches"][1], "valid",
                     model="claude-opus-5-5")

        metadata = _merge(phase_dir, groups, document)

        assert set(metadata["runners"]) == {
            str(b["batch_index"]) for b in document["batches"]
        }
        assert metadata["runners"]["0"] == "opencode/nemotron-3-ultra-free"
        assert metadata["runners"]["1"] == "claude-opus-5-5"
        expected = sorted(set(metadata["runners"].values()))
        assert metadata["model"] == ",".join(expected)
        assert "mixed" not in metadata["model"]

    def test_a_validation_round_record_is_not_credited_to_a_revalidation_round(
        self, tmp_path, monkeypatch
    ):
        """Same filenames, same bytes, different round: the record is not reused."""
        phase_dir, groups, document = _executor_round(tmp_path, monkeypatch)
        revalidation = json.loads(json.dumps(document))
        revalidation["kind"] = "revalidation"
        for batch in revalidation["batches"]:
            batch["task_id"] = f"validation-batch-{batch['batch_index']}"
            batch["group_ids"] = batch["group_ids"][:1]
            batch["group_indices"] = batch["group_indices"][:1]

        metadata = _merge(phase_dir, groups, revalidation)

        assert EXECUTOR_RUNNER not in metadata["runners"].values()
        assert "runner" not in metadata
        assert metadata["model"] == "self-reported-name-copied-from-a-suggestion"

    def test_no_report_falls_back_to_each_batch_self_report(self, tmp_path):
        """The host-native path: nothing was executor-dispatched."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()
        groups = [_validation_group(i) for i in range(8)]
        document = prepare_batched_validation_tasks(
            groups=groups, context="ctx", output_dir=str(phase_dir),
            plan_file="/path/to/plan.md", max_per_batch=4,
        )
        for batch in document["batches"]:
            _write_batch(phase_dir, batch, "valid", model="claude-code:sonnet")

        metadata = _merge(phase_dir, groups, document)

        assert metadata["model"] == "claude-code:sonnet"
        assert "runner" not in metadata
        assert set(metadata["runners"].values()) == {"claude-code:sonnet"}


# ---------------------------------------------------------------------------
# §7.2: consolidation_recommended is recomputed on every completion path
# ---------------------------------------------------------------------------

def _read_checkpoint(phase_dir):
    with open(phase_dir / ".status.json", encoding="utf-8") as f:
        return json.load(f)


def _consolidation_phase(tmp_path, statuses):
    """A bound round whose batches carry *statuses*, one per group, in order."""
    names = [f"Word{i:02d}x{'q' * (i + 1)}" for i in range(len(statuses))]
    plan, out_dir, phase_dir = _phase(tmp_path, names=names)
    _reaggregate(plan, out_dir, phase_dir)
    return plan, out_dir, phase_dir


def _complete(phase_dir, verdict_for_group):
    tasks = _tasks(phase_dir)
    for batch in tasks["batches"]:
        verdicts = [
            {
                "group_index": gi,
                "group_hash": batch["group_ids"][li],
                "status": verdict_for_group(gi),
                "reason": "r",
                "confidence": 0.9,
            }
            for li, gi in enumerate(batch["group_indices"])
        ]
        (phase_dir / f"validation_batch_{batch['batch_index']}.json").write_text(
            json.dumps({"groups": verdicts}), encoding="utf-8"
        )


def _reaggregate_with_result(plan, out_dir, phase_dir, threshold=None):
    context = cli_common.activate_result(cli_common.ResultContext())
    try:
        _reaggregate(plan, out_dir, phase_dir, supports_consolidation=True,
                     consolidation_threshold=threshold)
        payload = context.payload(0)
    finally:
        cli_common.deactivate_result()
    return payload


def _counted_round(tmp_path, n_groups, counted, threshold=None,
                   extra_status="needs-human-decision"):
    """First merge with `counted` of `n_groups` counted (`valid`)."""
    plan, out_dir, phase_dir = _consolidation_phase(tmp_path, [None] * n_groups)
    _complete(phase_dir, lambda gi: "valid" if gi < counted else "invalid")
    payload = _reaggregate_with_result(plan, out_dir, phase_dir, threshold)
    return plan, out_dir, phase_dir, payload


class TestTheRecommendationIsRecomputed:
    def test_a_reaggregate_after_a_flip_rewrites_the_count(self, tmp_path):
        plan, out_dir, phase_dir, payload = _counted_round(tmp_path, 20, 20)
        assert payload["consolidation_recommended"]["valid_groups"] == 20
        assert _read_checkpoint(phase_dir)["consolidation_recommended"]["valid_groups"] == 20

        _complete(phase_dir, lambda gi: "valid" if gi < 17 else "invalid")
        payload = _reaggregate_with_result(plan, out_dir, phase_dir)

        expected = {"valid_groups": 17, "threshold": 15, "command": "--consolidate"}
        assert _read_checkpoint(phase_dir)["consolidation_recommended"] == expected
        assert payload["consolidation_recommended"] == expected

    def test_dropping_below_the_default_threshold_removes_the_key(self, tmp_path):
        plan, out_dir, phase_dir, payload = _counted_round(tmp_path, 16, 15)
        assert payload["consolidation_recommended"]["valid_groups"] == 15

        _complete(phase_dir, lambda gi: "valid" if gi < 14 else "invalid")
        payload = _reaggregate_with_result(plan, out_dir, phase_dir)

        assert "consolidation_recommended" not in _read_checkpoint(phase_dir)
        assert "consolidation_recommended" not in payload

    def test_a_reaggregate_that_reaches_zero_groups_removes_the_key(self, tmp_path):
        plan, out_dir, phase_dir, payload = _counted_round(tmp_path, 16, 16)
        assert "consolidation_recommended" in _read_checkpoint(phase_dir)

        _complete(phase_dir, lambda gi: "invalid")
        payload = _reaggregate_with_result(plan, out_dir, phase_dir)
        assert "consolidation_recommended" not in _read_checkpoint(phase_dir)
        assert "consolidation_recommended" not in payload

    def test_a_reaggregate_with_no_suggestions_left_removes_the_key(self, tmp_path):
        plan, out_dir, phase_dir, _ = _counted_round(tmp_path, 16, 16)
        (phase_dir / "claude-code_sonnet.json").write_text("[]", encoding="utf-8")

        payload = _reaggregate_with_result(plan, out_dir, phase_dir)

        assert "consolidation_recommended" not in _read_checkpoint(phase_dir)
        assert "consolidation_recommended" not in payload

    def test_a_configured_threshold_is_honoured_both_ways(self, tmp_path):
        plan, out_dir, phase_dir, payload = _counted_round(tmp_path, 22, 22, threshold=20)
        assert payload["consolidation_recommended"]["threshold"] == 20

        _complete(phase_dir, lambda gi: "valid" if gi < 19 else "invalid")
        payload = _reaggregate_with_result(plan, out_dir, phase_dir, threshold=20)
        assert "consolidation_recommended" not in _read_checkpoint(phase_dir)
        assert "consolidation_recommended" not in payload

        _complete(phase_dir, lambda gi: "valid" if gi < 20 else "invalid")
        payload = _reaggregate_with_result(plan, out_dir, phase_dir, threshold=20)
        expected = {"valid_groups": 20, "threshold": 20, "command": "--consolidate"}
        assert _read_checkpoint(phase_dir)["consolidation_recommended"] == expected
        assert payload["consolidation_recommended"] == expected

    def test_a_needs_human_decision_group_counts(self, tmp_path):
        plan, out_dir, phase_dir, _ = _counted_round(tmp_path, 16, 0)
        _complete(phase_dir, lambda gi: "valid" if gi < 14 else "needs-human-decision")
        payload = _reaggregate_with_result(plan, out_dir, phase_dir)
        assert payload["consolidation_recommended"]["valid_groups"] == 16
        assert _read_checkpoint(phase_dir)["consolidation_recommended"]["valid_groups"] == 16

    def test_a_phase_without_consolidation_writes_nothing(self, tmp_path):
        plan, out_dir, phase_dir = _consolidation_phase(tmp_path, [None] * 16)
        _complete(phase_dir, lambda gi: "valid")
        _reaggregate(plan, out_dir, phase_dir, supports_consolidation=False)
        assert "consolidation_recommended" not in _read_checkpoint(phase_dir)


class TestALegacyNullRecommendation:
    """Checkpoints written before §7.2 may hold `"consolidation_recommended": null`."""

    def test_status_prints_no_line_and_the_result_omits_it(self, tmp_path, capsys):
        plan = tmp_path / "auth-feature.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        phase_dir = rob.get_phase_dir(plan, "review-plan")
        status_file.write_status(str(phase_dir), {
            "phase": "review-plan",
            "state": "complete",
            "consolidation_recommended": None,
        })
        assert _read_checkpoint(phase_dir)["consolidation_recommended"] is None

        context = cli_common.activate_result(cli_common.ResultContext())
        try:
            run_status.main(["--plan-file", str(plan), "--view", "run"])
            payload = context.payload(0)
        finally:
            cli_common.deactivate_result()
        out = capsys.readouterr().out
        assert "consolidation recommended" not in out
        assert "consolidation_recommended" not in payload

    def test_detach_classify_run_passes_the_null_through_as_none(self, tmp_path):
        from utils import detach

        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()
        status_file.write_status(str(phase_dir), {
            "phase": "review-plan", "state": "complete",
            "consolidation_recommended": None,
        })
        view = detach.classify_run(str(phase_dir))
        assert view.get("consolidation_recommended") is None


class TestRemoveStatusKeys:
    def test_the_key_goes_and_every_sibling_stays(self, tmp_path):
        phase_dir = str(tmp_path)
        status_file.write_status(phase_dir, {
            "phase": "review-plan",
            "consolidation_recommended": {"valid_groups": 48},
            "run": {"run_id": "r1", "state": "done"},
            "progress": {"step": "complete", "models_done": 3},
        })
        status_file.remove_status_keys(phase_dir, ["consolidation_recommended"])
        data = _read_checkpoint(tmp_path)
        assert "consolidation_recommended" not in data
        assert data["phase"] == "review-plan"
        assert data["run"] == {"run_id": "r1", "state": "done"}
        assert data["progress"] == {"step": "complete", "models_done": 3}

    def test_a_missing_key_is_a_no_op(self, tmp_path):
        phase_dir = str(tmp_path)
        status_file.write_status(phase_dir, {"phase": "review-plan"})
        before = (tmp_path / ".status.json").read_bytes()
        status_file.remove_status_keys(phase_dir, ["consolidation_recommended"])
        assert (tmp_path / ".status.json").read_bytes() == before

    def test_a_missing_file_is_a_no_op(self, tmp_path):
        status_file.remove_status_keys(str(tmp_path), ["consolidation_recommended"])
        assert not (tmp_path / ".status.json").exists()
