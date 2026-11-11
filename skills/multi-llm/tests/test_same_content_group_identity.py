"""Same-content groups keep their own identity from validation to apply.

One review-code finding reported in twelve files is twelve groups with one
content hash (``generate_group_id`` reads neither ``file`` nor
``line_range``); ``stamp_stable_ids`` tells them apart by discriminator, and
every record about a group is filed under that stamp. Readers that recomputed
the plain hash gave every group the first one's record:

- a ``--resume`` dropped all twelve once the first was recorded applied, and
  one human decision decided all twelve;
- validation task ids, the batch merge and the verdict join all used the
  plain hash, so one group's verdict reached all twelve: eleven invalid
  findings read valid and were applied.

Each test runs the real path: ``prepare_batched_validation_tasks`` →
validator replies copied from the batch prompt → ``merge_validation_batches``
→ ``validation.json`` → ``apply-code-fixes``, and the native apply leg's
``util batch-done`` record → ``apply-code-fixes --resume``. Files written
before the stamp reached every writer still resume and join, but a shared
plain hash never picks a group.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

import apply_code_fixes_orchestrator  # noqa: E402
from utils import batch_done  # noqa: E402
from utils import scope_policy as sp  # noqa: E402
from utils import scope_round  # noqa: E402
from utils.review_orchestrator_base import merge_validation_batches  # noqa: E402
from utils.state_manager import (  # noqa: E402
    StateManager,
    generate_group_id,
    save_groups_payload,
    stamp_stable_ids,
)
from utils.validation import (  # noqa: E402
    apply_validation_to_groups,
    merge_batched_validation_results,
    prepare_batched_validation_tasks,
)
from utils.write_phase_bridge import WRITE_TASK_FILENAME  # noqa: E402

FILES = [f"mod{i}.py" for i in range(12)]
#: The one file whose finding the validator upholds.
REAL = 11


def _finding_in(path):
    return {
        "theme": "Missing error handling", "category": "bug", "models": ["m"],
        "suggestions": [{
            "title": "Missing error handling",
            "desc": "Wrap the call in try/except.",
            "type": "bug", "importance": "MEDIUM",
            "file": path, "line_range": "1-1", "source_model": "m",
        }],
    }


def _groups():
    groups = [_finding_in(path) for path in FILES]
    stamp_stable_ids(groups)
    return groups


@pytest.fixture
def review(tmp_path, monkeypatch):
    """A repo holding a code-review cycle over twelve same-content groups,
    code-review recorded completed; returns (plan, code-review dir, groups)."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for path in FILES:
        (root / path).write_text("x = 1\n", encoding="utf-8")
    plan = root / "plans" / "p.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# Plan\n\n## Step 1\nHandle errors.\n", encoding="utf-8")
    code_review = root / "plans" / "p" / "code-review"
    code_review.mkdir(parents=True)
    groups = _groups()
    assert len({generate_group_id(g) for g in groups}) == 1
    assert len({g["group_hash"] for g in groups}) == len(FILES)
    (code_review / "grouped.json").write_text(
        json.dumps(save_groups_payload(groups)), encoding="utf-8")
    state = StateManager(plan)
    state.mark_phase_completed("code-review")
    state.save()
    monkeypatch.chdir(root)
    return plan, code_review, groups


def _write_validation(code_review, records):
    (code_review / "validation.json").write_text(
        json.dumps({"format_version": 2, "groups": records}), encoding="utf-8")


def _all_valid(code_review, groups):
    _write_validation(code_review, [
        {"group_index": i, "group_id": g["group_hash"], "status": "valid",
         "reason": "confirmed", "confidence": 0.9}
        for i, g in enumerate(groups)
    ])


def _emit(plan, capsys, *flags):
    """One native apply-code-fixes run: exit, stderr and the emitted tasks."""
    try:
        code = apply_code_fixes_orchestrator.main([
            "--plan-file", str(plan), "--yes", "--output-format", "json", *flags,
        ])
    except SystemExit as stop:
        code = stop.code
    err = capsys.readouterr().err
    task_file = plan.parent / "p" / "apply-fixes" / WRITE_TASK_FILENAME
    tasks = (json.loads(task_file.read_text(encoding="utf-8"))["batches"]
             if task_file.exists() else [])
    return code, err, tasks


def _emitted_ids(tasks):
    return [i for task in tasks for i in task["expected_ids"]]


def _resume_line(err):
    return next(line for line in err.splitlines() if line.startswith("[resume]"))


# ===========================================================================
# --resume
# ===========================================================================

class TestAResumeKeepsEverySameContentGroupApart:

    def test_a_recorded_batch_drops_only_its_own_group(self, review, capsys):
        plan, code_review, groups = review
        _all_valid(code_review, groups)
        code, err, first = _emit(plan, capsys)
        assert code == 0, err
        assert _emitted_ids(first) == [g["group_hash"] for g in groups]

        # The native leg records the first batch, as the host does after it.
        assert batch_done.main([
            "--phase-dir", str(plan.parent / "p" / "apply-fixes"),
            "--task-id", first[0]["task_id"], "--status", "completed",
        ]) == 0
        capsys.readouterr()

        code, err, second = _emit(plan, capsys, "--resume")
        assert code == 0, err
        assert _resume_line(err) == "[resume] 1 already processed, 11 remaining"
        assert _emitted_ids(second) == [g["group_hash"] for g in groups[1:]]

    def test_a_human_decision_decides_only_its_own_group(self, review, capsys):
        plan, code_review, groups = review
        _all_valid(code_review, groups)
        state = StateManager(plan)
        state.record_human_decision(
            phase="apply-fixes", group_id=groups[0]["group_hash"],
            decision="skipped", reason="not this one")
        state.save()

        code, err, tasks = _emit(plan, capsys, "--resume")
        assert code == 0, err
        assert _emitted_ids(tasks) == [g["group_hash"] for g in groups[1:]]

    def test_a_record_under_the_recomputed_hash_of_an_edited_group_still_counts(
        self, review, capsys,
    ):
        """Before records were filed under the stamp, a resume looked a group
        up by the content hash of the group as edited; such a record names
        one group alone and still drops it."""
        plan, code_review, groups = review
        _all_valid(code_review, groups)
        edited = copy.deepcopy(groups[0])
        edited["suggestions"][0]["desc"] = "Wrap the call in try/except and log it."
        (code_review / "user_selections.json").write_text(json.dumps({
            "format_version": 2, "plan_path": str(plan), "phase": "code-review",
            "edited_descriptions": {
                groups[0]["suggestions"][0]["suggestion_hash"]:
                    edited["suggestions"][0]["desc"],
            },
        }), encoding="utf-8")
        legacy_id = generate_group_id(edited)
        assert legacy_id not in {g["group_hash"] for g in groups}
        state = StateManager(plan)
        state.mark_item_processed("apply-fixes", legacy_id, "applied")
        state.save()

        code, err, tasks = _emit(plan, capsys, "--resume")
        assert code == 0, err
        assert _resume_line(err) == "[resume] 1 already processed, 11 remaining"
        assert _emitted_ids(tasks) == [g["group_hash"] for g in groups[1:]]


# ===========================================================================
# Validation, through the real merge and apply paths
# ===========================================================================

def _records_in(prompt):
    """The suggestion records a validation batch prompt embeds."""
    at = prompt.index("[", prompt.index("## Suggestion groups to validate"))
    return json.JSONDecoder().raw_decode(prompt[at:])[0]


def _reply_to_every_batch(document):
    """Answer each batch as a validator does: copy each record's group_hash;
    only the finding in mod11.py is real."""
    for batch in document["batches"]:
        verdicts = []
        for record in _records_in(batch["prompt"]):
            index = record["group_index"]
            verdicts.append({
                "group_index": index, "group_hash": record["group_hash"],
                "status": "valid" if index == REAL else "invalid",
                "reason": f"judged {record['suggestions'][0]['file']}",
                "confidence": 0.9,
                sp.RELATION_FIELD: sp.IN_SCOPE,
                sp.REASON_FIELD: f"scope of {record['suggestions'][0]['file']}",
            })
        Path(batch["output_path"]).write_text(json.dumps(
            {"groups": verdicts, "metadata": {"model": "validator"}}), encoding="utf-8")


class TestEachVerdictReachesOnlyItsOwnGroup:

    def test_eleven_invalid_findings_are_never_applied(self, review, capsys):
        plan, code_review, groups = review
        document = prepare_batched_validation_tasks(
            groups=groups, context="the diff", output_dir=str(code_review),
            plan_file=str(plan))
        task_ids = [gid for batch in document["batches"] for gid in batch["group_ids"]]
        assert sorted(task_ids) == sorted(g["group_hash"] for g in groups)
        _reply_to_every_batch(document)

        validated = merge_validation_batches(
            str(code_review), document, groups, str(code_review / "validation.json"))
        assert [g["validation_status"] for g in validated] == (
            ["invalid"] * REAL + ["valid"])
        assert [g["validation_reason"] for g in validated] == [
            f"judged {path}" for path in FILES]
        # Scope authority joined each group by its own stamp too.
        assert [g[sp.VERDICT_KEY]["reason"] for g in validated] == [
            f"scope of {path}" for path in FILES]
        saved = json.loads((code_review / "validation.json").read_text())["groups"]
        assert [r["group_id"] for r in saved] == [g["group_hash"] for g in groups]
        assert all(r.get(sp.PROPOSAL_DIGEST_FIELD) for r in saved)

        # The report path's own attach reads the same file the same way.
        reread = copy.deepcopy(groups)
        scope_round.attach_verdicts(reread, saved)
        assert [g[sp.VERDICT_KEY]["reason"] for g in reread] == [
            f"scope of {path}" for path in FILES]

        capsys.readouterr()
        code, err, tasks = _emit(plan, capsys)
        assert code == 0, err
        assert _emitted_ids(tasks) == [groups[REAL]["group_hash"]]
        assert [task["declared_files"] for task in tasks] == [[FILES[REAL]]]

    def test_a_validation_file_written_before_stamped_ids_never_crosses_groups(
        self, review, capsys,
    ):
        """Such a file gives every same-content group the first one's plain
        hash, so the hash says nothing: the review join leaves the twelve
        undecided, and the apply join places each by its own index."""
        plan, code_review, groups = review
        plain = generate_group_id(groups[0])
        legacy = [
            {"group_index": i, "group_id": plain,
             "status": "valid" if i == REAL else "invalid",
             "reason": f"judged {FILES[i]}", "confidence": 0.9}
            for i in range(len(FILES))
        ]
        _write_validation(code_review, legacy)

        rejoined = apply_validation_to_groups(groups, legacy)
        assert [g["validation_status"] for g in rejoined] == (
            ["needs-human-decision"] * len(FILES))

        code, err, tasks = _emit(plan, capsys)
        assert code == 0, err
        assert _emitted_ids(tasks) == [groups[REAL]["group_hash"]]

    def test_a_task_file_that_gave_two_groups_one_id_leaves_both_for_revalidation(
        self, review,
    ):
        """A round materialized before stamped ids: the batch's verdicts that
        echo the shared id cannot say which group they judged."""
        _plan, code_review, groups = review
        pair = groups[:2]
        plain = generate_group_id(pair[0])
        document = {
            "batches": [{
                "batch_index": 0, "group_indices": [0, 1],
                "group_ids": [plain, plain],
            }],
        }
        (code_review / "validation_batch_0.json").write_text(json.dumps({"groups": [
            {"group_index": 0, "group_hash": plain, "status": "invalid", "reason": "a"},
            {"group_index": 1, "group_hash": plain, "status": "valid", "reason": "b"},
        ]}), encoding="utf-8")

        merged = merge_batched_validation_results(
            str(code_review), document, total_groups=2, groups=pair)
        assert [r["status"] for r in merged] == ["validation_failed"] * 2

    def test_a_preserve_rounds_decision_lands_on_each_groups_own_record(self):
        from utils.review_orchestrator_base import scope_decisions_for

        groups = _groups()
        records = []
        for i, group in enumerate(groups):
            record = sp.stamp_validation_record(
                {"group_index": i, "group_id": group["group_hash"], "status": "valid",
                 sp.RELATION_FIELD: sp.SCOPE_EXTENSION if i == REAL else sp.IN_SCOPE,
                 sp.REASON_FIELD: f"scope of {FILES[i]}"},
                group, baseline_id="b1", source="validation", allow_selected_remedy=True)
            sp.attach_verdict(group, record, hash_matched=True)
            group["validation_status"] = "valid"
            records.append(record)

        counts = scope_decisions_for(
            groups, records, {"policy": sp.POLICY_PRESERVE, "round_baseline_id": "b1"})
        assert counts["deferred"] == 1
        assert [r[sp.DECISION_KEY]["outcome"] for r in records] == (
            ["eligible"] * REAL + ["deferred"])
        assert all(FILES[i] in r[sp.DECISION_KEY]["reason"] for i, r in enumerate(records))


# ===========================================================================
# The join rule itself
# ===========================================================================

class TestTheJoinRule:

    def test_the_stamp_joins_and_a_shared_plain_hash_names_no_one_else(self):
        from utils.state_manager import group_join_ids

        groups = _groups()
        assert group_join_ids(groups) == [(g["group_hash"],) for g in groups]

    def test_a_plain_hash_joins_only_the_one_group_it_names(self):
        from utils.state_manager import group_join_ids

        groups = _groups()[:2]
        edited = copy.deepcopy(groups[0])
        edited["suggestions"][0]["desc"] = "Something else."
        unstamped = [_finding_in("a.py"), _finding_in("b.py"), _finding_in("c.py")]
        unstamped[2]["suggestions"][0]["desc"] = "Unique."
        assert group_join_ids([edited, groups[1]]) == [
            (groups[0]["group_hash"], generate_group_id(edited)),
            (groups[1]["group_hash"],),
        ]
        # Unstamped same-content groups cannot be told apart at all.
        assert group_join_ids(unstamped) == [(), (), (generate_group_id(unstamped[2]),)]
        # Nor can groups a file stamped with one shared hash.
        shared = [dict(g, group_hash="h") for g in unstamped[:2]]
        assert group_join_ids(shared) == [(), ()]

    def test_an_id_two_records_carry_is_left_out(self):
        from utils.state_manager import records_by_unique_id

        records = [{"group_id": "a", "n": 1}, {"group_hash": "b", "n": 2},
                   {"group_id": "a", "n": 3}, {"n": 4}, "not a record"]
        assert records_by_unique_id(records, "group_id", "group_hash") == {
            "b": {"group_hash": "b", "n": 2}}

    def test_a_validation_file_saved_with_its_groups_files_each_under_its_stamp(
        self, tmp_path,
    ):
        from utils.validation import save_validation_results

        groups = _groups()
        target = tmp_path / "validation.json"
        save_validation_results(
            [{"group_index": i, "status": "invalid", "reason": "r"}
             for i in range(len(groups))],
            target, model="m", groups=groups, grouped_file=str(tmp_path / "none.json"))
        saved = json.loads(target.read_text())["groups"]
        assert [r["group_id"] for r in saved] == [g["group_hash"] for g in groups]
