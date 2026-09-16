"""D2b verdict-ingest tests: parse -> validate -> group_id map -> record.

Materializing judge prompts without a consumer leaves the loop open, so these
tests pin the consumer end to end: the verdict schema, the stable-``group_id``
mapping (never position), the exact ``record_human_decision`` translation that
makes both judging legs land on one recorded state, the work-set rewrite with
``salvaged_description`` substituted before batching, and the explicit handling of
missing and duplicate verdicts.
"""

import json
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils.human_decisions import (  # noqa: E402
    DECISION_APPROVE,
    DECISION_SALVAGE,
    DECISION_SKIP,
    JUDGE_TASKS_FILENAME,
    apply_verdicts_to_work_set,
    build_judge_tasks,
    get_persisted_verdicts,
    group_id_of,
    ingest_verdicts,
    judge_item_from_group,
    read_verdict_files,
    record_verdicts,
    substitute_salvaged_description,
    write_judge_tasks,
)
from utils.schema_validator import validate_human_decision_verdicts  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402
from utils.subagent_tasks import tasks_of, validate_subagent_task_file  # noqa: E402


GROUP_A = "aaaa000000000000"
GROUP_B = "bbbb000000000000"
GROUP_C = "cccc000000000000"


def _group(group_hash, title="A suggestion", desc="Original description",
           importance="MEDIUM"):
    return {
        "group_hash": group_hash,
        "theme": title,
        "validation_status": "needs-human-decision",
        "validation_reason": "Could go either way",
        "validation_confidence": 0.4,
        "models": ["model-a"],
        "suggestions": [
            {"title": title, "desc": desc, "importance": importance,
             "type": "modification"}
        ],
    }


def _verdict(group_id, decision, reason="because", salvaged=None, dropped=None):
    verdict = {"group_id": group_id, "decision": decision, "reason": reason}
    if decision == DECISION_SALVAGE:
        verdict["salvaged_description"] = salvaged or "Keep only the real part."
        verdict["dropped"] = dropped or "the speculative queue"
    return verdict


def _write_verdicts(phase_dir: Path, index: int, verdicts):
    path = phase_dir / f"human_decision_batch_{index}.json"
    path.write_text(
        json.dumps({"schema_version": "1.0", "verdicts": verdicts}, indent=2),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestVerdictSchema:
    def test_wrapper_object_with_verdicts_validates(self):
        document = {"schema_version": "1.0", "verdicts": [
            _verdict(GROUP_A, DECISION_APPROVE),
            _verdict(GROUP_B, DECISION_SALVAGE),
            _verdict(GROUP_C, DECISION_SKIP),
        ]}
        assert validate_human_decision_verdicts(document) == []

    def test_a_legitimately_empty_verdict_set_validates(self):
        """The wrapper (rather than a bare array) is what lets the ingest tell an
        empty verdict set apart from a truncated write."""
        assert validate_human_decision_verdicts(
            {"schema_version": "1.0", "verdicts": []}
        ) == []

    def test_bare_array_is_rejected(self):
        assert validate_human_decision_verdicts([_verdict(GROUP_A, "approve")])

    def test_missing_wrapper_key_is_rejected(self):
        assert validate_human_decision_verdicts({"verdicts": []})

    def test_unknown_decision_is_rejected(self):
        document = {"schema_version": "1.0",
                    "verdicts": [{"group_id": GROUP_A, "decision": "maybe",
                                  "reason": "r"}]}
        errors = validate_human_decision_verdicts(document)
        assert any("decision" in e for e in errors)

    def test_salvage_requires_non_empty_salvaged_description(self):
        document = {"schema_version": "1.0", "verdicts": [
            {"group_id": GROUP_A, "decision": "salvage", "reason": "r",
             "salvaged_description": "", "dropped": "x"}
        ]}
        assert any("salvaged_description" in e
                   for e in validate_human_decision_verdicts(document))

    def test_salvage_without_the_fields_is_rejected(self):
        document = {"schema_version": "1.0", "verdicts": [
            {"group_id": GROUP_A, "decision": "salvage", "reason": "r"}
        ]}
        errors = validate_human_decision_verdicts(document)
        assert any("salvaged_description" in e for e in errors)

    @pytest.mark.parametrize("decision", ["approve", "skip"])
    def test_non_salvage_must_leave_the_salvage_fields_null(self, decision):
        document = {"schema_version": "1.0", "verdicts": [
            {"group_id": GROUP_A, "decision": decision, "reason": "r",
             "salvaged_description": "sneaky rewrite", "dropped": None}
        ]}
        assert any("salvaged_description" in e
                   for e in validate_human_decision_verdicts(document))

    def test_explicit_nulls_on_approve_are_fine(self):
        document = {"schema_version": "1.0", "verdicts": [
            {"group_id": GROUP_A, "decision": "approve", "reason": "r",
             "salvaged_description": None, "dropped": None}
        ]}
        assert validate_human_decision_verdicts(document) == []

    def test_unknown_verdict_field_is_rejected(self):
        document = {"schema_version": "1.0", "verdicts": [
            {"group_id": GROUP_A, "decision": "approve", "reason": "r",
             "confidence": 0.9}
        ]}
        assert any("confidence" in e
                   for e in validate_human_decision_verdicts(document))

    def test_empty_group_id_is_rejected(self):
        document = {"schema_version": "1.0", "verdicts": [
            {"group_id": "", "decision": "approve", "reason": "r"}
        ]}
        assert any("group_id" in e
                   for e in validate_human_decision_verdicts(document))


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------

class TestJudgeTaskMaterialization:
    def test_one_task_per_item_keyed_by_group_id(self, tmp_path):
        items = [judge_item_from_group(_group(g)) for g in (GROUP_A, GROUP_B)]
        document = build_judge_tasks(items=items, phase_dir=str(tmp_path))
        assert validate_subagent_task_file(document) == []
        tasks = tasks_of(document)
        assert [t["group_id"] for t in tasks] == [GROUP_A, GROUP_B]
        assert [t["task_id"] for t in tasks] == [
            f"human_decision_judge-{GROUP_A}", f"human_decision_judge-{GROUP_B}"
        ]
        for task in tasks:
            assert task["kind"] == "human_decision_judge"
            assert task["result_channel"] == "file"
            assert task["capability_tier"] == "full"
            assert task["executor_role"] == "agent"
            assert task["prompt"]
            assert task["output_name"].startswith("human_decision_batch_")
        assert [t["output_name"] for t in tasks] == [
            f"human_decision_batch_{GROUP_A}.json",
            f"human_decision_batch_{GROUP_B}.json",
        ]

    def test_output_name_survives_a_partially_judged_resume(self, tmp_path):
        """A leftover item keeps its own verdict filename on re-materialization.

        The producer is called with the *remaining* items, so an index-keyed
        ``output_name`` would hand round two's task round one's verdict file —
        which the executor's resume rule (exists + parses + validates) reads as
        "already done", leaving that group judged by nobody.
        """
        first = build_judge_tasks(
            items=[judge_item_from_group(_group(g)) for g in (GROUP_A, GROUP_B)],
            phase_dir=str(tmp_path),
        )
        resumed = build_judge_tasks(
            items=[judge_item_from_group(_group(GROUP_B))], phase_dir=str(tmp_path)
        )
        assert tasks_of(resumed)[0]["output_name"] == tasks_of(first)[1]["output_name"]
        assert tasks_of(resumed)[0]["output_name"] != tasks_of(first)[0]["output_name"]

    def test_the_item_details_travel_inside_the_prompt(self, tmp_path):
        group = _group(GROUP_A, title="Retry uploads", desc="A 503 loses the file.")
        document = build_judge_tasks(
            items=[judge_item_from_group(group)], phase_dir=str(tmp_path)
        )
        prompt = tasks_of(document)[0]["prompt"]
        assert "Retry uploads" in prompt
        assert "A 503 loses the file." in prompt
        assert GROUP_A in prompt
        assert "Could go either way" in prompt

    def test_written_file_round_trips_through_the_reader(self, tmp_path):
        items = [judge_item_from_group(_group(GROUP_A))]
        path = write_judge_tasks(
            build_judge_tasks(items=items, phase_dir=str(tmp_path)), str(tmp_path)
        )
        assert Path(path).name == JUDGE_TASKS_FILENAME
        from utils.subagent_tasks import read_task_file

        document = read_task_file(path, phase_dir=str(tmp_path))
        assert tasks_of(document)[0]["group_id"] == GROUP_A

    def test_prompt_is_deterministic_for_the_same_item_set(self, tmp_path):
        items = [judge_item_from_group(_group(GROUP_A))]
        first = build_judge_tasks(items=items, phase_dir=str(tmp_path))
        second = build_judge_tasks(items=items, phase_dir=str(tmp_path))
        assert tasks_of(first)[0]["prompt"] == tasks_of(second)[0]["prompt"]
        assert first["inputs"] == second["inputs"]


class TestGroupIdOf:
    def test_prefers_the_stamped_hash(self):
        assert group_id_of({"group_hash": GROUP_A}) == GROUP_A

    def test_falls_back_to_group_id_then_computes(self):
        assert group_id_of({"group_id": GROUP_B}) == GROUP_B
        computed = group_id_of({"theme": "x", "suggestions": []})
        assert isinstance(computed, str) and computed


# ---------------------------------------------------------------------------
# Consumer: read + validate + map
# ---------------------------------------------------------------------------

class TestIngest:
    def test_maps_by_group_id_not_position(self, tmp_path):
        """A reordered batch must not misattribute a verdict."""
        _write_verdicts(tmp_path, 0, [
            _verdict(GROUP_B, DECISION_SKIP, "b is out"),
            _verdict(GROUP_A, DECISION_APPROVE, "a is in"),
        ])
        result = ingest_verdicts(
            phase_dir=str(tmp_path), expected_group_ids=[GROUP_A, GROUP_B]
        )
        assert result.ok
        assert result.verdicts[GROUP_A]["reason"] == "a is in"
        assert result.verdicts[GROUP_B]["reason"] == "b is out"

    def test_reads_multiple_files_in_deterministic_order(self, tmp_path):
        _write_verdicts(tmp_path, 0, [_verdict(GROUP_A, DECISION_APPROVE)])
        _write_verdicts(tmp_path, 1, [_verdict(GROUP_B, DECISION_SKIP)])
        files = [Path(p).name for p, _doc, _errs in read_verdict_files(str(tmp_path))]
        assert files == ["human_decision_batch_0.json", "human_decision_batch_1.json"]

    def test_schema_failure_names_the_file_and_the_index(self, tmp_path):
        path = tmp_path / "human_decision_batch_0.json"
        path.write_text(json.dumps({"schema_version": "1.0", "verdicts": [
            _verdict(GROUP_A, DECISION_APPROVE),
            {"group_id": GROUP_B, "decision": "maybe", "reason": "r"},
        ]}), encoding="utf-8")
        result = ingest_verdicts(
            phase_dir=str(tmp_path), expected_group_ids=[GROUP_A, GROUP_B]
        )
        assert not result.ok
        joined = " ".join(result.errors)
        assert str(path) in joined
        assert "verdicts[1]" in joined
        # A rejected file contributes nothing, so both groups stay unresolved.
        assert result.unresolved == [GROUP_A, GROUP_B]

    def test_unreadable_file_is_reported_not_crashed(self, tmp_path):
        (tmp_path / "human_decision_batch_0.json").write_text("{oops", encoding="utf-8")
        result = ingest_verdicts(phase_dir=str(tmp_path), expected_group_ids=[GROUP_A])
        assert not result.ok
        assert result.unresolved == [GROUP_A]

    def test_missing_verdict_stays_unresolved(self, tmp_path):
        """A group with no verdict is never silently approved or dropped."""
        _write_verdicts(tmp_path, 0, [_verdict(GROUP_A, DECISION_APPROVE)])
        result = ingest_verdicts(
            phase_dir=str(tmp_path),
            expected_group_ids=[GROUP_A, GROUP_B, GROUP_C],
        )
        assert result.ok
        assert result.unresolved == [GROUP_B, GROUP_C]

    def test_no_verdict_files_leaves_everything_unresolved(self, tmp_path):
        result = ingest_verdicts(
            phase_dir=str(tmp_path), expected_group_ids=[GROUP_A]
        )
        assert result.verdicts == {}
        assert result.unresolved == [GROUP_A]
        assert result.ok

    def test_identical_duplicate_verdict_first_wins(self, tmp_path):
        _write_verdicts(tmp_path, 0, [_verdict(GROUP_A, DECISION_APPROVE, "same")])
        _write_verdicts(tmp_path, 1, [_verdict(GROUP_A, DECISION_APPROVE, "same")])
        result = ingest_verdicts(
            phase_dir=str(tmp_path), expected_group_ids=[GROUP_A]
        )
        assert result.ok
        assert result.conflicts == []
        assert result.verdicts[GROUP_A]["reason"] == "same"

    def test_conflicting_duplicate_verdict_is_an_error(self, tmp_path):
        _write_verdicts(tmp_path, 0, [_verdict(GROUP_A, DECISION_APPROVE, "in")])
        _write_verdicts(tmp_path, 1, [_verdict(GROUP_A, DECISION_SKIP, "out")])
        result = ingest_verdicts(
            phase_dir=str(tmp_path), expected_group_ids=[GROUP_A]
        )
        assert not result.ok
        assert result.conflicts == [GROUP_A]
        assert any("repeats group_id" in e for e in result.errors)

    def test_out_of_scope_verdict_is_reported_and_ignored(self, tmp_path):
        _write_verdicts(tmp_path, 0, [
            _verdict(GROUP_A, DECISION_APPROVE),
            _verdict("dddd000000000000", DECISION_APPROVE),
        ])
        result = ingest_verdicts(
            phase_dir=str(tmp_path), expected_group_ids=[GROUP_A]
        )
        assert result.unexpected == ["dddd000000000000"]
        assert set(result.verdicts) == {GROUP_A}


# ---------------------------------------------------------------------------
# Recording — byte-identical to what an interactive pass produces
# ---------------------------------------------------------------------------

class TestRecording:
    def _state(self, tmp_path):
        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        return StateManager(plan)

    def test_approve_records_approved_with_the_plain_markers(self, tmp_path):
        state = self._state(tmp_path)
        _write_verdicts(tmp_path, 0, [_verdict(GROUP_A, DECISION_APPROVE, "sound")])
        ingest = ingest_verdicts(
            phase_dir=str(tmp_path), expected_group_ids=[GROUP_A]
        )
        record_verdicts(
            state=state,
            phase="apply-suggestions",
            ingest=ingest,
            items_by_group={GROUP_A: judge_item_from_group(_group(GROUP_A))},
        )
        record = state.get_human_decision("apply-suggestions", GROUP_A)
        assert record["decision"] == "approved"
        assert record["reason"] == "Claude auto-decide: sound"
        context = record["batch_context"]
        assert context["batch_action"] == "claude_auto_decide"
        assert context["decision_source"] == "claude_auto_decide"
        assert context["importance_at_decision"] == "MEDIUM"
        assert context["batch_id"].startswith("claude_decide_")
        assert "salvaged_description" not in context

    def test_skip_records_skipped(self, tmp_path):
        state = self._state(tmp_path)
        _write_verdicts(tmp_path, 0, [_verdict(GROUP_A, DECISION_SKIP, "nothing left")])
        ingest = ingest_verdicts(
            phase_dir=str(tmp_path), expected_group_ids=[GROUP_A]
        )
        record_verdicts(state=state, phase="apply-fixes", ingest=ingest,
                        items_by_group={})
        record = state.get_human_decision("apply-fixes", GROUP_A)
        assert record["decision"] == "skipped"
        assert record["batch_context"]["batch_action"] == "claude_auto_decide"

    def test_salvage_records_approved_with_the_salvage_markers(self, tmp_path):
        state = self._state(tmp_path)
        _write_verdicts(tmp_path, 0, [
            _verdict(GROUP_A, DECISION_SALVAGE, "partly",
                     salvaged="Only the retry.", dropped="the queue")
        ])
        ingest = ingest_verdicts(
            phase_dir=str(tmp_path), expected_group_ids=[GROUP_A]
        )
        record_verdicts(
            state=state,
            phase="apply-suggestions",
            ingest=ingest,
            items_by_group={
                GROUP_A: judge_item_from_group(_group(GROUP_A, importance="HIGH"))
            },
        )
        record = state.get_human_decision("apply-suggestions", GROUP_A)
        # decision stays "approved" — it approves the rewritten suggestion.
        assert record["decision"] == "approved"
        assert record["reason"] == (
            "Claude auto-decide (salvaged): kept Only the retry.; dropped the queue"
        )
        context = record["batch_context"]
        assert context["batch_action"] == "claude_auto_decide_salvage"
        assert context["decision_source"] == "claude_auto_decide_salvage"
        assert context["salvaged_description"] == "Only the retry."
        assert context["dropped"] == "the queue"
        assert context["importance_at_decision"] == "HIGH"

    def test_recorded_records_are_overlay_compatible(self, tmp_path):
        """The recorded shape is what the HTML report overlay already reads."""
        from utils.html_report_generator import _normalize_human_decision

        state = self._state(tmp_path)
        _write_verdicts(tmp_path, 0, [
            _verdict(GROUP_A, DECISION_SALVAGE, "partly"),
            _verdict(GROUP_B, DECISION_SKIP, "no"),
        ])
        ingest = ingest_verdicts(
            phase_dir=str(tmp_path), expected_group_ids=[GROUP_A, GROUP_B]
        )
        record_verdicts(state=state, phase="apply-suggestions", ingest=ingest,
                        items_by_group={})
        outcomes = {
            group_id: _normalize_human_decision(
                state.get_human_decision("apply-suggestions", group_id)
            )
            for group_id in (GROUP_A, GROUP_B)
        }
        assert outcomes[GROUP_A]["outcome"] == "salvaged"
        assert outcomes[GROUP_A]["claudeDecided"] is True
        assert outcomes[GROUP_B]["outcome"] == "skipped"

    def test_batch_id_is_stable_across_re_ingest(self, tmp_path):
        state = self._state(tmp_path)
        _write_verdicts(tmp_path, 0, [
            _verdict(GROUP_A, DECISION_APPROVE), _verdict(GROUP_B, DECISION_SKIP)
        ])
        ingest = ingest_verdicts(
            phase_dir=str(tmp_path), expected_group_ids=[GROUP_A, GROUP_B]
        )
        record_verdicts(state=state, phase="p", ingest=ingest, items_by_group={})
        first = state.get_human_decision("p", GROUP_A)["batch_context"]["batch_id"]
        record_verdicts(state=state, phase="p", ingest=ingest, items_by_group={})
        assert state.get_human_decision("p", GROUP_A)["batch_context"]["batch_id"] == first

    def test_verdicts_are_persisted_into_phase_state(self, tmp_path):
        """A crash between judging and applying resumes without re-judging."""
        state = self._state(tmp_path)
        _write_verdicts(tmp_path, 0, [
            _verdict(GROUP_A, DECISION_SALVAGE, "partly",
                     salvaged="Only the retry.", dropped="the queue")
        ])
        ingest = ingest_verdicts(
            phase_dir=str(tmp_path), expected_group_ids=[GROUP_A, GROUP_B]
        )
        record_verdicts(state=state, phase="apply-suggestions", ingest=ingest,
                        items_by_group={})
        stored = get_persisted_verdicts(state, "apply-suggestions")
        assert stored[GROUP_A]["salvaged_description"] == "Only the retry."
        assert stored[GROUP_A]["dropped"] == "the queue"
        assert state.state["human_decision_unresolved_apply-suggestions"] == [GROUP_B]

        state.save()
        reloaded = StateManager(tmp_path / "plan.md")
        assert get_persisted_verdicts(reloaded, "apply-suggestions")[GROUP_A][
            "decision"
        ] == DECISION_SALVAGE


# ---------------------------------------------------------------------------
# Work-set rewrite
# ---------------------------------------------------------------------------

class TestWorkSetRewrite:
    def test_approved_moves_into_the_apply_set(self):
        needs_human = [_group(GROUP_A)]
        valid, skipped = [], []
        counts = apply_verdicts_to_work_set(
            verdicts={GROUP_A: _verdict(GROUP_A, DECISION_APPROVE)},
            needs_human=needs_human, valid=valid, skipped=skipped,
        )
        assert counts == {"approved": 1, "salvaged": 0, "skipped": 0, "unresolved": 0}
        assert needs_human == []
        assert valid[0]["validation_status"] == "valid"
        assert valid[0]["auto_approved"] is True

    def test_skipped_moves_into_the_skip_set(self):
        needs_human = [_group(GROUP_A)]
        valid, skipped = [], []
        apply_verdicts_to_work_set(
            verdicts={GROUP_A: _verdict(GROUP_A, DECISION_SKIP)},
            needs_human=needs_human, valid=valid, skipped=skipped,
        )
        assert valid == []
        assert skipped[0]["validation_status"] == "skipped"

    def test_salvage_substitutes_the_rewrite_before_batching(self):
        """The apply prompt is built from the suggestion's desc, so the rewrite
        must land there — never the original."""
        needs_human = [_group(GROUP_A, desc="Original description")]
        valid, skipped = [], []
        apply_verdicts_to_work_set(
            verdicts={GROUP_A: _verdict(GROUP_A, DECISION_SALVAGE,
                                        salvaged="Only the retry.")},
            needs_human=needs_human, valid=valid, skipped=skipped,
        )
        group = valid[0]
        assert group["suggestions"][0]["desc"] == "Only the retry."
        assert "Original description" not in json.dumps(group)
        assert group["salvaged_description"] == "Only the retry."
        assert group["decision_source"] == "claude_auto_decide_salvage"

    def test_salvage_collapses_the_group_to_one_narrowed_suggestion(self):
        group = _group(GROUP_A)
        group["suggestions"].append(
            {"title": "second", "desc": "another", "importance": "LOW",
             "type": "addition"}
        )
        substitute_salvaged_description(group, "Only the retry.")
        assert len(group["suggestions"]) == 1
        assert group["suggestions"][0]["desc"] == "Only the retry."

    def test_a_group_with_no_verdict_stays_needs_human(self):
        needs_human = [_group(GROUP_A), _group(GROUP_B)]
        valid, skipped = [], []
        counts = apply_verdicts_to_work_set(
            verdicts={GROUP_A: _verdict(GROUP_A, DECISION_APPROVE)},
            needs_human=needs_human, valid=valid, skipped=skipped,
        )
        assert counts["unresolved"] == 1
        assert [g["group_hash"] for g in needs_human] == [GROUP_B]
        assert needs_human[0]["validation_status"] == "needs-human-decision"

    def test_the_work_set_is_rebuilt_from_decisions_not_the_snapshot(self):
        needs_human = [_group(GROUP_A), _group(GROUP_B), _group(GROUP_C)]
        valid, skipped = [], []
        counts = apply_verdicts_to_work_set(
            verdicts={
                GROUP_A: _verdict(GROUP_A, DECISION_APPROVE),
                GROUP_B: _verdict(GROUP_B, DECISION_SALVAGE),
                GROUP_C: _verdict(GROUP_C, DECISION_SKIP),
            },
            needs_human=needs_human, valid=valid, skipped=skipped,
        )
        assert counts == {"approved": 1, "salvaged": 1, "skipped": 1, "unresolved": 0}
        assert {g["group_hash"] for g in valid} == {GROUP_A, GROUP_B}
        assert [g["group_hash"] for g in skipped] == [GROUP_C]
        assert needs_human == []


# ---------------------------------------------------------------------------
# End-to-end through the apply orchestrator hook
# ---------------------------------------------------------------------------

class TestOrchestratorHook:
    """The apply orchestrators call the ingest after the judging phase."""

    def _orchestrator(self, tmp_path, claude_decide=True):
        import argparse

        from utils.apply_orchestrator_base import ApplyOrchestratorBase

        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        phase_dir = tmp_path / "phase"
        phase_dir.mkdir()

        class _Orchestrator(ApplyOrchestratorBase):
            phase_name = "apply-suggestions"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = False
            marks_phase_completed = False

            def get_output_path(self):
                return str(phase_dir / "orchestrator_output.json")

        args = argparse.Namespace(
            plan_file=str(plan), claude_decide=claude_decide, yes=False
        )
        orch = _Orchestrator(args)
        orch.plan_path = str(plan)
        orch.state = StateManager(plan)
        return orch, phase_dir

    def test_materializes_judging_tasks_for_in_scope_items(self, tmp_path, capsys):
        orch, phase_dir = self._orchestrator(tmp_path)
        orch.needs_human = [_group(GROUP_A), _group(GROUP_B)]
        orch._handle_human_decisions()
        out = capsys.readouterr().out
        assert "[HUMAN_DECISION_JUDGING_PENDING]" in out
        document = json.loads((phase_dir / JUDGE_TASKS_FILENAME).read_text())
        assert [t["group_id"] for t in tasks_of(document)] == [GROUP_A, GROUP_B]

    def test_scope_is_the_pre_marked_items_without_claude_decide(self, tmp_path):
        orch, phase_dir = self._orchestrator(tmp_path, claude_decide=False)
        marked = _group(GROUP_A)
        marked["claude_decide"] = True
        orch.needs_human = [marked, _group(GROUP_B)]
        orch._handle_human_decisions()
        document = json.loads((phase_dir / JUDGE_TASKS_FILENAME).read_text())
        assert [t["group_id"] for t in tasks_of(document)] == [GROUP_A]

    def test_nothing_is_materialized_when_no_item_is_in_scope(self, tmp_path):
        orch, phase_dir = self._orchestrator(tmp_path, claude_decide=False)
        orch.needs_human = [_group(GROUP_A)]
        orch._handle_human_decisions()
        assert not (phase_dir / JUDGE_TASKS_FILENAME).exists()

    def test_ingests_verdicts_and_rewrites_the_work_set(self, tmp_path):
        orch, phase_dir = self._orchestrator(tmp_path)
        orch.needs_human = [_group(GROUP_A), _group(GROUP_B)]
        _write_verdicts(phase_dir, 0, [
            _verdict(GROUP_A, DECISION_SALVAGE, salvaged="Only the retry."),
            _verdict(GROUP_B, DECISION_SKIP),
        ])
        orch._handle_human_decisions()

        assert [g["group_hash"] for g in orch.valid] == [GROUP_A]
        assert orch.valid[0]["suggestions"][0]["desc"] == "Only the retry."
        assert [g["group_hash"] for g in orch.skipped] == [GROUP_B]
        assert orch.needs_human == []
        # ... and the decisions are recorded, so --resume sees them.
        assert orch.state.get_human_decision("apply-suggestions", GROUP_A)[
            "decision"
        ] == "approved"
        assert orch.state.get_human_decision("apply-suggestions", GROUP_B)[
            "decision"
        ] == "skipped"
        # Nothing is left to judge, so no task file is re-materialized.
        assert not (phase_dir / JUDGE_TASKS_FILENAME).exists()

    def test_the_rebuilt_skip_list_marks_the_judge_as_the_source(self, tmp_path):
        """F2: the skip list is published beside its own count, and the judge's
        skip reached neither before the rebuild. The status word is derived from
        the `decision_source` this module stamps, so the two cannot drift."""
        orch, phase_dir = self._orchestrator(tmp_path)
        orch.needs_human = [_group(GROUP_A)]
        _write_verdicts(phase_dir, 0, [_verdict(GROUP_A, DECISION_SKIP)])
        orch._handle_human_decisions()
        orch._rebuild_formatted_skipped()

        assert len(orch.formatted_skipped) == len(orch.skipped) == 1
        entry = orch.formatted_skipped[0]
        assert entry["group_id"] == GROUP_A
        assert entry["status"] == "claude_skipped"
        assert entry["decision_source"] == "claude_auto_decide"
        assert entry["reason"].startswith("Claude auto-decide:")

    def test_unattended_run_skips_a_group_the_judge_did_not_cover(self, tmp_path, capsys):
        """Judging ran but missed one group: --yes skips it with a warning rather
        than stalling. Interactively it stays needs-human-decision."""
        orch, phase_dir = self._orchestrator(tmp_path)
        orch.args.yes = True
        orch.needs_human = [_group(GROUP_A), _group(GROUP_B)]
        _write_verdicts(phase_dir, 0, [_verdict(GROUP_A, DECISION_APPROVE)])
        orch._handle_human_decisions()

        assert [g["group_hash"] for g in orch.valid] == [GROUP_A]
        assert [g["group_hash"] for g in orch.skipped] == [GROUP_B]
        assert orch.needs_human == []
        assert "no judge verdict for group" in capsys.readouterr().err

    def test_interactive_run_keeps_an_uncovered_group_for_the_normal_ask(self, tmp_path):
        orch, phase_dir = self._orchestrator(tmp_path)
        orch.needs_human = [_group(GROUP_A), _group(GROUP_B)]
        _write_verdicts(phase_dir, 0, [_verdict(GROUP_A, DECISION_APPROVE)])
        orch._handle_human_decisions()

        assert [g["group_hash"] for g in orch.needs_human] == [GROUP_B]
        assert orch.skipped == []
        # ... and the leftover is re-materialized so it can still be judged.
        document = json.loads((phase_dir / JUDGE_TASKS_FILENAME).read_text())
        assert [t["group_id"] for t in tasks_of(document)] == [GROUP_B]

    def test_a_rejected_verdict_file_leaves_the_item_needing_a_decision(
        self, tmp_path, capsys
    ):
        orch, phase_dir = self._orchestrator(tmp_path)
        orch.needs_human = [_group(GROUP_A)]
        (phase_dir / "human_decision_batch_0.json").write_text(
            json.dumps({"schema_version": "1.0", "verdicts": [
                {"group_id": GROUP_A, "decision": "maybe", "reason": "r"}
            ]}),
            encoding="utf-8",
        )
        orch._handle_human_decisions()
        assert "verdict rejected" in capsys.readouterr().err
        assert [g["group_hash"] for g in orch.needs_human] == [GROUP_A]
        assert orch.valid == []
        # ... and the judging task is re-materialized so it can be re-run.
        assert (phase_dir / JUDGE_TASKS_FILENAME).exists()
