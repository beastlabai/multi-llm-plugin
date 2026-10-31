"""The scope-policy apply gate, deferrals, writer context and resume
(plans/scope-policy-and-report-tags.md §3, §5, §7).

Observable contracts, both legs: what each policy hands a writer, what it
defers and records, what a write prompt carries, how a policy change treats
earlier work, and how completion and the one blocker signal read to the
phases that follow. Validation fixtures are stamped with
``scope_policy.stamp_validation_record`` (the review side's own stamping).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

import apply_code_fixes_orchestrator as acf  # noqa: E402
import apply_suggestions_orchestrator as aso  # noqa: E402
import check_workflow_prerequisites as prereq  # noqa: E402
from utils import scope_apply  # noqa: E402
from utils import scope_policy as sp  # noqa: E402
from utils import write_prompt  # noqa: E402
from utils.apply_selection_helpers import merge_validation_with_groups  # noqa: E402
from utils.state_manager import StateManager, generate_group_id  # noqa: E402
from utils.schema_validator import validate_subagent_task_file  # noqa: E402

from tests.scope_apply_fixtures import (  # noqa: E402
    BID, CODE_CASES, PLAN_CASES, build_groups, build_validation, run_apply,
    write_code_review, write_review,
)
from tests.test_write_executor import FakeState, env  # noqa: E402,F401
from tests.test_write_executor_orchestrators import repo  # noqa: E402,F401

PHASE = "apply-suggestions"

#: What preserve hands a writer from the full plan fixture, and what it defers.
PRESERVE_ELIGIBLE = {"in_scope", "low_in_scope", "remedy", "judge_approve"}
PRESERVE_DEFERRED = {
    "extension", "unclear", "legacy", "malformed", "stale", "edited",
    "member_skip", "leaked", "judge_salvage", "judge_skip", "judge_extension",
}


def _phase_dir(plan, phase=PHASE):
    return plan.parent / plan.stem / phase


def _output(plan, phase=PHASE):
    return json.loads((_phase_dir(plan, phase) / "orchestrator_output.json").read_text())


def _tasks(plan, phase=PHASE):
    path = _phase_dir(plan, phase) / "write_tasks.json"
    return json.loads(path.read_text())["batches"] if path.exists() else []


def _state(plan):
    return json.loads((plan.parent / plan.stem / "state.json").read_text())


def _keys(ids, group_ids):
    by_id = {v: k for k, v in ids.items()}
    return {by_id[g] for g in group_ids if g in by_id}


def _handed_out(plan, ids, phase=PHASE):
    return _keys(ids, [i for task in _tasks(plan, phase) for i in task["expected_ids"]])


def _record(plan, review="review-plan"):
    return _state(plan)["scope_policy"][review]


# ===========================================================================
# The apply-time merge: authority only through the stable-hash join
# ===========================================================================


class TestTheMergeTransfersAuthorityOnlyByHash:

    def _fixture(self):
        cases = {k: PLAN_CASES[k] for k in ("in_scope", "extension")}
        groups, _ids = build_groups(cases)
        validation = build_validation(groups, cases)["groups"]
        return groups, validation

    def test_the_hash_join_attaches_the_stamped_verdict(self):
        groups, validation = self._fixture()
        merged = merge_validation_with_groups(groups, list(reversed(validation)))
        assert [sp.authority(g, BID).relation for g in merged] == ["in_scope", "scope_extension"]
        assert all(sp.authority(g, BID).authoritative for g in merged)

    def test_the_index_fallback_keeps_validity_but_never_scope(self):
        groups, validation = self._fixture()
        positional = [{k: v for k, v in r.items() if k != "group_id"} for r in validation]
        merged = merge_validation_with_groups(groups, positional)
        assert [g["validation_status"] for g in merged] == ["valid", "valid"]
        assert all(sp.VERDICT_KEY not in g for g in merged)
        assert not any(sp.scope_decision(g, "preserve", BID).eligible for g in merged)

    def test_a_mismatched_hash_cannot_carry_another_groups_verdict(self):
        """Shuffled ids: the in-scope verdict names a hash no group has, so
        no group receives it — not even the one at its index."""
        groups, validation = self._fixture()
        validation[0]["group_id"] = "0" * 16
        merged = merge_validation_with_groups(groups, validation)
        assert sp.VERDICT_KEY not in merged[0]
        assert merged[0]["validation_status"] == "valid"  # validity by index, as before
        assert sp.authority(merged[1], BID).relation == "scope_extension"

    def test_a_verdict_the_group_arrived_with_is_never_authority(self):
        groups, validation = self._fixture()
        forged = copy.deepcopy(groups)
        forged[1][sp.VERDICT_KEY] = {"relation": "in_scope", "baseline_id": BID,
                                     "proposal_digest": sp.proposal_digest(forged[1])}
        merged = merge_validation_with_groups(forged, [validation[0]])
        assert sp.VERDICT_KEY not in merged[1]

    def test_metadata_never_moves_the_stable_ids(self):
        groups, validation = self._fixture()
        merged = merge_validation_with_groups(groups, validation)
        assert [generate_group_id(g) for g in merged] == [generate_group_id(g) for g in groups]


# ===========================================================================
# Default vs override, native leg (materialization + util write-prompt)
# ===========================================================================


class TestPreserveOnTheNativeLeg:

    def test_preserve_hands_out_only_bound_in_scope_items_and_defers_the_rest(self, repo):
        review = write_review(repo.root, review_policy="preserve")
        plan, ids = review["plan"], review["ids"]
        assert run_apply(aso, plan, "--yes") == 0

        assert _handed_out(plan, ids) == PRESERVE_ELIGIBLE
        output = _output(plan)
        block = output[scope_apply.OUTPUT_KEY]
        assert block["policy"] == "preserve" and block["policy_source"] == "review"
        assert _keys(ids, [d["group_id"] for d in block["deferred_items"]]) == PRESERVE_DEFERRED
        # Deferrals are their own outcome: not skipped, not invalid, not processed.
        skipped = _keys(ids, [s["group_id"] for s in output["skipped_items"]])
        assert skipped == {"invalid"}
        assert {u["title"] for u in output["user_skipped_items"]} == {
            "Sort rows", "Escape quotes (and add XLSX)"}
        assert not _state(plan).get(f"processed_{PHASE}")
        recorded = _record(plan)["deferrals"]
        assert _keys(ids, [d["group_id"] for d in recorded]) == PRESERVE_DEFERRED
        assert all(d["baseline_id"] == BID and d["reason"] for d in recorded)

    def test_every_task_carries_its_binding_and_validates(self, repo):
        review = write_review(repo.root, review_policy="preserve")
        plan = review["plan"]
        assert run_apply(aso, plan, "--yes") == 0
        document = json.loads((_phase_dir(plan) / "write_tasks.json").read_text())
        assert validate_subagent_task_file(document) == []
        for task in document["batches"]:
            block = task[scope_apply.TASK_KEY]
            assert block["policy"] == "preserve" and block["baseline_id"] == BID
            assert {i["id"] for i in block["items"]} == set(task["expected_ids"])
            assert all(i["relation"] == "in_scope" for i in block["items"])

    def test_the_write_prompt_carries_the_writer_guidance(self, repo):
        review = write_review(repo.root, review_policy="preserve")
        plan = review["plan"]
        assert run_apply(aso, plan, "--yes") == 0
        for task in _tasks(plan):
            text, _t, _i = write_prompt.render_write_prompt(_phase_dir(plan), task["task_id"])
            assert "## Scope Policy: Preserve" in text
            assert "decline that item with the reason" in " ".join(text.split())
            assert scope_apply.BLOCKER_PREFIX in text
            assert BID in text

    def test_a_narrowed_remedy_is_the_instruction_and_the_extension_is_gone(self, repo):
        review = write_review(repo.root, review_policy="preserve")
        plan, ids = review["plan"], review["ids"]
        assert run_apply(aso, plan, "--yes") == 0
        task = next(t for t in _tasks(plan) if ids["remedy"] in t["expected_ids"])
        text, _t, _i = write_prompt.render_write_prompt(_phase_dir(plan), task["task_id"])
        assert "Write the CSV as UTF-8; nothing else." in text
        assert "settings page" not in text  # the discarded member's extension
        item = next(i for i in _output(plan)["to_apply"] if i["group_id"] == ids["remedy"])
        assert item["description"] == "Write the CSV as UTF-8; nothing else."
        assert "settings page" in item["scope"]["original_description"]  # evidence only

    def test_expand_override_of_the_same_review_hands_everything_valid_out(self, repo):
        review = write_review(repo.root, review_policy="preserve")
        plan, ids = review["plan"], review["ids"]
        assert run_apply(aso, plan, "--yes", scope_policy="expand") == 0
        handed = _handed_out(plan, ids)
        assert {"extension", "unclear", "legacy", "leaked", "judge_extension",
                "edited", "judge_salvage"} <= handed
        assert not handed & {"invalid", "user_skipped", "judge_skip"}
        for task in _tasks(plan):
            assert scope_apply.TASK_KEY not in task
            text, _t, _i = write_prompt.render_write_prompt(_phase_dir(plan), task["task_id"])
            assert "Scope Policy" not in text
        # The selected remedy is not used under expand: both members' text.
        item = next(i for i in _output(plan)["to_apply"] if i["group_id"] == ids["remedy"])
        assert "settings page" in item["description"]
        assert scope_apply.OUTPUT_KEY not in _output(plan)
        assert _record(plan)["apply_policy"] == "expand"

    def test_an_explicit_choice_survives_a_later_omission(self, repo):
        review = write_review(repo.root, review_policy="preserve")
        plan, ids = review["plan"], review["ids"]
        assert run_apply(aso, plan, "--yes", scope_policy="expand") == 0
        assert run_apply(aso, plan, "--yes") == 0  # no flag: the override holds
        assert "extension" in _handed_out(plan, ids)

    def test_preserve_on_an_expand_review_defers_unclassified_legacy_findings(self, repo):
        review = write_review(repo.root, review_policy=None)  # legacy round
        plan, ids = review["plan"], review["ids"]
        assert run_apply(aso, plan, "--yes", scope_policy="preserve") == 0
        deferred = _output(plan)[scope_apply.OUTPUT_KEY]["deferred_items"]
        legacy = next(d for d in deferred if d["group_id"] == ids["legacy"])
        assert "fresh classified review" in legacy["reason"]


class TestNoBypass:
    """Bulk approvals, --force, --no-confirm, --agent-decide and
    --accept-stale-consolidation cannot carry a leaked extension past an
    effective preserve gate; under expand the same extension is eligible."""

    FLAGS = ("--yes", "--approve-all", "--include-high", "--force", "--no-confirm",
             "--agent-decide", "--accept-stale-consolidation", "--approve-all-low",
             "--approve-importance", "HIGH", "--approve-validation-failed")

    def test_leaked_extension_dispatches_no_write_under_preserve(self, repo):
        review = write_review(repo.root, review_policy="preserve")
        plan, ids = review["plan"], review["ids"]
        assert run_apply(aso, plan, *self.FLAGS) == 0
        handed = _handed_out(plan, ids)
        assert not handed & {"leaked", "judge_extension", "extension"}
        assert {"leaked", "judge_extension"} <= _keys(
            ids, [d["group_id"] for d in _output(plan)[scope_apply.OUTPUT_KEY]["deferred_items"]])

    def test_the_same_leaked_extension_is_eligible_under_expand(self, repo):
        review = write_review(repo.root, review_policy="preserve")
        plan, ids = review["plan"], review["ids"]
        assert run_apply(aso, plan, *self.FLAGS, scope_policy="expand") == 0
        assert {"leaked", "judge_extension", "extension"} <= _handed_out(plan, ids)

    def test_an_invalid_scope_policy_value_is_exit_3_before_anything_runs(self, repo):
        review = write_review(repo.root)
        assert run_apply(aso, review["plan"], "--yes", scope_policy="restrict") == 3
        assert not (_phase_dir(review["plan"]) / "write_tasks.json").exists()


class TestPriorityStillApplies:

    def test_a_low_in_scope_fix_stays_eligible_at_the_default_floor(self, repo):
        review = write_review(repo.root, review_policy="preserve")
        assert run_apply(aso, review["plan"], "--yes") == 0
        assert "low_in_scope" in _handed_out(review["plan"], review["ids"])

    def test_an_explicit_higher_floor_still_excludes_it(self, repo):
        review = write_review(repo.root, review_policy="preserve")
        plan, ids = review["plan"], review["ids"]
        assert run_apply(aso, plan, "--yes", "--min-priority", "medium") == 0
        assert "low_in_scope" not in _handed_out(plan, ids)
        skipped = {s["group_id"]: s["status"] for s in _output(plan)["skipped_items"]}
        assert skipped[ids["low_in_scope"]] == "below_priority"  # not a deferral


class TestDryRunAndConfirmation:

    def test_the_dry_run_names_the_deferrals_and_writes_nothing(self, repo, capsys):
        review = write_review(repo.root, review_policy="preserve")
        plan = review["plan"]
        before = _record(plan)
        capsys.readouterr()
        assert run_apply(aso, plan, "--yes", "--dry-run") == 0
        err = capsys.readouterr().err
        assert "Would defer by --scope-policy preserve (11, not applied):" in err
        assert "Deferred (scope policy preserve): 10" in err  # before the judge
        assert _record(plan) == before

    def test_the_confirmation_describes_the_eligible_set(self, repo, capsys):
        review = write_review(repo.root, review_policy="preserve", selections=False,
                              verdicts=False)
        capsys.readouterr()
        assert run_apply(aso, review["plan"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "confirmation_needed"
        assert "ALL valid items validated as within scope will be applied" in payload["message"]
        # extension, unclear, legacy, malformed, stale and leaked (no edit
        # or member skip yet: those two are still bound).
        assert "6 valid item(s) are deferred by --scope-policy preserve" in payload["message"]


# ===========================================================================
# The unattended executor leg: mocked end to end
# ===========================================================================

E2E = {k: PLAN_CASES[k] for k in ("in_scope", "extension", "invalid", "user_skipped",
                                  "leaked")}


def _run_executor(repo, plan, attest, *flags, scope_policy=None, declined=None,
                  edits=True, result=None):
    repo.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
        {"path": f"plans/{plan.name}", "mode": "patch", "content": "edit\n"}] if edits else []))
    if result is not None:
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_RESULT", json.dumps(result))
    else:
        repo.monkeypatch.delenv("MOCK_LLM_WRITE_RESULT", raising=False)
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(sorted(attest)))
    if declined is not None:
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_DECLINED", json.dumps(declined))
    else:
        repo.monkeypatch.delenv("MOCK_LLM_WRITE_DECLINED", raising=False)
    return run_apply(aso, plan, "--yes", "--exec-subagents", *flags,
                     scope_policy=scope_policy)


def _applied(plan, ids, phase=PHASE):
    processed = _state(plan).get(f"processed_{phase}") or {}
    return _keys(ids, [k for k, r in processed.items() if r.get("status") in ("applied", "no-op")])


def _dispatches(plan, phase=PHASE):
    path = _phase_dir(plan, phase) / "subagent_audit.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r.get("task_id") for r in rows if r.get("event") == "dispatched"]


class TestTheExecutorLeg:

    def test_no_option_applies_a_valid_extension(self, repo):
        review = write_review(repo.root, cases=E2E, review_policy="expand", verdicts=False)
        plan, ids = review["plan"], review["ids"]
        assert _run_executor(repo, plan, ids.values()) == 0
        assert _applied(plan, ids) == {"in_scope", "extension", "leaked"}

    def test_preserve_applies_in_scope_and_defers_extension_and_unclear(self, repo):
        cases = dict(E2E, unclear=PLAN_CASES["unclear"], legacy=PLAN_CASES["legacy"])
        review = write_review(repo.root, cases=cases, review_policy="preserve", verdicts=False)
        plan, ids = review["plan"], review["ids"]
        assert _run_executor(repo, plan, ids.values()) == 0
        assert _applied(plan, ids) == {"in_scope"}
        assert _keys(ids, [d["group_id"] for d in _record(plan)["deferrals"]]) == {
            "extension", "leaked", "unclear", "legacy"}
        # Eligible work completed, deferrals kept: the phase is finished, and
        # its dependents read it so (no repeated "unapplied" count, no loop).
        state = StateManager(plan)
        assert state.is_phase_completed(PHASE)
        assert _record(plan)[scope_apply.COMPLETION_KEY]["outcome"] == "eligible work completed"
        assert _record(plan)[scope_apply.COMPLETION_KEY]["deferred"] == 4
        assert prereq.check_apply_suggestions_prerequisite(
            state, plan.parent / plan.stem)["met"] is True
        assert prereq.collect_missing(state, plan.parent / plan.stem, "generate-tasks") == []
        summary = (_phase_dir(plan) / "summary.md").read_text()
        assert "## Deferred by scope policy (4, not applied)" in summary
        status = prereq.get_workflow_status(state, plan, plan.parent / plan.stem)
        assert status["phases"][PHASE]["scope_deferred"] == 4

    def test_invalid_and_user_skipped_stay_excluded_under_both(self, repo):
        for policy in ("expand", "preserve"):
            review = write_review(repo.root, cases=E2E, review_policy=policy,
                                  verdicts=False, plan_name=f"both-{policy}")
            plan, ids = review["plan"], review["ids"]
            assert _run_executor(repo, plan, ids.values()) == 0
            assert not _applied(plan, ids) & {"invalid", "user_skipped"}

    def test_a_completed_preserve_apply_releases_its_deferrals_under_expand(self, repo, capsys):
        """The override offers the deferred items again — not an "already
        complete" no-op — and never dispatches the landed batch a second time."""
        review = write_review(repo.root, cases=E2E, review_policy="preserve", verdicts=False)
        plan, ids = review["plan"], review["ids"]
        assert _run_executor(repo, plan, ids.values()) == 0
        first = _dispatches(plan)
        assert _applied(plan, ids) == {"in_scope"}
        landed_task = next(t["task_id"] for t in _tasks(plan))

        capsys.readouterr()
        assert _run_executor(repo, plan, ids.values(), scope_policy="expand") == 0
        err = capsys.readouterr().err
        assert "[SCOPE_POLICY_CHANGED]" in err
        assert _applied(plan, ids) == {"in_scope", "extension", "leaked"}
        second = _dispatches(plan)[len(first):]
        assert landed_task not in second  # no batch id renumbered onto landed work
        handed = _handed_out(plan, ids)
        assert "in_scope" not in handed  # the landed item is not handed out again
        assert _record(plan)["deferrals"] == []
        output = _output(plan)
        assert output[scope_apply.OUTPUT_KEY]["released_count"] == 2

    #: Two in-scope items on one section (one batch): one applied, one declined.
    PAIR = {"in_scope": PLAN_CASES["in_scope"], "edited": PLAN_CASES["edited"],
            "extension": PLAN_CASES["extension"]}

    def _decline_run(self, repo, reason):
        review = write_review(repo.root, cases=self.PAIR, review_policy="preserve",
                              verdicts=False, selections=False)
        plan, ids = review["plan"], review["ids"]
        code = _run_executor(repo, plan, [ids["in_scope"]],
                             declined=[{"id": ids["edited"], "reason": reason}])
        return plan, ids, code

    def test_a_preserve_writer_blocker_stops_and_dependents_refuse(self, repo, capsys):
        reason = f"{scope_apply.BLOCKER_PREFIX} the CSV total needs a new aggregation service"
        capsys.readouterr()
        plan, ids, code = self._decline_run(repo, reason)
        assert code == 3
        err = capsys.readouterr().err
        assert "[SCOPE_BLOCKER]" in err
        record = _record(plan)
        assert record[scope_apply.COMPLETION_KEY]["outcome"] == "blocked"
        assert record[scope_apply.BLOCKERS_KEY][0]["group_id"] == ids["edited"]
        state = StateManager(plan)
        missing = prereq.collect_missing(state, plan.parent / plan.stem, "generate-tasks")
        assert [m["phase"] for m in missing] == ["apply-suggestions"]
        assert missing[0].get("scope_blocked")
        message = prereq.build_unmet_message("generate-tasks", plan, plan.parent / plan.stem, missing)
        assert "scope blocker" in message and "--scope-policy expand" in message
        status = prereq.get_workflow_status(state, plan, plan.parent / plan.stem)
        assert "scope blocker" in status["phases"][PHASE]["scope_blocker"]
        # The way out the blocker names: an expand apply clears it.
        assert _run_executor(repo, plan, ids.values(), scope_policy="expand") == 0
        assert scope_apply.recorded_blockers(_state(plan), PHASE) == []
        assert prereq.collect_missing(
            StateManager(plan), plan.parent / plan.stem, "generate-tasks") == []

    def test_an_ordinary_decline_is_not_a_blocker(self, repo):
        plan, _ids, code = self._decline_run(repo, "the plan already says this")
        assert code == 0
        assert scope_apply.recorded_blockers(_state(plan), PHASE) == []
        assert _record(plan)[scope_apply.COMPLETION_KEY]["outcome"] == "eligible work completed"


# ===========================================================================
# Resume and stale tasks (both dispatch paths)
# ===========================================================================


class TestStaleTasksAreRefused:

    def test_native_write_prompt_refuses_a_legacy_task_once_preserve_is_effective(self, repo):
        """A task file materialized under expand (as one from before the
        feature) cannot acquire authority from a newly recorded policy."""
        review = write_review(repo.root, review_policy="expand")
        plan = review["plan"]
        assert run_apply(aso, plan, "--yes") == 0
        state = StateManager(plan)
        sp.record_apply_policy(state.state, PHASE, "preserve", "explicit")
        state.save()
        for task in _tasks(plan):
            with pytest.raises(write_prompt.WritePromptError) as caught:
                write_prompt.render_write_prompt(_phase_dir(plan), task["task_id"])
            assert "materialized under --scope-policy expand" in str(caught.value)
            assert write_prompt.main(["--phase-dir", str(_phase_dir(plan)),
                                      "--task-id", task["task_id"], "--stdout"]) == 3

    def test_native_write_prompt_refuses_a_preserve_task_under_expand(self, repo):
        review = write_review(repo.root, review_policy="preserve")
        plan = review["plan"]
        assert run_apply(aso, plan, "--yes") == 0
        state = StateManager(plan)
        sp.record_apply_policy(state.state, PHASE, "expand", "explicit")
        state.save()
        task = _tasks(plan)[0]
        with pytest.raises(write_prompt.WritePromptError) as caught:
            write_prompt.render_write_prompt(_phase_dir(plan), task["task_id"])
        assert "wrong policy" in str(caught.value)

    def test_a_tampered_binding_is_refused(self, repo):
        review = write_review(repo.root, review_policy="preserve")
        plan = review["plan"]
        assert run_apply(aso, plan, "--yes") == 0
        path = _phase_dir(plan) / "write_tasks.json"
        document = json.loads(path.read_text())
        document["batches"][0]["scope"]["items"][0]["relation"] = "in_scope"
        document["batches"][0]["scope"]["items"][0]["reason"] = "edited by hand"
        path.write_text(json.dumps(document))
        with pytest.raises(write_prompt.WritePromptError) as caught:
            write_prompt.render_write_prompt(_phase_dir(plan), document["batches"][0]["task_id"])
        assert "binding" in str(caught.value)

    def test_missing_scope_metadata_never_blocks_an_expand_dispatch(self, env):
        state = FakeState(env.tmp_path / "state.json")
        task = env.apply_task(0, expected_ids=["s0"])
        assert scope_apply.stale_task_refusal(task, "expand") is None
        assert scope_apply.stale_task_refusal(task, None) is None

    def test_the_executor_refuses_a_legacy_task_under_preserve(self, env):
        state = FakeState(env.tmp_path / "state.json")
        sp.record_review_round(state.state, "review-plan", "preserve", "explicit",
                               {"baseline_id": BID})
        state.save()
        calls = []

        def respond(task, prompt):  # pragma: no cover - must not be reached
            calls.append(task["task_id"])
            return {"status": "applied", "ids": ["s0"], "files_touched": [], "summary": "x"}

        result = env.scheduler([env.apply_task(0, expected_ids=["s0"])],
                               respond=respond, state=state).run()
        assert calls == []
        assert result.exit_code == 3
        reasons = " ".join(str(o.reason) for o in result.outcomes)
        assert "materialized under --scope-policy expand" in reasons

    def test_the_executor_dispatches_a_legacy_task_under_expand(self, env):
        state = FakeState(env.tmp_path / "state.json")

        def respond(task, prompt):
            assert "Scope Policy" not in prompt
            env.plan.write_text(env.plan.read_text() + "x\n")
            return {"status": "applied", "ids": ["s0"], "files_touched": ["plans/p.md"],
                    "summary": "x"}

        result = env.scheduler([env.apply_task(0, expected_ids=["s0"])],
                               respond=respond, state=state).run()
        assert result.exit_code == 0, result.messages


class TestPolicyChangeOverEarlierWork:

    def test_no_started_work_retires_the_old_tasks_and_rematerializes(self, repo, capsys):
        review = write_review(repo.root, review_policy="preserve")
        plan, ids = review["plan"], review["ids"]
        assert run_apply(aso, plan, "--yes") == 0
        capsys.readouterr()
        assert run_apply(aso, plan, "--yes", scope_policy="expand") == 0
        assert "[SCOPE_POLICY_CHANGED]" in capsys.readouterr().err
        retired = list(_phase_dir(plan).glob("write_tasks.retired-preserve-*.json"))
        assert len(retired) == 1
        assert all(scope_apply.TASK_KEY not in t for t in _tasks(plan))
        assert _tasks(plan)[0]["batch_index"] == 0

    def test_native_completed_and_pending_batches_keep_their_ids(self, repo):
        """Native resume reads processed records: the batch the host recorded
        stays done, and the rebuilt batches never take its id."""
        from utils import batch_done

        review = write_review(repo.root, review_policy="preserve")
        plan, ids = review["plan"], review["ids"]
        assert run_apply(aso, plan, "--yes") == 0
        first = _tasks(plan)
        done = first[0]
        assert batch_done.main(["--phase-dir", str(_phase_dir(plan)),
                                "--task-id", done["task_id"], "--status", "completed"]) == 0

        assert run_apply(aso, plan, "--yes", scope_policy="expand") == 0
        second = _tasks(plan)
        assert not set(done["expected_ids"]) & {i for t in second for i in t["expected_ids"]}
        assert min(t["batch_index"] for t in second) > max(t["batch_index"] for t in first)
        assert {"extension", "leaked"} <= _handed_out(plan, ids)

    def test_native_resume_under_the_same_policy_drops_recorded_batches(self, repo):
        from utils import batch_done

        review = write_review(repo.root, review_policy="preserve")
        plan, ids = review["plan"], review["ids"]
        assert run_apply(aso, plan, "--yes") == 0
        done = _tasks(plan)[0]
        assert batch_done.main(["--phase-dir", str(_phase_dir(plan)),
                                "--task-id", done["task_id"], "--status", "completed"]) == 0
        assert run_apply(aso, plan, "--yes", "--resume") == 0
        remaining = _handed_out(plan, ids)
        landed = _keys(ids, done["expected_ids"])
        assert not remaining & landed
        assert PRESERVE_ELIGIBLE - landed <= remaining
        assert all(t["scope"]["policy"] == "preserve" for t in _tasks(plan))

    def test_an_unfinished_batch_under_a_policy_change_is_exit_3_with_a_way_out(self, repo, capsys):
        review = write_review(repo.root, cases=E2E, review_policy="preserve", verdicts=False)
        plan, ids = review["plan"], review["ids"]
        # Run 1: the runner fails the batch (nothing moved): a `failed` record.
        code = _run_executor(repo, plan, [], edits=False, result={
            "status": "failed", "ids": [], "files_touched": [], "summary": "could not"})
        processed = _state(plan).get(f"processed_{PHASE}") or {}
        assert any(r.get("status") == "failed" for r in processed.values()), processed
        assert code == 3
        capsys.readouterr()
        assert _run_executor(repo, plan, ids.values(), scope_policy="expand") == 3
        err = capsys.readouterr().err
        assert "did not finish" in err and "--resume --scope-policy preserve" in err
        assert "util restore-batch" in err


# ===========================================================================
# apply-code-fixes: format_fix_for_output under a selected remedy
# ===========================================================================


class TestCodeFixRemedy:

    def test_preserve_sends_the_selected_remedy_only(self, repo):
        review = write_code_review(repo.root, review_policy="preserve")
        plan, ids = review["plan"], review["ids"]
        assert run_apply(acf, plan, "--yes", "--base-ref", "HEAD") == 0
        tasks = _tasks(plan, "apply-fixes")
        handed = _keys(ids, [i for t in tasks for i in t["expected_ids"]])
        assert handed == {"in_scope", "remedy"}
        task = next(t for t in tasks if ids["remedy"] in t["expected_ids"])
        text, _t, _i = write_prompt.render_write_prompt(_phase_dir(plan, "apply-fixes"),
                                                        task["task_id"])
        assert "Pass encoding='utf-8' to open(); add no setting." in text
        assert "encoding setting to the config" not in text
        assert "## Scope Policy: Preserve" in text

    def test_expand_concatenates_as_before(self, repo):
        review = write_code_review(repo.root, review_policy="preserve")
        plan, ids = review["plan"], review["ids"]
        assert run_apply(acf, plan, "--yes", "--base-ref", "HEAD", scope_policy="expand") == 0
        item = next(i for i in _output(plan, "apply-fixes")["to_apply"]
                    if i["group_id"] == ids["remedy"])
        assert "encoding setting to the config" in item["description"]
        assert "scope" not in item

    def test_format_fix_for_output_keeps_its_default(self):
        group = {"theme": "t", "suggestions": [{"desc": "a"}, {"desc": "b"}]}
        assert acf.format_fix_for_output(group, 0)["description"] == "a\n\nb"
        assert acf.format_fix_for_output(group, 0, selected_remedy="only a")["description"] == "only a"


class TestAllDeferred:

    def test_nothing_eligible_completes_the_phase_with_its_deferrals(self, repo):
        cases = {k: PLAN_CASES[k] for k in ("extension", "unclear")}
        review = write_review(repo.root, cases=cases, review_policy="preserve", verdicts=False,
                              selections=False)
        plan, ids = review["plan"], review["ids"]
        assert run_apply(aso, plan, "--yes") == 0
        output = _output(plan)
        assert output["status"] == "completed"
        assert output["summary"]["scope_deferred_count"] == 2
        state = StateManager(plan)
        assert state.is_phase_completed(PHASE)
        assert _record(plan)[scope_apply.COMPLETION_KEY]["outcome"] == "eligible work completed"
        assert prereq.collect_missing(state, plan.parent / plan.stem, "generate-tasks") == []


# ===========================================================================
# apply-task-suggestions: the same gate
# ===========================================================================

TASKS_MD = """# Tasks

### Task T001: Write the CSV
**Description**: Write the weekly CSV.
**Files to modify**: src/main.py
**Files to create**: none
**Complexity**: low
**Executor**: agent
**Depends on**: none

### Task T002: Test the CSV
**Description**: Test the weekly CSV.
**Files to modify**: tests/test_main.py
**Files to create**: none
**Complexity**: low
**Executor**: agent
**Depends on**: T001
"""


def _task_review(root, review_policy):
    import apply_task_suggestions_orchestrator  # noqa: F401

    from utils.state_manager import get_or_create_state

    cases = {
        "in_scope": ([{"title": "Tighten T001", "desc": "Name the CSV columns in T001.",
                       "type": "modification", "reference": "T001", "importance": "MEDIUM"}],
                     "valid", ("bound", "in_scope")),
        "extension": ([{"title": "Add T003", "desc": "Add a task for a PDF export.",
                        "type": "addition", "reference": "T002", "importance": "LOW"}],
                      "valid", ("bound", "scope_extension")),
    }
    plan = root / "plans" / "tasks.md"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("# Plan\n\n## Goal\nWeekly CSV.\n", encoding="utf-8")
    out = root / "plans" / "tasks"
    (out / "tasks").mkdir(parents=True, exist_ok=True)
    (out / "tasks" / "tasks.md").write_text(TASKS_MD, encoding="utf-8")
    review = out / "review-tasks"
    review.mkdir(parents=True, exist_ok=True)
    groups, ids = build_groups(cases)
    (review / "grouped.json").write_text(json.dumps({"format_version": 2, "groups": groups}))
    (review / "validation.json").write_text(json.dumps(build_validation(groups, cases)))
    state = get_or_create_state(plan)
    for phase in ("review-plan", "apply-suggestions", "generate-tasks", "review-tasks"):
        state.mark_phase_completed(phase)
    if review_policy:
        sp.record_review_round(state.state, "review-tasks", review_policy, "explicit",
                               {"baseline_id": BID})
    state.save()
    return plan, ids


class TestApplyTaskSuggestions:

    @pytest.mark.parametrize("review_policy,expected", [
        ("preserve", {"in_scope"}), ("expand", {"in_scope", "extension"})])
    def test_the_gate_holds_for_the_task_list(self, repo, review_policy, expected):
        import apply_task_suggestions_orchestrator as ats

        plan, ids = _task_review(repo.root, review_policy)
        assert run_apply(ats, plan, "--yes") == 0
        handed = _keys(ids, [i for t in _tasks(plan, "apply-task-suggestions")
                             for i in t["expected_ids"]])
        assert handed == expected
        if review_policy == "preserve":
            assert _record(plan, "review-tasks")["deferrals"][0]["group_id"] == ids["extension"]


# ===========================================================================
# The composites' "already complete" guards
# ===========================================================================


class TestOverrideReleasesTheCompositeGuard:

    def _records(self, **over):
        from utils import improve_workflow as iw

        base = dict(mode="improve-plan", review_completed=True, review_on_record=True,
                    apply_word="completed", apply_word_at="2026-10-31T10:00:00")
        base.update(over)
        return iw.Records(**base)

    def _args(self, policy=None):
        import argparse

        return argparse.Namespace(force=False, scope_policy=policy)

    def test_an_explicit_expand_runs_the_apply_step_again(self):
        from utils import improve_workflow as iw

        records = self._records(scope_deferrals=2)
        assert iw.skip_reason("improve-plan", "apply-suggestions", self._args("expand"),
                              records) is None
        assert not iw.nothing_to_do("improve-plan", self._args("expand"), records)

    def test_omission_and_preserve_keep_the_guard(self):
        from utils import improve_workflow as iw

        records = self._records(scope_deferrals=2)
        for policy in (None, "preserve"):
            assert iw.skip_reason("improve-plan", "apply-suggestions", self._args(policy),
                                  records) == "completed"
        assert iw.skip_reason("improve-plan", "apply-suggestions", self._args("expand"),
                              self._records()) == "completed"

    def test_read_records_counts_the_rounds_deferrals(self, repo):
        from utils import improve_workflow as iw

        review = write_review(repo.root, cases=E2E, review_policy="preserve", verdicts=False)
        plan = review["plan"]
        assert run_apply(aso, plan, "--yes") == 0
        records = iw.read_records("improve-plan", plan, self._args())
        assert records.scope_deferrals == 2

    def test_the_release_predicate_is_for_apply_phases_only(self):
        state = {}
        sp.record_review_round(state, "review-plan", "preserve", "explicit")
        sp.record_deferrals(state, "review-plan", [{"group_id": "g"}])
        assert scope_apply.releases_deferrals(state, "apply-suggestions", "expand")
        assert not scope_apply.releases_deferrals(state, "apply-suggestions", None)
        assert not scope_apply.releases_deferrals(state, "review-plan", "expand")


class TestTheStateRecordValidates:

    def test_a_state_with_the_scope_policy_record_validates(self, repo):
        from utils.schema_validator import validate_state_file

        review = write_review(repo.root, cases=E2E, review_policy="preserve", verdicts=False)
        plan = review["plan"]
        assert run_apply(aso, plan, "--yes", scope_policy="preserve") == 0
        ok, errors = validate_state_file(_state(plan))
        assert ok, errors
        assert _state(plan)["scope_policy"]["review-plan"]["deferrals"]


class TestTheNativeLegCloses:

    def test_mark_completed_records_eligible_work_completed(self, repo):
        from utils import batch_done

        review = write_review(repo.root, cases=E2E, review_policy="preserve", verdicts=False)
        plan = review["plan"]
        assert run_apply(aso, plan, "--yes") == 0
        for task in _tasks(plan):
            assert batch_done.main(["--phase-dir", str(_phase_dir(plan)), "--task-id",
                                    task["task_id"], "--status", "completed"]) == 0
        assert run_apply(aso, plan, "--mark-completed") == 0
        assert _record(plan)[scope_apply.COMPLETION_KEY]["outcome"] == "eligible work completed"
        assert len(_record(plan)["deferrals"]) == 2

    def test_a_native_blocker_decline_stops_mark_completed_at_exit_3(self, repo, capsys):
        from utils import batch_done

        pair = TestTheExecutorLeg.PAIR
        review = write_review(repo.root, cases=pair, review_policy="preserve",
                              verdicts=False, selections=False)
        plan, ids = review["plan"], review["ids"]
        assert run_apply(aso, plan, "--yes") == 0
        task = next(t for t in _tasks(plan) if ids["edited"] in t["expected_ids"])
        reason = f"{scope_apply.BLOCKER_PREFIX} needs a reporting service the plan excludes"
        assert batch_done.main(["--phase-dir", str(_phase_dir(plan)), "--task-id",
                                task["task_id"], "--status", "completed",
                                "--declined", f"{ids['edited']}={reason}"]) == 0
        capsys.readouterr()
        assert run_apply(aso, plan, "--mark-completed") == 3
        assert "[SCOPE_BLOCKER]" in capsys.readouterr().err
        assert scope_apply.recorded_blockers(_state(plan), PHASE)[0]["group_id"] == ids["edited"]


# ===========================================================================
# A blocker holds until it is settled; a policy change keeps landed work
# applied; a refused override changes nothing
# ===========================================================================


def _native_blocked(repo, cases):
    """A native preserve run whose batch declined ``edited`` with a blocker,
    closed by ``--mark-completed`` (exit 3, ``blocked``)."""
    from utils import batch_done

    review = write_review(repo.root, cases=cases, review_policy="preserve",
                          verdicts=False, selections=False)
    plan, ids = review["plan"], review["ids"]
    assert run_apply(aso, plan, "--yes") == 0
    reason = f"{scope_apply.BLOCKER_PREFIX} needs a reporting service the plan excludes"
    for task in _tasks(plan):
        declined = []
        if ids["edited"] in task["expected_ids"]:
            declined = ["--declined", f"{ids['edited']}={reason}"]
        assert batch_done.main(["--phase-dir", str(_phase_dir(plan)), "--task-id",
                                task["task_id"], "--status", "completed", *declined]) == 0
    assert run_apply(aso, plan, "--mark-completed") == 3
    assert scope_apply.recorded_blockers(_state(plan), PHASE)[0]["group_id"] == ids["edited"]
    return plan, ids


class TestABlockerHoldsUntilSettled:

    @pytest.mark.parametrize("cases", [
        # The remaining extension stays deferred: the all-deferred exit.
        TestTheExecutorLeg.PAIR,
        # Nothing deferred either: the emit-time empty loop.
        {k: PLAN_CASES[k] for k in ("in_scope", "edited")},
    ], ids=["all-deferred-exit", "empty-loop"])
    def test_a_native_resume_over_a_blocked_phase_stays_blocked(self, repo, capsys, cases):
        """`--resume` drops the declined item (its record counts as
        processed) and hands out nothing; its completion still derives the
        blocker from the phase's records, so the phase stays blocked."""
        plan, ids = _native_blocked(repo, cases)
        capsys.readouterr()
        assert run_apply(aso, plan, "--yes", "--resume") == 3
        assert "[SCOPE_BLOCKER]" in capsys.readouterr().err
        record = _record(plan)
        assert record[scope_apply.COMPLETION_KEY]["outcome"] == "blocked"
        assert scope_apply.recorded_blockers(_state(plan), PHASE)[0]["group_id"] == ids["edited"]
        missing = prereq.collect_missing(StateManager(plan), plan.parent / plan.stem,
                                         "generate-tasks")
        assert [m["phase"] for m in missing] == [PHASE] and missing[0]["scope_blocked"]
        # The way out it names still works: an expand apply clears it — at
        # its completion, the native host's `--mark-completed`; until then
        # the phase stays blocked.
        from utils import batch_done

        assert run_apply(aso, plan, "--yes", scope_policy="expand") == 0
        assert scope_apply.recorded_blockers(_state(plan), PHASE)
        assert ids["edited"] in {i for t in _tasks(plan) for i in t["expected_ids"]}
        for task in _tasks(plan):
            assert batch_done.main(["--phase-dir", str(_phase_dir(plan)), "--task-id",
                                    task["task_id"], "--status", "completed"]) == 0
        assert run_apply(aso, plan, "--mark-completed") == 0
        assert scope_apply.recorded_blockers(_state(plan), PHASE) == []
        assert prereq.collect_missing(StateManager(plan), plan.parent / plan.stem,
                                      "generate-tasks") == []

    def test_a_resume_with_its_own_no_items_exit_stays_blocked(self, repo, capsys):
        """apply-code-fixes' own nothing-left exit (record completed, exit 0)
        must not stand in for the scope completion of a blocked phase."""
        from utils import batch_done

        cases = {k: CODE_CASES[k] for k in ("in_scope", "remedy")}
        review = write_code_review(repo.root, cases=cases, review_policy="preserve")
        plan, ids = review["plan"], review["ids"]
        assert run_apply(acf, plan, "--yes", "--base-ref", "HEAD") == 0
        phase_dir = _phase_dir(plan, "apply-fixes")
        reason = f"{scope_apply.BLOCKER_PREFIX} the fix needs a new config option"
        for task in _tasks(plan, "apply-fixes"):
            declined = []
            if ids["remedy"] in task["expected_ids"]:
                declined = ["--declined", f"{ids['remedy']}={reason}"]
            assert batch_done.main(["--phase-dir", str(phase_dir), "--task-id",
                                    task["task_id"], "--status", "completed", *declined]) == 0
        assert run_apply(acf, plan, "--mark-completed") == 3
        capsys.readouterr()
        assert run_apply(acf, plan, "--yes", "--resume", "--base-ref", "HEAD") == 3
        assert "[SCOPE_BLOCKER]" in capsys.readouterr().err
        assert scope_apply.recorded_blockers(_state(plan), "apply-code-fixes")[0][
            "group_id"] == ids["remedy"]

    def test_only_a_completion_clears_a_blocked_record(self):
        """A gate run's new account of the deferrals keeps a `blocked`
        completion (a run that stops before completing leaves the phase
        blocked); it still drops any other stale completion."""
        state = {}
        sp.record_review_round(state, "review-plan", "preserve", "explicit")
        applied = scope_apply.resolve_apply_policy(None, state, PHASE)
        blocker = {"group_id": "g1", "task_id": "apply-batch-0", "reason": "r"}
        scope_apply.record_completion(state, PHASE, applied, [blocker])
        scope_apply.persist_deferrals(state, PHASE, [])
        assert scope_apply.recorded_blockers(state, PHASE) == [blocker]
        scope_apply.record_completion(state, PHASE, applied, [])
        assert scope_apply.recorded_blockers(state, PHASE) == []
        sp.record_deferrals(state, PHASE, [{"group_id": "g2"}])
        scope_apply.record_completion(state, PHASE, applied, [])
        scope_apply.persist_deferrals(state, PHASE, [])
        assert scope_apply.COMPLETION_KEY not in state["scope_policy"]["review-plan"]


class TestTheBlockerIsAnApplyOutcome:
    """The record is keyed by the review round, but only the apply phase is
    blocked: the apply that must run again to clear it is not refused."""

    def test_the_apply_that_clears_it_is_not_blocked_by_its_own_prerequisite(self, repo):
        plan, _ids = _native_blocked(repo, TestTheExecutorLeg.PAIR)
        state = StateManager(plan)
        plan_dir = plan.parent / plan.stem
        assert prereq.check_phase_prerequisite(state, "review-plan")["met"] is True
        assert scope_apply.recorded_blockers(state.state, "review-plan") == []
        assert prereq.collect_missing(state, plan_dir, PHASE) == []
        assert prereq.ensure_prerequisites(plan, PHASE, unattended=True).met is True
        # Its dependents still refuse, naming the apply mode and the review.
        missing = prereq.collect_missing(state, plan_dir, "generate-tasks")
        message = prereq.build_unmet_message("generate-tasks", plan, plan_dir, missing)
        plan_abs = str(plan.resolve())
        assert prereq.dispatcher_command(
            PHASE, plan_abs, "--yes", "--scope-policy", "expand") in message
        assert prereq.dispatcher_command("review-plan", plan_abs, "--yes", "--force",
                                         "--rerun-all") in message
        assert "review-plan" not in message.split("--scope-policy expand")[0].rsplit("— ", 1)[-1]

    def test_skip_prerequisites_cannot_skip_a_scope_blocker(self, repo):
        plan, _ids = _native_blocked(repo, TestTheExecutorLeg.PAIR)
        outcome = prereq.ensure_prerequisites(
            plan, "generate-tasks", unattended=True, skip_prerequisites=True)
        assert outcome.met is False and outcome.exit_code == 3
        assert outcome.skipped == ()
        assert "cannot skip apply-suggestions for generate-tasks" in outcome.message
        assert "scope blocker" in outcome.message
        assert not StateManager(plan).is_phase_skipped(PHASE)

    def test_implement_cannot_skip_a_scope_blocker_either(self, repo, capsys):
        import implement_orchestrator

        plan, _ids = _native_blocked(repo, TestTheExecutorLeg.PAIR)
        capsys.readouterr()
        try:
            code = implement_orchestrator.main([
                "--plan-file", str(plan), "--output", str(plan.parent / "tasks.json"),
                "--yes", "--skip-prerequisites"])
        except SystemExit as exc:
            code = int(exc.code or 0)
        out = capsys.readouterr().out
        assert code == 3
        assert "cannot skip apply-suggestions for implement" in out
        assert "--scope-policy expand" in out
        assert not StateManager(plan).is_phase_skipped(PHASE)


class TestABlockedTaskSuggestionsApplyGatesImplement:

    def _blocked_task_apply(self, root):
        plan, ids = _task_review(root, "preserve")
        state = StateManager(plan)
        state.mark_phase_completed("apply-task-suggestions")
        applied = scope_apply.resolve_apply_policy(None, state.state, "apply-task-suggestions")
        scope_apply.record_completion(state.state, "apply-task-suggestions", applied, [
            {"group_id": ids["in_scope"], "task_id": "apply-task-batch-0",
             "reason": "T001 cannot name the columns without a schema service"}])
        state.save()
        return plan

    def test_the_check_reads_the_blocker(self, repo):
        plan = self._blocked_task_apply(repo.root)
        state = StateManager(plan)
        check = prereq.check_apply_task_suggestions_prerequisite(state, plan.parent / plan.stem)
        assert check["met"] is False and check["scope_blocked"] is True
        missing = prereq.collect_missing(state, plan.parent / plan.stem, "implement")
        assert [m["phase"] for m in missing] == ["apply-task-suggestions"]
        message = prereq.build_unmet_message("implement", plan, plan.parent / plan.stem, missing)
        plan_abs = str(plan.resolve())
        assert prereq.dispatcher_command("apply-task-suggestions", plan_abs, "--yes",
                                         "--scope-policy", "expand") in message
        assert prereq.dispatcher_command("review-tasks", plan_abs, "--yes", "--force",
                                         "--rerun-all") in message
        outcome = prereq.ensure_prerequisites(
            plan, "implement", unattended=True, skip_prerequisites=True)
        assert outcome.met is False and outcome.skipped == ()

    def test_implement_stops_instead_of_skipping_it_as_an_advisory(self, repo, capsys):
        import implement_orchestrator

        plan = self._blocked_task_apply(repo.root)
        capsys.readouterr()
        try:
            code = implement_orchestrator.main([
                "--plan-file", str(plan), "--output", str(plan.parent / "tasks.json"),
                "--yes"])
        except SystemExit as exc:
            code = int(exc.code or 0)
        out = capsys.readouterr().out
        assert code == 3
        assert "[PREREQUISITE_UNMET] implement:" in out and "scope blocker" in out
        assert "[TASK_SUGGESTIONS_ADVISORY]" not in out
        assert not StateManager(plan).is_phase_skipped("apply-task-suggestions")

    def test_without_a_blocker_the_completion_satisfies_it(self, repo):
        plan, _ids = _task_review(repo.root, "preserve")
        state = StateManager(plan)
        state.mark_phase_completed("apply-task-suggestions")
        state.save()
        check = prereq.check_apply_task_suggestions_prerequisite(state, plan.parent / plan.stem)
        assert check["met"] is True and "scope_blocked" not in check
        assert prereq.collect_missing(state, plan.parent / plan.stem, "implement") == []


class TestAPolicyChangeKeepsLandedWorkApplied:

    def test_preserve_after_a_completed_expand_apply_defers_nothing_it_applied(
            self, repo, capsys):
        from utils import batch_done

        cases = {k: PLAN_CASES[k] for k in ("in_scope", "extension")}
        review = write_review(repo.root, cases=cases, review_policy="expand",
                              verdicts=False, selections=False)
        plan, ids = review["plan"], review["ids"]
        assert run_apply(aso, plan, "--yes") == 0
        for task in _tasks(plan):
            assert batch_done.main(["--phase-dir", str(_phase_dir(plan)), "--task-id",
                                    task["task_id"], "--status", "completed"]) == 0
        assert run_apply(aso, plan, "--mark-completed") == 0

        capsys.readouterr()
        assert run_apply(aso, plan, scope_policy="preserve") == 0
        payload = json.loads(capsys.readouterr().out)
        assert "0 valid item(s) are deferred by --scope-policy preserve" in payload["message"]

        capsys.readouterr()
        assert run_apply(aso, plan, "--yes", scope_policy="preserve") == 0
        err = capsys.readouterr().err
        assert "[SCOPE_POLICY_CHANGED]" in err
        assert scope_apply.DEFERRED_MARKER not in err
        assert "valid but deferred" not in err
        record = _record(plan)
        assert record.get("deferrals") == []
        assert scope_apply.COMPLETION_KEY not in record  # no "eligible work completed, deferred=N"
        processed = _state(plan)[f"processed_{PHASE}"]
        assert {processed[ids[k]]["status"] for k in cases} == {"applied"}
        output = _output(plan)
        assert output[scope_apply.OUTPUT_KEY]["deferred_count"] == 0
        assert output["batches"] == []  # nothing handed out again either


class TestARefusedOverrideChangesNothing:

    def test_the_unsettled_refusal_records_no_apply_policy(self, repo, capsys):
        review = write_review(repo.root, cases=E2E, review_policy="preserve", verdicts=False)
        plan, ids = review["plan"], review["ids"]
        assert _run_executor(repo, plan, [], edits=False, result={
            "status": "failed", "ids": [], "files_touched": [], "summary": "could not"}) == 3
        state_file = plan.parent / plan.stem / "state.json"
        before = state_file.read_bytes()
        capsys.readouterr()
        assert _run_executor(repo, plan, ids.values(), scope_policy="expand") == 3
        assert "did not finish" in capsys.readouterr().err
        assert state_file.read_bytes() == before
        assert "apply_policy" not in _record(plan)
        # The recovery the refusal names works with the flag omitted: the
        # round is still preserve, for util write-prompt and for --resume.
        task = _tasks(plan)[0]
        text, _task, _info = write_prompt.render_write_prompt(_phase_dir(plan), task["task_id"])
        assert "## Scope Policy: Preserve" in text
        capsys.readouterr()
        assert _run_executor(repo, plan, [ids["in_scope"]], "--resume") == 0
        assert "did not finish" not in capsys.readouterr().err
        assert _applied(plan, ids) == {"in_scope"}
