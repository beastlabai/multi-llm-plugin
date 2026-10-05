"""D6 — the unattended contract: fifteen sites, three dispositions, no questions.

What these tests pin, in the order the design states it:

1. **The table is closed and typed.** Fifteen rows, each landing on exactly the
   disposition D6's second table names — auto-select, fail closed or explicit
   flag. "Ask in free text" is not a fourth option, so a row with no disposition,
   or with one outside that set, is a failure.
2. **D5a: no decision exists only as a question.** Every row carries a decision
   flag (or is one of the three documented flagless rows), and every named flag
   is *actually accepted* by the parser that owns it — built and inspected here,
   not asserted from documentation.
3. **Row 15 is the validator behind all of them.** An empty or unrecognized
   decision value is exit 3 at parse time, never a silent default, and never
   argparse's exit 2 (which a host reads as "nothing to do").
4. **Rows 5/6/13 fail closed rather than auto-skipping** (D6a), the auto-run
   inside the workflow happens at most once, and `--skip-prerequisites` is loud,
   recorded and never implied by `--yes`.
5. **D6b: Python never sniffs.** No `isatty()`, no `CI` read, no `--print`
   detection anywhere — the same argv must produce the same phases in a terminal
   and in a pipe.
6. **D6c: the uv backstop.** One message file, one probe, exit 3, and no
   `python3 -m` fallback anywhere.

`tests/test_unattended_e2e.py` is the other half: it observes that a `--yes` run
of every standalone mode reaches no prompt *with stdin closed*, rather than
inferring it from the absence of an `AskUserQuestion` string.
"""

import argparse
import ast
import os
import functools
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

SKILL_ROOT = Path(__file__).parent.parent

import check_workflow_prerequisites as prereq
import implement_orchestrator
import modes as mode_table
import multi_llm
from utils import cli_common
from utils import unattended as un
from utils.apply_orchestrator_base import (
    ApplyOrchestratorBase,
    apply_unattended_defaults,
)
from utils.cli_common import build_parent_parser, flag_value, parse_result_line
from utils.filtering import should_bypass_no_selection_confirmation
from utils.state_manager import StateManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def plan_with_unapplied_suggestions(tmp_path):
    """A plan whose review-plan phase produced valid, never-applied suggestions."""
    plan = tmp_path / "gate-plan.md"
    plan.write_text("# Gate Plan\n\n## Overview\nTest.\n", encoding="utf-8")
    review = tmp_path / "gate-plan" / "review-plan"
    review.mkdir(parents=True)
    (review / "validation.json").write_text(json.dumps({
        "g1": {"status": "valid", "importance": "HIGH"},
        "g2": {"status": "valid", "importance": "HIGH"},
        "g3": {"status": "valid", "importance": "MEDIUM"},
        "g4": {"status": "invalid", "importance": "LOW"},
    }), encoding="utf-8")
    return plan


@pytest.fixture
def plan_without_tasks(tmp_path):
    """A plan that never ran generate-tasks (row 13's subject)."""
    plan = tmp_path / "no-tasks.md"
    plan.write_text("# No Tasks\n\n## Overview\nTest.\n", encoding="utf-8")
    return plan


# ---------------------------------------------------------------------------
# 1. The table: fifteen rows, one disposition each
# ---------------------------------------------------------------------------

#: The D6 disposition table, transcribed from the design so a code change that
#: quietly re-dispositions a site fails here instead of shipping.
EXPECTED_DISPOSITIONS = {
    1: (un.FAIL_CLOSED,),
    2: (un.AUTO_SELECT, un.FAIL_CLOSED, un.EXPLICIT_FLAG),
    3: (un.AUTO_SELECT, un.FAIL_CLOSED),
    4: (un.AUTO_SELECT,),
    5: (un.AUTO_SELECT, un.FAIL_CLOSED, un.EXPLICIT_FLAG),
    6: (un.AUTO_SELECT, un.FAIL_CLOSED, un.EXPLICIT_FLAG),
    7: (un.AUTO_SELECT,),
    8: (un.AUTO_SELECT, un.EXPLICIT_FLAG),
    9: (un.FAIL_CLOSED, un.EXPLICIT_FLAG),
    10: (un.AUTO_SELECT,),
    11: (un.AUTO_SELECT,),
    12: (un.AUTO_SELECT, un.EXPLICIT_FLAG),
    13: (un.FAIL_CLOSED,),
    14: (un.FAIL_CLOSED,),
    15: (un.FAIL_CLOSED,),
}

#: The only rows whose answer is out-of-band by design (1, 13) or which *are* the
#: validator (15). Every other row must name a flag.
FLAGLESS_ROWS = {1, 13, 15}


class TestTheTable:
    def test_exactly_fifteen_rows_numbered_one_to_fifteen(self):
        assert len(un.ROWS) == 15
        assert [r.number for r in un.ROWS] == list(range(1, 16))

    def test_site_ids_are_unique(self):
        ids = [r.site_id for r in un.ROWS]
        assert len(set(ids)) == len(ids)

    @pytest.mark.parametrize("number", sorted(EXPECTED_DISPOSITIONS))
    def test_every_row_lands_on_its_named_disposition(self, number):
        assert un.row(number).dispositions == EXPECTED_DISPOSITIONS[number]

    def test_only_three_dispositions_exist(self):
        for r in un.ROWS:
            assert r.dispositions, f"row {r.number} has no disposition"
            for disposition in r.dispositions:
                assert disposition in un.DISPOSITIONS, (
                    f"row {r.number}: {disposition!r} is not one of "
                    f"{un.DISPOSITIONS} — a free-text question is not a fourth "
                    "option"
                )

    def test_every_row_names_where_it_is_enforced(self):
        for r in un.ROWS:
            assert r.enforced_in, f"row {r.number} names no owner"
            for target in r.enforced_in:
                assert (SKILL_ROOT / target).exists(), (
                    f"row {r.number} points at a missing file: {target}"
                )

    def test_every_row_is_reached_by_a_real_mode(self):
        known = set(mode_table.mode_names()) | {"*"}
        for r in un.ROWS:
            unknown = set(r.modes) - known
            assert not unknown, f"row {r.number} names unknown modes: {unknown}"


# ---------------------------------------------------------------------------
# 2. D5a: every row has a flag, and every flag is really accepted
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _help_text(script: str) -> str:
    """The script's own ``--help``, which is the authoritative flag list.

    Asked as a subprocess rather than by reaching into a parser object: every
    orchestrator builds its parser inside ``main(argv)`` (import must stay
    side-effect free), and ``--help`` is exactly what a host reads to discover
    the flag. Cached, so 13 flags cost 9 launches.
    """
    result = subprocess.run(
        [sys.executable, str(SKILL_ROOT / script), "--help"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(SKILL_ROOT), stdin=subprocess.DEVNULL, timeout=120,
    )
    return result.stdout + result.stderr


def _parent_parser_options():
    parser = build_parent_parser()
    options = set()
    for action in parser._actions:
        options.update(action.option_strings)
    return options


def _accepts(parser_name: str, flag: str) -> bool:
    if parser_name == "parent":
        return flag in _parent_parser_options()
    return flag in _help_text(parser_name)


#: Every row's flag is now owned by a real parser: `full`'s orchestrator landed
#: with the Phase-3 sequencing, so nothing is validated by the dispatcher alone.
PARSERS_WITHOUT_A_SCRIPT_YET: set = set()


class TestDecisionInputContract:
    def test_every_row_has_a_flag_or_is_documented_flagless(self):
        for r in un.ROWS:
            if r.number in FLAGLESS_ROWS:
                assert not r.flags, (
                    f"row {r.number} is documented flagless but names {r.flags}"
                )
                continue
            assert r.flags, (
                f"row {r.number} has no decision flag — adding a gate without its "
                "flag is a review error (D5a)"
            )
            assert un.flags_for_row(r.number), (
                f"row {r.number}'s flags are not in DECISION_FLAGS"
            )

    def test_every_flag_maps_back_to_a_row(self):
        numbers = {r.number for r in un.ROWS}
        for spec in un.DECISION_FLAGS:
            assert spec.rows, f"{spec.name} answers no row"
            assert set(spec.rows) <= numbers

    @pytest.mark.parametrize("spec", un.DECISION_FLAGS, ids=lambda s: s.name)
    def test_the_owning_parser_accepts_the_flag(self, spec):
        for parser_name in spec.parsers:
            if parser_name in PARSERS_WITHOUT_A_SCRIPT_YET:
                continue
            assert _accepts(parser_name, spec.name), (
                f"{parser_name} does not accept {spec.name}, so D6 row(s) "
                f"{spec.rows} have no non-interactive answer there"
            )
            if spec.choices and spec.value_form == "one":
                rendered = "{" + ",".join(spec.choices) + "}"
                assert rendered in _help_text(parser_name), (
                    f"{parser_name}{spec.name} must validate with choices "
                    f"{spec.choices} (D5a: values are validated, not guessed)"
                )

    def test_skip_prerequisites_is_defined_once_in_the_shared_parent(self):
        assert "--skip-prerequisites" in _parent_parser_options()

    def test_yes_does_not_imply_skip_prerequisites(self):
        parser = argparse.ArgumentParser(parents=[build_parent_parser()])
        args = parser.parse_args(["--yes"])
        assert args.yes is True
        assert args.skip_prerequisites is False, (
            "consent to run unattended is not consent to stale input (D6a)"
        )


class TestFullWorkflowOwnsItsRows:
    """Rows 12 and 14 belong to `full`, and `full` is now a real parser."""

    def test_review_tasks_flag_is_declared_for_full(self):
        spec = un.flag_spec("--review-tasks")
        assert spec is not None and spec.rows == (12,)
        assert "full_workflow_orchestrator.py" in spec.parsers

    def test_resume_is_the_row_fourteen_re_invocation(self):
        spec = un.flag_spec("--resume")
        assert spec is not None and spec.rows == (14,)
        assert "full_workflow_orchestrator.py" in spec.parsers

    def test_the_mode_accepts_both_flags_and_reaches_the_plan_check(self, capsys):
        # The exit 3 is about the missing plan, never about an unrecognized
        # decision flag: the flags parse, so the workflow got as far as its
        # pre-flight.
        assert multi_llm.main(
            ["full", "p.md", "--yes", "--review-tasks", "--resume"]
        ) == 1
        assert "Plan file not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 3. Row 15: empty or unrecognized is exit 3 at parse time
# ---------------------------------------------------------------------------

class TestRowFifteen:
    @pytest.mark.parametrize("argv", [
        ["--models", ""],
        ["--models", "", "--yes"],
        ["--models"],
        ["--models=", "--yes"],
        ["--human-task", "T001="],
        ["--human-task", "T001=maybe"],
        ["--human-task", "=done"],
        ["--human-task", ""],
        ["--force-strategy", "bogus"],
        ["--force-strategy", ""],
        ["--force-strategy=nope"],
    ])
    def test_rejected(self, argv):
        with pytest.raises(un.NeedsHuman) as exc:
            un.validate_decision_flags(argv)
        assert exc.value.exit_code == 3

    @pytest.mark.parametrize("argv", [
        ["--models", "a:b"],
        ["--models", "a:b", "c:d", "--yes"],
        ["--models=a:b"],
        ["--human-task", "T001=done"],
        ["--human-task", "T001=skip", "--human-task", "T002=done"],
        ["--force-strategy", "skip-dependents"],
        ["--force-strategy=skip-continue"],
        ["--yes", "--detach"],
        [],
    ])
    def test_accepted(self, argv):
        un.validate_decision_flags(argv)

    def test_repeating_a_task_with_two_answers_is_exit_three(self):
        with pytest.raises(un.NeedsHuman):
            un.parse_human_task_decisions(["T1=done", "T1=skip"])

    def test_repeating_a_task_with_the_same_answer_is_fine(self):
        assert un.parse_human_task_decisions(["T1=done", "T1=done"]) == {"T1": "done"}

    @pytest.mark.parametrize("argv,needle", [
        (["review-plan", "p.md", "--yes", "--models", ""], "--models"),
        (["implement", "p.md", "--human-task", "T1="], "--human-task"),
        (["implement", "p.md", "--force-strategy", "bogus"], "--force-strategy"),
    ])
    def test_the_dispatcher_exits_three_not_argparse_two(self, argv, needle, capsys):
        assert multi_llm.main(argv) == 3
        captured = capsys.readouterr()
        assert needle in captured.err
        result = parse_result_line(captured.out)
        assert result["exit"] == 3 and result["state"] == "needs-human"


# ---------------------------------------------------------------------------
# 4. Rows 5/6/13 — D6a
# ---------------------------------------------------------------------------

class TestPrerequisiteGates:
    def test_standalone_unattended_is_exit_three(self, plan_with_unapplied_suggestions):
        outcome = prereq.ensure_prerequisites(
            plan_with_unapplied_suggestions, "generate-tasks", unattended=True
        )
        assert outcome.met is False
        assert outcome.exit_code == 3
        assert outcome.message.startswith("[PREREQUISITE_UNMET] generate-tasks:")

    def test_the_message_names_both_fixes_with_absolute_paths(
        self, plan_with_unapplied_suggestions
    ):
        outcome = prereq.ensure_prerequisites(
            plan_with_unapplied_suggestions, "generate-tasks", unattended=True
        )
        message = outcome.message
        # Both fixes, in the D5a shape the agent copies rather than composes.
        assert "1) Apply them first" in message
        assert "2) Proceed on unapplied review output (recorded)" in message
        assert "apply-suggestions" in message and "--skip-prerequisites" in message
        # The breakdown the design's example shows, not a bare count.
        assert "2 HIGH / 1 MEDIUM" in message
        # Absolute paths only (D1c), and the real artifact this counted.
        plan_abs = str(plan_with_unapplied_suggestions.resolve())
        assert plan_abs in message
        assert str((plan_with_unapplied_suggestions.parent / "gate-plan"
                    / "review-plan" / "validation.json").resolve()) in message
        assert "${CLAUDE_SKILL_DIR}" not in message and "{SKILL_DIR}" not in message

    def test_implement_prints_the_same_block_with_implement_substituted(
        self, plan_with_unapplied_suggestions
    ):
        outcome = prereq.ensure_prerequisites(
            plan_with_unapplied_suggestions, "implement", unattended=True
        )
        assert outcome.exit_code == 3
        assert "[PREREQUISITE_UNMET] implement:" in outcome.message

    def test_attended_keeps_todays_payload(self, plan_with_unapplied_suggestions):
        outcome = prereq.ensure_prerequisites(
            plan_with_unapplied_suggestions, "generate-tasks", unattended=False
        )
        assert outcome.met is False
        assert outcome.exit_code == 0, (
            "with --yes unset, interactive behavior is unchanged"
        )

    def test_nothing_is_produced_before_the_gate(self, plan_with_unapplied_suggestions):
        plan_dir = plan_with_unapplied_suggestions.parent / "gate-plan"
        before = sorted(p.name for p in plan_dir.iterdir())
        prereq.ensure_prerequisites(
            plan_with_unapplied_suggestions, "generate-tasks", unattended=True
        )
        assert sorted(p.name for p in plan_dir.iterdir()) == before
        assert not (plan_dir / "generate-tasks").exists()

    def test_skip_prerequisites_is_loud_recorded_and_non_fatal(
        self, plan_with_unapplied_suggestions
    ):
        outcome = prereq.ensure_prerequisites(
            plan_with_unapplied_suggestions,
            "generate-tasks",
            unattended=True,
            skip_prerequisites=True,
        )
        assert outcome.met is True and outcome.exit_code == 0
        assert outcome.skipped == ("apply-suggestions",)
        assert any(n.startswith("[PREREQUISITE_SKIPPED] ") for n in outcome.notices)
        state = StateManager(plan_with_unapplied_suggestions)
        assert state.is_phase_skipped("apply-suggestions")
        assert prereq.SKIP_PREREQUISITES_REASON in (
            state.get_phase_skip_reason("apply-suggestions") or ""
        )

    def test_a_recorded_skip_surfaces_upstream_and_is_not_re_asked(
        self, plan_with_unapplied_suggestions
    ):
        prereq.ensure_prerequisites(
            plan_with_unapplied_suggestions, "generate-tasks",
            unattended=True, skip_prerequisites=True,
        )
        # A later phase gates on its own prerequisite only.
        outcome = prereq.ensure_prerequisites(
            plan_with_unapplied_suggestions, "implement", unattended=True
        )
        assert any(
            n.startswith(prereq.SKIPPED_UPSTREAM_MARKER) for n in outcome.notices
        ), outcome.notices
        assert prereq.SKIP_PREREQUISITES_REASON in " ".join(outcome.notices)

    def test_row_thirteen_fails_closed_and_refuses_skip_prerequisites(
        self, plan_without_tasks
    ):
        outcome = prereq.ensure_prerequisites(
            plan_without_tasks, "review-tasks", unattended=True
        )
        assert outcome.exit_code == 3
        assert "generate-tasks" in outcome.message
        assert "--skip-prerequisites" not in outcome.message, (
            "generating tasks is not a repair review-tasks may make"
        )
        refused = prereq.ensure_prerequisites(
            plan_without_tasks, "review-tasks",
            unattended=True, skip_prerequisites=True,
        )
        assert refused.exit_code == 3
        assert "cannot skip" in refused.message


class TestWorkflowAutoRun:
    """The `--yes` auto-run inside `--full`: verified, bounded, never a loop."""

    def test_auto_run_then_re_check_satisfies_the_gate(
        self, plan_with_unapplied_suggestions
    ):
        calls = []

        def runner(phase):
            calls.append(phase)
            state = StateManager(plan_with_unapplied_suggestions)
            state.mark_phase_completed(phase)
            state.save()
            return 0

        outcome = prereq.ensure_prerequisites(
            plan_with_unapplied_suggestions, "generate-tasks",
            unattended=True, in_workflow=True, apply_runner=runner,
        )
        assert calls == ["apply-suggestions"]
        assert outcome.met is True and outcome.exit_code == 0
        assert outcome.auto_ran == ("apply-suggestions",)

    def test_a_second_unmet_recheck_is_exit_three_not_a_retry(
        self, plan_with_unapplied_suggestions
    ):
        calls = []

        def runner(phase):
            calls.append(phase)
            return 0  # exits 0 but resolves nothing

        outcome = prereq.ensure_prerequisites(
            plan_with_unapplied_suggestions, "generate-tasks",
            unattended=True, in_workflow=True, apply_runner=runner,
        )
        assert calls == ["apply-suggestions"], "at most once per phase per run"
        assert outcome.exit_code == 3
        assert "re-check still reports" in outcome.message

    def test_a_failing_apply_phase_stops_the_workflow(
        self, plan_with_unapplied_suggestions
    ):
        outcome = prereq.ensure_prerequisites(
            plan_with_unapplied_suggestions, "generate-tasks",
            unattended=True, in_workflow=True, apply_runner=lambda phase: 1,
        )
        assert outcome.exit_code == 3
        assert "exited 1" in outcome.message

    def test_without_a_runner_workflow_mode_behaves_like_standalone(
        self, plan_with_unapplied_suggestions
    ):
        outcome = prereq.ensure_prerequisites(
            plan_with_unapplied_suggestions, "generate-tasks",
            unattended=True, in_workflow=True,
        )
        assert outcome.exit_code == 3, (
            "no runner means nothing can silently proceed on stale input"
        )


class TestPrerequisiteCli:
    def test_yes_exits_three_and_stdout_stays_pure_json(
        self, plan_with_unapplied_suggestions, capsys
    ):
        code = prereq.main([
            "--plan-file", str(plan_with_unapplied_suggestions),
            "--mode", "generate-tasks", "--yes",
        ])
        captured = capsys.readouterr()
        assert code == 3
        payload = json.loads(captured.out)  # pure JSON: no prose leaked to stdout
        assert payload["prerequisites_met"] is False
        assert payload["state"] == "needs-human" and payload["exit"] == 3
        assert "[PREREQUISITE_UNMET]" in captured.err
        assert "prompt" not in payload, "an unattended run renders nothing (D6b)"

    def test_attended_still_emits_the_prompt_payload(
        self, plan_with_unapplied_suggestions, capsys
    ):
        code = prereq.main([
            "--plan-file", str(plan_with_unapplied_suggestions),
            "--mode", "generate-tasks",
        ])
        payload = json.loads(capsys.readouterr().out)
        assert code == 0
        assert payload["prompt"]["options"]

    def test_skip_prerequisites_exits_zero_and_records(
        self, plan_with_unapplied_suggestions, capsys
    ):
        code = prereq.main([
            "--plan-file", str(plan_with_unapplied_suggestions),
            "--mode", "generate-tasks", "--yes", "--skip-prerequisites",
        ])
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert code == 0
        assert payload["prerequisites_met"] is True
        assert payload["skipped_prerequisites"] == ["apply-suggestions"]
        assert "[PREREQUISITE_SKIPPED]" in captured.err


# ---------------------------------------------------------------------------
# 5. Rows 6-9 in the implement orchestrator
# ---------------------------------------------------------------------------

def _implement(plan, *flags):
    """Run the implement orchestrator, catching its ``sys.exit``.

    ``--output`` is passed so the default-output resolution (which needs a git
    root) never runs: these tests are about the gate, which sits before any
    output is written.
    """
    argv = [
        "--plan-file", str(plan),
        "--output", str(Path(plan).parent / "tasks.json"),
        *flags,
    ]
    try:
        return implement_orchestrator.main(argv)
    except SystemExit as exc:
        return 0 if exc.code is None else int(exc.code)


class TestImplementGate:
    def test_unattended_gate_is_exit_three_and_writes_nothing(
        self, plan_with_unapplied_suggestions, capsys
    ):
        code = _implement(plan_with_unapplied_suggestions, "--yes")
        out = capsys.readouterr().out
        assert code == 3
        assert "[PREREQUISITE_UNMET] implement:" in out
        assert not (
            plan_with_unapplied_suggestions.parent / "gate-plan" / "implement"
        ).exists()

    def test_attended_gate_keeps_the_prerequisite_check_marker(
        self, plan_with_unapplied_suggestions, capsys
    ):
        code = _implement(plan_with_unapplied_suggestions)
        out = capsys.readouterr().out
        assert code == 0
        assert "[PREREQUISITE_CHECK]" in out

    def test_skip_prerequisites_proceeds_loudly(
        self, plan_with_unapplied_suggestions, capsys
    ):
        code = _implement(
            plan_with_unapplied_suggestions, "--yes", "--skip-prerequisites"
        )
        out = capsys.readouterr().out
        assert "[PREREQUISITE_SKIPPED] implement:" in out
        # It got past the gate; what stops it now is the missing task list —
        # under --yes that is exit 3 naming the generate-tasks re-invocation, not
        # an exit 0 that would report "done" for a run that implemented nothing.
        assert code == 3 and "[TASKS_MISSING]" in out
        assert "[PREREQUISITE_UNMET] implement: no task list" in out
        state = StateManager(plan_with_unapplied_suggestions)
        assert state.is_phase_skipped("apply-suggestions")


class TestHumanTaskStrategy:
    """Row 8: `skip-dependents`, persisted exactly as `--force-strategy` does."""

    def _state(self, tmp_path):
        plan = tmp_path / "p.md"
        plan.write_text("# P\n", encoding="utf-8")
        return StateManager(plan)

    def test_no_human_tasks_resolves_to_nothing(self, tmp_path):
        state = self._state(tmp_path)
        assert implement_orchestrator.resolve_human_task_strategy(
            state, [], force_strategy=None, unattended=True, resume=False
        ) is None

    def test_unattended_resolves_to_skip_dependents_and_persists(self, tmp_path):
        state = self._state(tmp_path)
        resolved = implement_orchestrator.resolve_human_task_strategy(
            state, ["T002"], force_strategy=None, unattended=True, resume=False
        )
        assert resolved == "skip-dependents" == un.UNATTENDED_HUMAN_TASK_STRATEGY
        assert state.get("human_task_strategy") == "skip-dependents"

    def test_force_strategy_wins_over_the_unattended_default(self, tmp_path):
        state = self._state(tmp_path)
        resolved = implement_orchestrator.resolve_human_task_strategy(
            state, ["T002"], force_strategy="skip-continue",
            unattended=True, resume=False,
        )
        assert resolved == "skip-continue"

    def test_resume_reuses_the_saved_strategy_without_re_resolving(self, tmp_path):
        state = self._state(tmp_path)
        state.set("human_task_strategy", "skip-continue")
        state.save()
        resolved = implement_orchestrator.resolve_human_task_strategy(
            state, ["T002"], force_strategy=None, unattended=True, resume=True
        )
        assert resolved == "skip-continue"

    def test_attended_leaves_the_gate_to_the_agent(self, tmp_path):
        state = self._state(tmp_path)
        assert implement_orchestrator.resolve_human_task_strategy(
            state, ["T002"], force_strategy=None, unattended=False, resume=False
        ) is None


class TestHumanTaskPause:
    """Row 9: fail closed, or answered by `--human-task`."""

    @pytest.mark.parametrize("argv", [
        ["implement", "p.md", "--yes", "--force-strategy", "pause-and-ask"],
        ["implement", "p.md", "--force-strategy=pause-and-ask", "--non-interactive"],
        ["implement", "p.md", "--detach", "--force-strategy", "pause-and-ask"],
    ])
    def test_pause_and_ask_with_unattended_is_exit_three(self, argv, capsys):
        assert multi_llm.main(argv) == 3
        err = capsys.readouterr().err
        assert "pause-and-ask" in err

    def test_direct_invocation_refuses_it_too(self, tmp_path, capsys):
        plan = tmp_path / "p.md"
        plan.write_text("# P\n", encoding="utf-8")
        code = _implement(plan, "--yes", "--force-strategy", "pause-and-ask")
        assert code == 3
        assert "pause-and-ask" in capsys.readouterr().err

    def test_answers_are_validated_before_anything_is_read(self, tmp_path, capsys):
        plan = tmp_path / "p.md"
        plan.write_text("# P\n", encoding="utf-8")
        code = _implement(plan, "--human-task", "T004=")
        assert code == 3
        assert "--human-task" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 6. Rows 10/11 — the two "already covered" claims
# ---------------------------------------------------------------------------

class TestApplyPhaseRows:
    def test_yes_sets_agent_decide_itself(self):
        args = argparse.Namespace(yes=True, agent_decide=False)
        apply_unattended_defaults(args)
        assert args.agent_decide is True, (
            "D5a row 10: --yes sets both --agent-decide and --no-confirm"
        )

    def test_without_yes_nothing_is_added(self):
        args = argparse.Namespace(yes=False, agent_decide=False)
        apply_unattended_defaults(args)
        assert args.agent_decide is False

    def test_row_10_names_agent_decide_and_accepts_the_legacy_spellings(self):
        assert un.row(10).flags[0] == "--agent-decide"
        for spelling in ("--agent-decide", "--claude-decide", "--let-claude-decide"):
            assert un.flag_spec(spelling).name == "--agent-decide"

    def test_an_explicit_agent_decide_is_untouched(self):
        args = argparse.Namespace(yes=True, agent_decide=True)
        apply_unattended_defaults(args)
        assert args.agent_decide is True

    def test_yes_bypasses_the_no_selection_confirmation(self):
        assert should_bypass_no_selection_confirmation(yes=True) is True
        assert should_bypass_no_selection_confirmation() is False


# ---------------------------------------------------------------------------
# 7. D6b — Python never sniffs
# ---------------------------------------------------------------------------

#: Directories that are not this skill's source.
_NOT_OURS = {"tests", "__pycache__", ".venv", "plans", "assets", "templates"}


def _runtime_python_files():
    """Every runtime .py file this skill ships (tests and vendored code excluded)."""
    for path in sorted(SKILL_ROOT.rglob("*.py")):
        if _NOT_OURS & set(path.relative_to(SKILL_ROOT).parts):
            continue
        yield path


@functools.lru_cache(maxsize=None)
def _code_signals(path_str):
    """Attribute names, bare names and non-docstring string literals of a module.

    Parsed rather than grepped so a *comment* explaining that this skill never
    calls ``isatty()`` does not read as a call to it — the AST sees only real
    code. Docstrings are excluded for the same reason.
    """
    tree = ast.parse(Path(path_str).read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            docstrings.add(id(first.value))
    attrs, names, strings = set(), set(), []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attrs.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                strings.append(node.value)
    return attrs, names, tuple(strings)


class TestNoHostSniffing:
    #: The one permitted `isatty()` in the whole skill: choosing whether to emit
    #: ANSI colour on stderr. It decides how a line *looks*, never whether a
    #: decision is rendered, and it cannot change which phases run.
    ANSI_COLOR_ALLOWLIST = {"utils/metrics.py"}

    def test_no_isatty_outside_the_ansi_colour_allowlist(self):
        offenders = []
        for path in _runtime_python_files():
            relative = path.relative_to(SKILL_ROOT).as_posix()
            if relative in self.ANSI_COLOR_ALLOWLIST:
                continue
            attrs, names, _ = _code_signals(str(path))
            if "isatty" in attrs or "isatty" in names:
                offenders.append(relative)
        assert not offenders, (
            "D6b: interactivity is requested (--interactive), never detected. "
            f"isatty() found in: {offenders}"
        )

    def test_the_allowlisted_isatty_only_gates_colour(self):
        source = (SKILL_ROOT / "utils" / "metrics.py").read_text(encoding="utf-8")
        assert source.count("isatty") == 1
        assert "_should_colorize_stderr" in source
        # It must not be reachable from a decision: nothing else in the skill
        # calls the colour helper.
        callers = [
            p.relative_to(SKILL_ROOT).as_posix() for p in _runtime_python_files()
            if "_should_colorize_stderr" in p.read_text(encoding="utf-8")
        ]
        assert callers == ["utils/metrics.py"]

    def test_no_ci_environment_sniffing(self):
        forbidden = {"CI", "GITHUB_ACTIONS", "BUILD_NUMBER", "JENKINS_URL"}
        offenders = []
        for path in _runtime_python_files():
            _, _, strings = _code_signals(str(path))
            hits = forbidden & set(strings)
            if hits:
                offenders.append(
                    f"{path.relative_to(SKILL_ROOT).as_posix()}: {sorted(hits)}"
                )
        assert not offenders, f"CI= sniffing found in: {offenders}"

    def test_no_print_flag_detection_of_our_own_argv(self):
        """`--print` may only ever appear as a *provider CLI's* argument.

        A provider adapter spelling `cursor-agent --print` is building someone
        else's command line; reading `--print` out of *our* argv would be a host
        sniff, which is what this forbids.
        """
        offenders = []
        for path in _runtime_python_files():
            relative = path.relative_to(SKILL_ROOT).as_posix()
            if relative.startswith("utils/providers/"):
                continue  # provider argv, not ours
            _, _, strings = _code_signals(str(path))
            hits = [s for s in strings if "--print" in s]
            if hits:
                offenders.append(f"{relative}: {hits}")
        assert not offenders, (
            "D6b: no --print detection anywhere outside provider adapters: "
            f"{offenders}"
        )

    def test_no_stdin_reads_in_the_dispatcher_or_the_gates(self):
        """Nothing on the unattended path may block on stdin (D6b)."""
        for name in (
            "multi_llm.py", "check_workflow_prerequisites.py",
            "implement_orchestrator.py", "utils/unattended.py",
        ):
            text = (SKILL_ROOT / name).read_text(encoding="utf-8")
            assert "input(" not in text, f"{name} blocks on stdin"
            assert "sys.stdin.read" not in text, f"{name} blocks on stdin"

    def test_the_instruction_core_has_no_tty_branch(self):
        for path in sorted((SKILL_ROOT / "instructions").glob("*.md")):
            text = path.read_text(encoding="utf-8")
            assert "no TTY and no" not in text
            # `full-workflow.md` may only mention isatty to say it is never used.
            for lineno, line in enumerate(text.splitlines(), 1):
                if "isatty" not in line:
                    continue
                assert "never by" in line or "never" in line, (
                    f"{path.name}:{lineno} reintroduces a TTY branch: {line}"
                )

    @pytest.mark.parametrize("argv", [
        ["review-plan", "p.md", "--yes", "--interactive"],
        ["review-plan", "p.md", "--non-interactive", "-i"],
        ["ask", "p.md", "q", "-y", "--interactive"],
        # No parser in the skill passes `allow_abbrev=False`, so argparse binds
        # `--y`/`--intera` to the full options: a rule that scanned for the exact
        # spelling only was bypassable by typing three fewer characters.
        ["review-plan", "p.md", "--y", "--intera"],
        ["review-plan", "p.md", "--non-inter", "--interact"],
    ])
    def test_contradictions_are_exit_three_never_precedence(self, argv, capsys):
        assert multi_llm.main(argv) == 3
        assert "contradictory" in capsys.readouterr().err

    def test_an_abbreviated_detach_is_refused_like_the_full_spelling(self, capsys):
        """The one contradiction with no second layer behind it.

        `--yes --interactive` is caught again in `resolve_models`; an unsupported
        `--detach` is caught **only** here, so an abbreviation that slipped past
        ran the mode with a flag that does nothing at all.
        """
        assert multi_llm.main(["status", "p.md", "--det"]) == 3
        assert "--detach is not available" in capsys.readouterr().err

    def test_flag_value_reads_the_last_occurrence_like_a_store_action(self):
        """Argparse's last-wins, so a repeated flag cannot hide its real value."""
        repeated = ["--force-strategy", "skip-dependents",
                    "--force-strategy", "pause-and-ask"]
        assert flag_value(repeated, "--force-strategy") == "pause-and-ask"
        assert flag_value(["--force-strat=pause-and-ask"],
                          "--force-strategy") == "pause-and-ask"
        # A longer real option never reads as an abbreviation of a shorter one.
        assert flag_value(["--force-strategy-extra", "x"],
                          "--force-strategy") is None

    def test_an_exact_option_is_never_read_as_a_longer_one(self):
        """argparse resolves an exact match before it considers an abbreviation.

        `--force` is a real option, so the token after it is that flag's
        neighbour, never a `--force-strategy` value — reading it as one made the
        dispatcher's guard fire on a command line that never asked for the
        strategy at all.
        """
        assert flag_value(["--force", "pause-and-ask"],
                          "--force-strategy") is None
        assert flag_value(["--force", "--force-strategy", "pause-and-ask"],
                          "--force-strategy") == "pause-and-ask"
        # A genuine abbreviation still binds.
        assert flag_value(["--force-s", "pause-and-ask"],
                          "--force-strategy") == "pause-and-ask"

    def test_the_exact_option_set_covers_every_shadowing_option(self):
        """The set is derived data: every prefix pair the parsers really have."""
        registered = set()
        for path in _runtime_python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "add_argument"):
                    continue
                for arg in node.args:
                    if (isinstance(arg, ast.Constant)
                            and isinstance(arg.value, str)
                            and arg.value.startswith("--")):
                        registered.add(arg.value)
        shadowing = {
            short for short in registered
            for other in registered
            if other != short and other.startswith(short)
        }
        assert shadowing == set(cli_common._EXACT_OPTIONS), (
            "utils/cli_common._EXACT_OPTIONS must list exactly the long options "
            "that are a strict prefix of another registered one"
        )


# ---------------------------------------------------------------------------
# 8. D6c — the uv backstop
# ---------------------------------------------------------------------------

class TestUvBackstop:
    def test_the_message_file_is_the_single_source(self):
        path = SKILL_ROOT / "hosts" / "uv-missing.txt"
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "astral.sh/uv/install.sh" in text, "the official installer"
        assert "brew install uv" in text, "the package-manager alternative"
        assert "no bare-`python` fallback" in text
        assert un.uv_missing_text().strip() in text

    def test_no_python_dash_m_fallback_anywhere(self):
        offenders = []
        for path in _runtime_python_files():
            text = path.read_text(encoding="utf-8")
            if "python3 -m" in text or "python -m pip" in text:
                offenders.append(path.relative_to(SKILL_ROOT).as_posix())
        assert not offenders, (
            "D6c: there is no bare-interpreter fallback, deliberately — "
            f"found in {offenders}"
        )

    def test_missing_uv_is_exit_three_with_the_file_text(self, monkeypatch, capsys):
        monkeypatch.setattr(un, "find_uv", lambda: None)
        monkeypatch.setattr(multi_llm, "require_uv", un.require_uv)
        assert multi_llm.main(["review-plan", "p.md", "--yes"]) == 3
        captured = capsys.readouterr()
        assert "[UV_MISSING]" in captured.err
        assert "astral.sh/uv/install.sh" in captured.err
        result = parse_result_line(captured.out)
        assert result["exit"] == 3 and result["state"] == "needs-human"

    def test_the_probe_order_is_env_then_path_then_candidates(self, monkeypatch, tmp_path):
        stub = tmp_path / "uv"
        stub.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setenv(un.UV_ENV, str(stub))
        assert un.find_uv() == str(stub.resolve()) or un.find_uv() == str(stub)
        monkeypatch.delenv(un.UV_ENV)
        monkeypatch.setattr(un.shutil, "which", lambda name: "/usr/bin/uv")
        assert un.find_uv() == "/usr/bin/uv"

    def test_the_candidate_list_matches_the_shim_contract(self):
        contract = (SKILL_ROOT / "references" / "uv-check.md").read_text(
            encoding="utf-8"
        )
        for candidate in un.UV_CANDIDATES:
            leaf = candidate.replace("~/", "")
            assert leaf in contract, (
                f"{candidate} is probed by Python but absent from the shim-side "
                "preflight contract"
            )

    def test_uv_check_reference_no_longer_offers_to_install(self):
        text = (SKILL_ROOT / "references" / "uv-check.md").read_text(encoding="utf-8")
        assert "AskUserQuestion" not in text
        assert "hosts/uv-missing.txt" in text
        assert "exit 3" in text


# ---------------------------------------------------------------------------
# 9. Recording: an auto-resolution is attributable afterwards
# ---------------------------------------------------------------------------

class TestResolutionRecording:
    def test_marker_names_the_site_and_the_decision(self, capsys):
        un.record_resolution(None, "model-selection", "defaults.models: a:b")
        out = capsys.readouterr().out
        assert out.startswith("[AUTO_RESOLVED] model-selection: defaults.models")

    def test_an_unknown_site_is_a_programming_error(self):
        with pytest.raises(KeyError):
            un.record_resolution(None, "not-a-site", "x")

    def test_a_detached_run_lands_it_in_status_json(self, tmp_path, monkeypatch):
        from utils import detach, status_file
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()
        plan = tmp_path / "p.md"
        plan.write_text("# P\n", encoding="utf-8")
        handle = detach.RunHandle(
            str(phase_dir), run_id="run-1", mode="review-plan",
            plan_file=str(plan), detached=True,
        )
        status_file.claim_run(str(phase_dir), {
            "run_id": "run-1", "state": "running",
        })
        monkeypatch.setattr(detach, "_ACTIVE", handle)
        un.record_resolution(str(phase_dir), "model-selection", "defaults.models: a:b")
        run = status_file.read_run(str(phase_dir))
        assert [r["site"] for r in run["resolutions"]] == ["model-selection"]
        assert run["resolutions"][0]["decision"].startswith("defaults.models")
        assert run["resolutions"][0]["at"]

    def test_a_foreground_run_adds_no_keys(self, tmp_path):
        from utils import status_file
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()
        status_file.write_status(str(phase_dir), {"state": "models_running"})
        un.record_resolution(str(phase_dir), "model-selection", "defaults.models: a:b")
        assert "run" not in status_file.read_status(str(phase_dir)), (
            "with --detach unset, .status.json gains no new keys (D4)"
        )


# ---------------------------------------------------------------------------
# 10. Row 3 — partial results are usable output
# ---------------------------------------------------------------------------

class TestPartialResults:
    def test_partial_is_exit_zero_with_a_marker(self, capsys):
        code = un.report_partial_results(completed=["a:b"], failed=["c:d"])
        out = capsys.readouterr().out
        assert code == 0
        assert "[PARTIAL_RESULTS: c:d]" in out

    def test_everything_completed_prints_nothing(self, capsys):
        code = un.report_partial_results(completed=["a:b"], failed=[])
        assert code == 0
        assert capsys.readouterr().out == ""

    def test_zero_results_is_exit_three_naming_the_retry(self, capsys):
        code = un.report_partial_results(
            completed=[], failed=["a:b"], retry_command="uv run … --force",
        )
        assert code == 3
        err = capsys.readouterr().err
        assert "nothing to" in err and "--force" in err

    def test_the_auto_resolved_wording_says_continued_not_reaggregated(
        self, tmp_path, capsys
    ):
        """The fan-out that hit this row is continuing in-process past the
        models that failed -- nothing was reaggregated, and a separate
        `--reaggregate` invocation is a different code path entirely."""
        phase_dir = tmp_path / "code-review"
        phase_dir.mkdir()
        code = un.report_partial_results(
            completed=["a:b"], failed=["c:d"],
            phase_dir=str(phase_dir), unattended=True,
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "continued with partial results; failed: c:d" in out
        assert "reaggregated" not in out


# ---------------------------------------------------------------------------
# 11. The exit-3 convention itself
# ---------------------------------------------------------------------------

class TestExitThreeConvention:
    def test_three_and_two_stay_distinguishable(self):
        from utils.cli_common import EXIT_NEEDS_HUMAN, EXIT_NOTHING_TO_DO
        assert EXIT_NEEDS_HUMAN == 3 and EXIT_NOTHING_TO_DO == 2

    def test_needs_human_carries_exit_three(self):
        assert un.NeedsHuman("x").exit_code == 3

    def test_the_dispatcher_documents_it(self):
        help_text = multi_llm.render_help()
        assert "3 needs a human" in help_text
        assert "unattended run that cannot proceed" in help_text

    def test_every_mode_word_is_covered_by_the_table(self):
        """No user-facing mode may be absent from the D6 inventory."""
        covered = set()
        for r in un.ROWS:
            covered.update(r.modes)
        if "*" in covered:
            return  # the wildcard rows (1, 14, 15) cover every mode
        for mode in mode_table.user_facing_modes():
            assert mode.name in covered, mode.name

# ---------------------------------------------------------------------------
# 12. Every resolution is documented where an agent will read it
# ---------------------------------------------------------------------------

#: The mode file(s) that must state each row's `--yes` resolution, on top of
#: `full-workflow.md` (which must mention all fifteen).
ROW_DOCS = {
    1: ("review-plan.md", "review-code.md", "init-config.md"),
    2: ("review-plan.md", "review-tasks.md", "review-code.md"),
    3: ("review-plan.md", "review-tasks.md", "review-code.md", "ask.md"),
    4: ("review-plan.md",),
    5: ("generate-tasks.md",),
    6: ("implement.md",),
    7: ("implement.md",),
    8: ("implement.md",),
    9: ("implement.md",),
    10: ("apply-suggestions.md", "apply-task-suggestions.md",
         "apply-code-fixes.md", "review-tasks.md"),
    11: ("apply-suggestions.md", "apply-task-suggestions.md",
         "apply-code-fixes.md"),
    12: ("full-workflow.md",),
    13: ("review-tasks.md",),
    14: ("full-workflow.md",),
    15: ("full-workflow.md",),
}


def _mentions_row(text: str, number: int) -> bool:
    """True when *text* attributes something to D6 row *number*."""
    import re

    for match in re.finditer(r"rows?\s+((?:\d+)(?:\s*[/,]\s*\d+)*)", text):
        numbers = {int(n) for n in re.findall(r"\d+", match.group(1))}
        if number in numbers:
            return True
    return False


class TestDocumentedResolutions:
    def test_full_workflow_documents_all_fifteen(self):
        text = (SKILL_ROOT / "instructions" / "full-workflow.md").read_text(
            encoding="utf-8"
        )
        missing = [r.number for r in un.ROWS if not _mentions_row(text, r.number)]
        assert not missing, (
            "full-workflow.md must state every row's --yes resolution; missing "
            f"rows: {missing}"
        )

    @pytest.mark.parametrize("number", sorted(ROW_DOCS))
    def test_the_mode_files_state_the_resolution(self, number):
        for name in ROW_DOCS[number]:
            text = (SKILL_ROOT / "instructions" / name).read_text(encoding="utf-8")
            assert _mentions_row(text, number) or "--yes" in text, (
                f"{name} does not state row {number}'s --yes resolution"
            )

    def test_no_instruction_file_still_offers_to_install_uv(self):
        """Row 1 is fail-closed: instructions may forbid the offer, never make it."""
        for path in sorted((SKILL_ROOT / "instructions").glob("*.md")):
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                lowered = line.lower()
                if "install" not in lowered or "uv" not in lowered:
                    continue
                if "offer" not in lowered:
                    continue
                assert "never" in lowered, (
                    f"{path.name}:{lineno} still offers to install uv: {line}"
                )

    def test_the_deleted_no_tty_branch_is_gone_from_full_workflow(self):
        text = (SKILL_ROOT / "instructions" / "full-workflow.md").read_text(
            encoding="utf-8"
        )
        assert "No-TTY automation without" not in text
        assert "Non-interactive mode: skipped by default" not in text, (
            "the skip-by-default branch was decided by a TTY probe (D6 row 12)"
        )
        assert "--review-tasks" in text, "the explicit opt-in must be named"

    def test_phase_failure_recovery_is_fail_closed_under_yes(self):
        text = (SKILL_ROOT / "instructions" / "full-workflow.md").read_text(
            encoding="utf-8"
        )
        assert '"ask whether to continue" still applies' not in text, (
            "row 14: unattended, a crash is a terminal condition, never a question"
        )
        assert "--resume" in text


# ---------------------------------------------------------------------------
# 6b. Rows 10/11 — taken is not the same as recorded
# ---------------------------------------------------------------------------

class _StubApplyOrchestrator(ApplyOrchestratorBase):
    """The smallest concrete apply phase: enough to reach both D6 sites."""

    phase_name = "apply-stub"
    review_subdir = "review-plan"
    item_noun = "suggestion"
    supports_revalidation = False
    supports_skip_flag = False

    GROUPS = [
        {
            "theme": "A",
            "group_hash": "aaaa1111",
            "suggestions": [{"title": "S1", "desc": "d", "importance": "MEDIUM"}],
        }
    ]
    VALIDATION = [{"status": "valid", "reason": "OK", "confidence": 0.9}]

    def load_data(self):
        return (list(self.GROUPS), list(self.VALIDATION))

    def parse_user_edits(self, report_path):
        return {}

    def merge_user_edits(self, groups, edited_descriptions):
        return (groups, [])

    def parse_skips_from_report(self, report_path):
        return (set(), set(), set())

    def parse_validation_overrides_from_report(self, report_path):
        return ({}, {})

    def format_item_for_output(self, group, index):
        return {"index": index, "group_id": group.get("group_hash", "")}

    def create_batches(self, items):
        return []

    def generate_batch_prompts(self, batches):
        return batches

    def build_output_json(self, batches, *, resume_info=None):
        return {"batches": []}

    def get_output_path(self):
        phase_dir = os.path.join(self.out_dir, self.phase_name)
        os.makedirs(phase_dir, exist_ok=True)
        return os.path.join(phase_dir, "orchestrator_output.json")

    def print_text_summary(self, batches, output_path):
        return None


def _stub_apply_args(tmp_path, **flags):
    """The flags the stub phase takes, over a real plan dir."""
    plan = tmp_path / "p.md"
    plan.write_text("# P\n", encoding="utf-8")
    (tmp_path / "p" / "review-plan").mkdir(parents=True, exist_ok=True)
    defaults = dict(
        plan_file=str(plan), yes=False, force=False, dry_run=False, fresh=False,
        resume=False, approve_all=False, include_low=False, min_priority=None,
        verbose=False, approve_all_low=False, skip_all_human=False,
        approve_importance=None, skip_human_review=False, no_batch=False,
        max_batch_size=4, batch_review_mode="by-importance",
        accept_stale_consolidation=False, no_confirm=False, include_high=False,
        agent_decide=False,
    )
    defaults.update(flags)
    return argparse.Namespace(**defaults)


def _stub_apply_run(tmp_path, **flags):
    """Run the stub phase against a real plan dir; returns the orchestrator."""
    orch = _StubApplyOrchestrator(_stub_apply_args(tmp_path, **flags))
    assert orch.run() == 0
    return orch


class TestApplyPhaseRowsAreRecorded:
    """D6's own standard: an auto-selection nobody can see is a degradation."""

    def test_the_yes_that_flips_agent_decide_announces_itself(self, tmp_path, capsys):
        _stub_apply_run(tmp_path, yes=True)
        out = capsys.readouterr().out
        assert f"[AUTO_RESOLVED] {un.row(10).site_id}:" in out
        assert "--agent-decide" in out
        assert "--claude-decide" not in out

    def test_applying_everything_unasked_announces_itself(self, tmp_path, capsys):
        """Row 11: the decision a `--yes` apply run actually makes."""
        _stub_apply_run(tmp_path, yes=True)
        out = capsys.readouterr().out
        assert f"[AUTO_RESOLVED] {un.row(11).site_id}:" in out
        # and names the flag that made the bypass deliberate, not just "bypassed"
        assert "--yes" in out.split(un.row(11).site_id, 1)[1]

    def test_applying_everything_unasked_says_what_was_loaded_and_what_is_used(
        self, tmp_path, capsys, monkeypatch
    ):
        """2026-10-05, Codex racing run 3: the record read "11 loaded task
        suggestion group(s)" beside a review that reported 10 valid and 1
        invalid, as if all eleven were applied."""
        monkeypatch.setattr(_StubApplyOrchestrator, "GROUPS", [
            {"theme": "A", "group_hash": "aaaa1111",
             "suggestions": [{"title": "S1", "desc": "d", "importance": "MEDIUM"}]},
            {"theme": "B", "group_hash": "bbbb2222",
             "suggestions": [{"title": "S2", "desc": "d", "importance": "MEDIUM"}]},
        ])
        monkeypatch.setattr(_StubApplyOrchestrator, "VALIDATION", [
            {"group_id": "aaaa1111", "status": "valid", "reason": "OK",
             "confidence": 0.9},
            {"group_id": "bbbb2222", "status": "invalid", "reason": "no",
             "confidence": 0.9},
        ])
        _stub_apply_run(tmp_path, yes=True)
        line = next(
            l for l in capsys.readouterr().out.splitlines()
            if f"[AUTO_RESOLVED] {un.row(11).site_id}:" in l
        )
        assert "2 suggestion group(s) loaded: 1 valid (applied), 1 invalid (left out)" in line
        assert "--yes" in line

    def test_without_yes_neither_is_announced(self, tmp_path, capsys):
        """`--no-confirm` alone is a flag the caller typed, not a question answered."""
        _stub_apply_run(tmp_path, no_confirm=True)
        assert "[AUTO_RESOLVED]" not in capsys.readouterr().out

    def test_a_detached_apply_run_lands_both_in_status_json(
        self, tmp_path, monkeypatch, capsys
    ):
        """The half that was silently lost: both sites fire BEFORE the detach.

        `run.resolutions` stayed `[]` on exactly the runs that needed it, because
        `detach.record_resolution` is a no-op until the worker owns the phase dir.
        """
        from utils import detach, status_file

        plan = tmp_path / "p.md"
        plan.write_text("# P\n", encoding="utf-8")
        (tmp_path / "p" / "review-plan").mkdir(parents=True, exist_ok=True)
        phase_dir = tmp_path / "p" / "apply-stub"
        phase_dir.mkdir(parents=True)
        handle = detach.RunHandle(
            str(phase_dir), run_id="run-1", mode="apply-stub",
            plan_file=str(plan), detached=True,
        )
        status_file.claim_run(str(phase_dir), {"run_id": "run-1", "state": "running"})
        monkeypatch.setattr(detach, "_ACTIVE", handle)

        _stub_apply_run(tmp_path, yes=True)
        capsys.readouterr()

        run = status_file.read_run(str(phase_dir))
        assert [r["site"] for r in run["resolutions"]] == [
            un.row(10).site_id, un.row(11).site_id
        ]
        assert all(r["decision"] and r["at"] for r in run["resolutions"])

    def test_a_detach_launch_announces_a_row_once_on_the_worker_leg(
        self, tmp_path, capsys
    ):
        """Both sites fire before the boundary, so both legs reach them.

        The invoker prints the line and then re-executes itself with
        `--internal-worker`, which reaches the same site and prints it again —
        one decision, read as two. The worker is the leg that keeps it: it is
        the leg that writes the entry to `run.resolutions`.
        """
        _StubApplyOrchestrator(_stub_apply_args(tmp_path, yes=True, detach=True))
        assert "[AUTO_RESOLVED]" not in capsys.readouterr().out

        _StubApplyOrchestrator(
            _stub_apply_args(
                tmp_path, yes=True, detach=True, internal_worker=True
            )
        )
        assert f"[AUTO_RESOLVED] {un.row(10).site_id}:" in capsys.readouterr().out

    def test_a_detached_dry_run_still_announces_it(self, tmp_path, capsys):
        """`--detach --dry-run` forks nothing, so its only leg is the invoker's."""
        _StubApplyOrchestrator(
            _stub_apply_args(tmp_path, yes=True, detach=True, dry_run=True)
        )
        assert f"[AUTO_RESOLVED] {un.row(10).site_id}:" in capsys.readouterr().out

    def test_a_dry_run_does_not_record_applying_everything(self, tmp_path, capsys):
        """Row 11 names a decision `--dry-run` never takes: it applies nothing.

        `--dry-run` is itself one of the nine flags that bypass the no-selection
        confirmation, so the site is reached — and the run then exits at step 30
        with nothing applied.
        """
        orch = _StubApplyOrchestrator(
            _stub_apply_args(tmp_path, yes=True, dry_run=True)
        )
        with pytest.raises(SystemExit):
            orch.run()
        out = capsys.readouterr().out
        assert f"[AUTO_RESOLVED] {un.row(11).site_id}:" not in out
        # Row 10 is untouched: its claim (judged, never prompted) holds.
        assert f"[AUTO_RESOLVED] {un.row(10).site_id}:" in out

    def test_both_sites_are_where_the_table_says_they_are(self):
        """Rows 10/11 declare `utils/apply_orchestrator_base.py`; check it calls."""
        source = (
            SKILL_ROOT / "utils" / "apply_orchestrator_base.py"
        ).read_text(encoding="utf-8")
        assert "utils/apply_orchestrator_base.py" in un.row(10).enforced_in
        assert "utils/apply_orchestrator_base.py" in un.row(11).enforced_in
        assert "_unattended.row(10).site_id" in source
        assert "_unattended.row(11).site_id" in source
