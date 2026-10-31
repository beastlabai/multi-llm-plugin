"""``--scope-policy``: one declaration, every applicable parser, the exit-3 contract.

plans/scope-policy-and-report-tags.md §4: the flag is declared once
(``cli_common.add_scope_policy_argument``), accepted by every review, apply,
commit, generation, implementation, composite and ``full`` parser, absent where
it means nothing (``ask``, ``status``), validated as a D6 decision flag (an
empty or unknown value is exit 3 at parse time, never argparse's 2), registered
as D6 row 16, and carried verbatim by the replay list, retry commands and the
detached worker's argv. The run-record helpers (``utils/scope_policy_runs.py``)
and both packaged tool routes are checked here too.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import apply_code_fixes_orchestrator  # noqa: E402
import apply_suggestions_orchestrator  # noqa: E402
import apply_task_suggestions_orchestrator  # noqa: E402
import code_review_orchestrator  # noqa: E402
import full_workflow_orchestrator  # noqa: E402
import generate_tasks_orchestrator  # noqa: E402
import implement_orchestrator  # noqa: E402
import multi_llm  # noqa: E402
import review_plan_orchestrator  # noqa: E402
import review_tasks_orchestrator  # noqa: E402
from utils import cli_common, detach, improve_steps  # noqa: E402
from utils import improve_workflow as iw  # noqa: E402
from utils import scope_policy as sp  # noqa: E402
from utils import scope_policy_runs as runs  # noqa: E402
from utils import unattended as un  # noqa: E402
from utils.cli_common import parse_result_line  # noqa: E402

SKILL_ROOT = Path(__file__).parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent

#: Each applicable mode's own parser, called the way its ``main`` calls it.
PARSERS = {
    "review-plan": review_plan_orchestrator.parse_args,
    "review-tasks": review_tasks_orchestrator.parse_args,
    "review-code": code_review_orchestrator.parse_args,
    "apply-suggestions": apply_suggestions_orchestrator.ApplySuggestionsOrchestrator.parse_args,
    "apply-task-suggestions":
        apply_task_suggestions_orchestrator.ApplyTaskSuggestionsOrchestrator.parse_args,
    "apply-code-fixes": apply_code_fixes_orchestrator.ApplyCodeFixesOrchestrator.parse_args,
    "generate-tasks": generate_tasks_orchestrator.parse_args,
    "implement": implement_orchestrator.parse_args,
    "full": full_workflow_orchestrator.parse_args,
    "improve-plan": lambda argv: iw.parse_args("improve-plan", argv),
    "improve-code": lambda argv: iw.parse_args("improve-code", argv),
    "improve-commits": lambda argv: iw.parse_args("improve-commits", argv),
}


# ---------------------------------------------------------------------------
# 1. Every applicable parser takes it; the others do not
# ---------------------------------------------------------------------------

class TestEveryApplicableParser:
    @pytest.mark.parametrize("mode", sorted(PARSERS))
    @pytest.mark.parametrize("value", sp.POLICIES)
    def test_the_flag_parses_to_scope_policy(self, mode, value):
        args = PARSERS[mode](["--plan-file", "p.md", "--scope-policy", value])
        assert args.scope_policy == value

    @pytest.mark.parametrize("mode", sorted(PARSERS))
    def test_omitted_is_none_never_expand(self, mode):
        """Omission is ``None`` until resolution: a continuation inherits,
        so the parser must not turn "not typed" into ``expand``."""
        assert PARSERS[mode](["--plan-file", "p.md"]).scope_policy is None

    @pytest.mark.parametrize("mode", sorted(PARSERS))
    def test_argparse_still_refuses_an_unknown_value(self, mode, capsys):
        """``choices`` is the second layer for a direct invocation."""
        with pytest.raises(SystemExit) as excinfo:
            PARSERS[mode](["--plan-file", "p.md", "--scope-policy", "narrow"])
        assert excinfo.value.code == 2
        assert "invalid choice" in capsys.readouterr().err

    def test_util_improve_steps_takes_it_for_every_composite(self):
        for mode in improve_steps.MODES:
            args = improve_steps.parse_args(["p.md", "--mode", mode, "--scope-policy", "preserve"])
            assert args.scope_policy == "preserve"
            assert improve_steps.parse_args(["p.md", "--mode", mode]).scope_policy is None

    def test_the_mode_list_is_the_applicable_modes(self):
        """The help's mode list and the D6 row name the same modes."""
        assert set(cli_common.SCOPE_POLICY_MODES) == set(un.row(16).modes)
        assert "ask" not in cli_common.SCOPE_POLICY_MODES
        assert "status" not in cli_common.SCOPE_POLICY_MODES

    @pytest.mark.parametrize("mode", ["ask", "status", "init"])
    def test_meaningless_modes_do_not_offer_it(self, mode, capsys):
        assert multi_llm.main([mode, "--help"]) == 0
        assert "--scope-policy" not in capsys.readouterr().out

    def test_ask_refuses_it(self, tmp_path, capsys):
        import ask_orchestrator

        with pytest.raises(SystemExit) as excinfo:
            ask_orchestrator.parse_args(
                ["--plan-file", "p.md", "--question", "q", "--scope-policy", "preserve"])
        assert excinfo.value.code == 2
        assert "unrecognized arguments: --scope-policy" in capsys.readouterr().err

    @pytest.mark.parametrize("mode", ["review-plan", "implement", "improve-code", "full"])
    def test_help_names_the_default_inheritance_and_the_modes(self, mode, capsys):
        assert multi_llm.main([mode, "--help"]) == 0
        out = " ".join(capsys.readouterr().out.split())
        assert "--scope-policy {expand,preserve}" in out
        assert "expand (the default for a new run)" in out
        assert "Omitted, a continuation keeps the policy on record" in out
        assert "ask and status take none" in out

    @pytest.mark.parametrize("role", ["review", "apply", "generate-tasks", "implement",
                                      "composite", "full"])
    def test_one_declaration_with_a_role_sentence(self, role):
        parser = argparse.ArgumentParser()
        action = cli_common.add_scope_policy_argument(parser, role=role)
        assert action.option_strings == [sp.FLAG]
        assert action.dest == sp.DEST and action.default is None
        assert tuple(action.choices) == sp.POLICIES
        assert cli_common._SCOPE_POLICY_ROLES[role] in action.help

    def test_the_commit_doors_name_it_among_the_forwarded_flags(self, capsys):
        for door in ("review-commits", "apply-commit-fixes", "improve-commits"):
            assert multi_llm.main([door, "--help"]) == 0
            assert "--scope-policy" in capsys.readouterr().out, door


# ---------------------------------------------------------------------------
# 2. Invalid or empty: exit 3, the decision-flag message
# ---------------------------------------------------------------------------

class TestTheExitThreeContract:
    @pytest.mark.parametrize("tokens,needle", [
        (["--scope-policy", ""], "--scope-policy was passed an empty value"),
        (["--scope-policy="], "--scope-policy was passed an empty value"),
        (["--scope-policy"], "--scope-policy was passed with no value"),
        (["--scope-policy", "--yes"], "--scope-policy was passed with no value"),
        (["--scope-policy", "narrow"], "--scope-policy got unrecognized value 'narrow'"),
        (["--scope-policy", "Preserve"], "--scope-policy got unrecognized value 'Preserve'"),
    ])
    def test_the_validator_refuses_it(self, tokens, needle):
        with pytest.raises(un.NeedsHuman) as excinfo:
            un.validate_decision_flags(tokens)
        assert excinfo.value.exit_code == 3
        assert needle in excinfo.value.message
        assert "Legal values: expand, preserve." in excinfo.value.message
        assert excinfo.value.site == "scope-policy"

    @pytest.mark.parametrize("tokens", [
        ["--scope-policy", "expand"], ["--scope-policy=preserve"], [],
    ])
    def test_a_legal_value_passes(self, tokens):
        un.validate_decision_flags(tokens)

    @pytest.mark.parametrize("mode", ["review-plan", "apply-code-fixes", "implement",
                                      "improve-plan", "full", "generate-tasks"])
    @pytest.mark.parametrize("value", ["", "narrow"])
    def test_the_dispatcher_exits_three_not_two(self, mode, value, capsys):
        assert multi_llm.main([mode, "p.md", "--scope-policy", value, "--yes"]) == 3
        captured = capsys.readouterr()
        assert "--scope-policy" in captured.err
        assert "Legal values: expand, preserve." in captured.err
        result = parse_result_line(captured.out)
        assert result["exit"] == 3 and result["state"] == "needs-human"

    def test_the_commit_door_exits_three_before_resolving_a_range(self, capsys):
        assert multi_llm.main(["review-commits", "--scope-policy", ""]) == 3
        assert "--scope-policy was passed an empty value" in capsys.readouterr().err

    def test_util_improve_steps_exits_three(self, tmp_path, capsys):
        plan = tmp_path / "p.md"
        plan.write_text("# P\n", encoding="utf-8")
        assert multi_llm.main(["util", "improve-steps", str(plan), "--mode", "improve-code",
                               "--scope-policy", ""]) == 3
        captured = capsys.readouterr()
        assert "--scope-policy was passed an empty value" in captured.err
        assert "[IMPROVE_STEP]" not in captured.out


# ---------------------------------------------------------------------------
# 3. D6 row 16
# ---------------------------------------------------------------------------

class TestTheDecisionSite:
    def test_row_sixteen_is_answered_by_the_flag_and_auto_selected(self):
        row = un.row_for_site("scope-policy")
        assert row.number == 16
        assert row.dispositions == (un.AUTO_SELECT, un.EXPLICIT_FLAG)
        assert row.flags == (sp.FLAG,)
        spec = un.flag_spec(sp.FLAG)
        assert spec.rows == (16,) and spec.choices == sp.POLICIES
        assert spec.value_form == "one"

    def test_it_is_never_a_question(self):
        entry = un.inventory_row("scope-policy")
        assert entry.renderable is False
        assert un.presentation("scope-policy", un.ASK_STRUCTURED) == "never rendered"
        assert un.presentation("scope-policy", un.ASK_FREE_TEXT) == "never rendered"

    def test_an_unattended_resolution_is_recorded(self, capsys):
        decision = runs.announce(None, "expand", sp.SOURCE_DEFAULT, unattended=True)
        assert capsys.readouterr().out == (
            "[AUTO_RESOLVED] scope-policy: expand (no --scope-policy given: the "
            "default for a new run)\n")
        assert decision.startswith("expand")

    def test_attended_or_typed_records_nothing(self, capsys):
        assert runs.announce(None, "expand", sp.SOURCE_DEFAULT, unattended=False) is None
        assert runs.announce(None, "preserve", sp.SOURCE_EXPLICIT, unattended=True) is None
        assert capsys.readouterr().out == ""

    def test_the_composite_core_cell_names_the_new_row_for_improve_commits(self):
        assert 16 in {r.number for r in un.ROWS if "improve-commits" in r.modes}


# ---------------------------------------------------------------------------
# 4. Replay, retry and detach carry the explicit value verbatim
# ---------------------------------------------------------------------------

class TestContinuationsCarryIt:
    def test_replay_flags_keep_it(self):
        argv = ["--plan-file", "/abs/p.md", "--quick", "--scope-policy", "preserve", "--yes"]
        assert cli_common.replay_flags("review-plan", argv) == [
            "--quick", "--scope-policy", "preserve", "--yes"]
        assert cli_common.replay_flags("apply-code-fixes", ["--scope-policy=expand"]) == [
            "--scope-policy=expand"]

    def test_a_retry_command_keeps_it_beside_the_verb(self):
        positionals, tokens = cli_common.retry_tokens(
            "plans/p.md", "--force", flags=["--scope-policy", "preserve", "--force"])
        assert positionals == ["plans/p.md"]
        assert tokens == ["--scope-policy", "preserve", "--force"]
        command = cli_common.retry_command(
            "review-plan", "plans/p.md", "--reaggregate", flags=["--scope-policy", "preserve"])
        assert command.endswith("review-plan plans/p.md --scope-policy preserve --reaggregate")

    def test_the_detached_worker_argv_keeps_it(self):
        argv = ["--plan-file", "p.md", "--scope-policy", "preserve", "--detach"]
        worker = detach.build_worker_argv("improve-code", argv, run_id="R1")
        index = worker.index("--scope-policy")
        assert worker[index + 1] == "preserve"
        assert worker.count("--scope-policy") == 1
        assert "--detach" not in worker and "--internal-worker" in worker

    def test_a_detached_run_record_replays_it(self):
        argv = ["--plan-file", "/abs/p.md", "--scope-policy", "preserve", "--detach",
                "--yes", "--exec-subagents"]
        replay = cli_common.replay_flags("implement", argv)
        assert replay == ["--scope-policy", "preserve", "--detach"]
        assert detach.decode_replay_flags(json.dumps(replay)) == replay

    def test_a_tool_route_resume_line_keeps_it(self):
        """``status``'s ``resume with:`` drops what a host composes, never the
        user's own policy."""
        tokens = ["--host", "codex", "--yes", "--scope-policy", "preserve", "--resume"]
        assert cli_common.drop_host_composed_flags(tokens) == [
            "--scope-policy", "preserve", "--resume"]

    def test_it_is_no_prefix_shadow(self):
        """No option is a strict prefix of it, and it of none."""
        assert sp.FLAG not in cli_common._EXACT_OPTIONS
        assert cli_common.flag_value(["--scope-policy", "preserve"], sp.FLAG) == "preserve"
        assert cli_common.flag_value(["--scope", "preserve"], sp.FLAG) == "preserve"


# ---------------------------------------------------------------------------
# 5. The run records (utils/scope_policy_runs.py)
# ---------------------------------------------------------------------------

def _stamp(minutes: float) -> str:
    return (datetime(2026, 10, 31, 12, 0, 0) + timedelta(minutes=minutes)).isoformat()


class TestRunRecords:
    def test_a_typed_value_is_explicit(self):
        assert runs.resolve_run_choice("preserve", {}, "full") == ("preserve", "explicit")
        assert runs.resolve_run_choice("expand", {}, "full") == ("expand", "explicit")

    def test_omitted_with_no_record_is_no_choice(self):
        assert runs.resolve_run_choice(None, {}, "full") is None
        assert runs.effective_choice(None) == ("expand", "default")
        assert runs.forwarded_flags(None) == []

    def test_a_live_record_answers_a_later_omission(self):
        state: dict = {}
        runs.record_run_choice(state, "improve-code", "preserve")
        assert state[runs.RUNS_KEY]["improve-code"]["source"] == "explicit"
        assert runs.resolve_run_choice(None, state, "improve-code") == (
            "preserve", "continuation")
        assert runs.forwarded_flags(("preserve", "continuation")) == [
            "--scope-policy", "preserve"]

    def test_a_typed_value_wins_over_the_record(self):
        state: dict = {}
        runs.record_run_choice(state, "full", "preserve")
        assert runs.resolve_run_choice("expand", state, "full") == ("expand", "explicit")

    def test_a_new_run_never_inherits(self):
        state: dict = {}
        runs.record_run_choice(state, "full", "preserve")
        assert runs.resolve_run_choice(None, state, "full", new_run=True) is None

    def test_the_finished_run_answers_nothing(self):
        state: dict = {}
        record = runs.record_run_choice(state, "improve-code", "preserve")
        record["recorded_at"] = _stamp(0)
        state["phases_completed"] = {"apply-code-fixes": _stamp(5)}
        assert runs.live_run_choice(state, "improve-code") is None
        # A completion from before the record is an earlier cycle's.
        state["phases_completed"] = {"apply-code-fixes": _stamp(-5)}
        assert runs.live_run_choice(state, "improve-code") == "preserve"
        # A newer skip finishes it too.
        state["phases_skipped"] = {"apply-code-fixes": {"skipped_at": _stamp(1)}}
        assert runs.live_run_choice(state, "improve-code") is None

    def test_closing_finishes_it(self):
        state: dict = {}
        runs.record_run_choice(state, "full", "preserve")
        assert runs.close_run_choice(state, "full") is True
        assert runs.close_run_choice(state, "full") is False
        assert runs.live_run_choice(state, "full") is None

    def test_keys_are_independent(self):
        state: dict = {}
        runs.record_run_choice(state, "implement", "preserve")
        assert runs.live_run_choice(state, "full") is None
        assert runs.live_run_choice(state, "implement") == "preserve"

    @pytest.mark.parametrize("mode,phases", [
        ("full", ("apply-code-fixes",)),
        ("improve-plan", ("apply-suggestions",)),
        ("improve-code", ("apply-code-fixes",)),
        ("improve-commits", ("apply-code-fixes",)),
        ("implement", ("implement",)),
        ("generate-tasks", ("generate-tasks",)),
    ])
    def test_each_run_ends_with_its_last_phase(self, mode, phases):
        assert runs.ending_phases(mode) == phases

    @pytest.mark.parametrize("record", [
        {"policy": "narrow", "recorded_at": _stamp(0)},
        {"policy": "preserve"},
        {"policy": "preserve", "recorded_at": "yesterday"},
        "preserve",
    ])
    def test_a_malformed_record_is_no_choice(self, record):
        assert runs.live_run_choice({runs.RUNS_KEY: {"full": record}}, "full") is None

    def test_the_prompt_blocks_are_empty_under_expand(self):
        for block in (runs.generation_guidance, runs.implementation_guidance):
            assert block("expand") == "" and block(None) == ""
            assert "Scope Policy: Preserve" in block("preserve")
        assert runs.with_guidance("x\n## Reply\ny", "") == "x\n## Reply\ny"
        assert runs.with_guidance("a ## Reply\nb\n## Reply\nc", "B", before="## Reply\n") == (
            "a ## Reply\nb\nB\n\n## Reply\nc")


# ---------------------------------------------------------------------------
# 6. Both packaged tool routes forward it as an ordinary extra flag
# ---------------------------------------------------------------------------

def _codex_server():
    source = REPO_ROOT / "packaging" / "codex" / "mcp_server.py"
    spec = importlib.util.spec_from_file_location("scope_policy_mcp_server", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestToolRoutes:
    @pytest.mark.parametrize("host", ["codex", "opencode"])
    @pytest.mark.parametrize("extra", [["--scope-policy", "preserve"], ["--scope-policy=expand"]])
    def test_extra_flags_keep_it(self, host, extra):
        sys.path.insert(0, str(REPO_ROOT / "packaging" / "common"))
        try:
            import surface as surface_mod
        finally:
            sys.path.pop(0)
        surface = surface_mod.cli_surface(host)
        server = _codex_server()
        assert server.reserved_hit(extra[0], surface) is None
        assert server.split_extra_flags(list(extra), surface) == (list(extra), [], None)
        row, error = server.validate_args(
            {"mode": "improve-code", "plan": "plans/p.md", "question": "",
             "extra_flags": list(extra)}, surface)
        assert error is None and row["name"] == "improve-code"


# ---------------------------------------------------------------------------
# 7. A next-phase command carries no policy: the next phase inherits or chooses
# ---------------------------------------------------------------------------

class TestNextPhaseCommands:
    def test_the_apply_after_a_preserve_review_is_named_without_it(
        self, tmp_path, monkeypatch, capsys
    ):
        """The documented choice (docs/development/dispatcher.md, "Shared
        flags"): the apply inherits the review round's recorded policy, so the
        command the user is handed repeats no flag, like every selector."""
        from utils.state_manager import StateManager

        monkeypatch.chdir(SKILL_ROOT)
        plan = tmp_path / "p.md"
        plan.write_text("# P\n\n## Step\nDo it.\n", encoding="utf-8")
        state = StateManager(plan)
        state.mark_phase_completed("review-plan")
        sp.record_review_round(state.state, "review-plan", "preserve", sp.SOURCE_EXPLICIT)
        state.save()
        code = multi_llm.main(["util", "next-command", str(plan), "--after", "review-plan",
                               "--host", "claude-code"])
        out = capsys.readouterr().out
        assert code == 0, out
        assert "--apply-suggestions" in out
        assert "--scope-policy" not in out
