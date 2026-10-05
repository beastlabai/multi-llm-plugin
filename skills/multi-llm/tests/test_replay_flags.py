"""D2 — the retry the plugin authors carries the caller's flags forward.

A ``--quick`` review stays a ``--quick`` review. Every retry command the tree
prints is built from one builder, ``cli_common.retry_command``, fed by one
contract, ``cli_common.replay_flags``, and the three retry routes — the phase
failing in this process, ``status`` on a run some other process launched, and
the durable triple a document stores — each rebuild the command from a
different source. These rows pin each source, the precedence between them, the
never-sanitize rule, the selector-fidelity line of the plan's Goal, and the
per-site shape ``cli_common.retry_shape`` picks.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

import modes as mode_table  # noqa: E402
from utils import cli_common, detach, run_status, status_file, validation  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.cli_common import (  # noqa: E402
    SHAPE_FORCE,
    SHAPE_REAGGREGATE,
    SHAPE_REPAIR,
    current_replay_flags,
    replay_flags,
    retry_command,
    retry_shape,
)
from utils.retry_ladder import (  # noqa: E402
    STATUS_INVALID_MODEL,
    STATUS_OK,
    STATUS_RATE_LIMITED,
    STATUS_RUNNER_ERROR,
    STATUS_TIMEOUT,
)

# The executor suite owns the mock runner harness; importing its fixture into
# this namespace is what makes ``runner_env`` resolvable here.
from tests.test_subagent_executor import (  # noqa: E402,F401
    _validation_batch,
    _write_validation_task_file,
    runner_env,
)

PLAN = "/tmp/a plan.md"


@pytest.fixture(autouse=True)
def _no_live_invocation():
    cli_common.clear_invocation()
    yield
    cli_common.clear_invocation()


def _argv(*extra, plan: str = PLAN):
    """The dispatcher's normalized argv: the plan positional already a flag."""
    return ["--plan-file", plan, *extra]


def _tokens(command: str):
    import shlex

    return shlex.split(command)


def _live(mode: str, *extra, plan: str = PLAN):
    cli_common.record_invocation(mode, _argv(*extra, plan=plan))


# ===========================================================================
# Step 1 — the replay-argv contract
# ===========================================================================

class TestReplayFlags:
    def test_quick_survives(self):
        assert replay_flags("review-plan", _argv("--quick")) == ["--quick"]

    def test_spellings_survive_as_typed(self):
        assert replay_flags("review-plan", _argv("-q", "--models=a,b")) == [
            "-q", "--models=a,b",
        ]
        assert replay_flags("review-plan", _argv("--models", "a,b")) == [
            "--models", "a,b",
        ]

    def test_argparse_abbreviations_are_recognized_not_passed_through(self):
        # `--internal-run` binds to `--internal-run-id` the way argparse would,
        # so the abbreviation vanishes with its value rather than surviving as
        # an unknown token; `--mod=a,b` is the caller's and stays as typed.
        assert replay_flags(
            "review-plan", _argv("--internal-run", "X", "--mod=a,b")
        ) == ["--mod=a,b"]

    def test_plan_path_with_a_space_round_trips_once_quoted(self):
        flags = replay_flags("review-plan", _argv("--quick"))
        command = retry_command("review-plan", PLAN, "--reaggregate", flags=flags)
        assert command.count(cli_common._quote(PLAN)) == 1
        assert "--plan-file" not in command
        assert _tokens(command)[-4:] == ["review-plan", PLAN, "--quick", "--reaggregate"]

    @pytest.mark.parametrize("spelling", [
        ["--internal-run-id", "8f2c-uuid"], ["--internal-run-id=8f2c-uuid"],
    ])
    def test_internal_run_id_vanishes_with_its_value(self, spelling):
        flags = replay_flags(
            "review-plan", _argv("--quick", "--internal-worker", *spelling)
        )
        # `--internal-worker` replays as the `--detach` it replaced.
        assert flags == ["--quick", "--detach"]
        assert "8f2c-uuid" not in " ".join(flags)

    @pytest.mark.parametrize("spelling", ["--yes", "-y", "--non-interactive"])
    def test_typed_yes_survives_in_every_spelling(self, spelling):
        assert replay_flags("review-plan", _argv(spelling, "--quick")) == [
            spelling, "--quick",
        ]
        assert replay_flags("review-plan", _argv("--exec-subagents")) == [
            "--exec-subagents",
        ]

    @pytest.mark.parametrize("spelling", ["--yes", "-y", "--non-interactive"])
    def test_implied_yes_and_exec_subagents_do_not_reappear(self, spelling):
        # `--detach` implies both (`multi_llm._apply_flag_implications`), so
        # under it the pair is dropped in every spelling and the implier stays:
        # the replay re-implies them.
        argv = _argv("--quick", "--detach", spelling, "--exec-subagents")
        assert replay_flags("review-plan", argv) == ["--quick", "--detach"]
        # And the worker's own argv, where `build_worker_argv` replaced
        # `--detach` with `--internal-worker` and appended the pair: the
        # re-entry flag is replayed as the `--detach` it stands for, so the
        # worker's retry re-implies the pair exactly as the launcher's does.
        worker = _argv(
            "--quick", spelling, "--exec-subagents", "--internal-worker",
            "--internal-run-id", "u",
        )
        assert replay_flags("review-plan", worker) == ["--quick", "--detach"]

    def test_the_drop_set_is_derived_from_the_parser_not_typed(self):
        """Every alias of the dropped dests comes from ``build_parent_parser``."""
        parser = cli_common.build_parent_parser()
        by_dest = {a.dest: tuple(a.option_strings) for a in parser._actions}
        assert cli_common._REPLAY_DROP_IMPLIED["yes"][0] == by_dest["yes"]
        assert set(by_dest["yes"]) == {"--yes", "-y", "--non-interactive"}
        assert cli_common._REPLAY_DROP_ALWAYS["internal_run_id"] == (
            by_dest["internal_run_id"], True,
        )
        assert cli_common._REPLAY_DROP_ALWAYS["internal_worker"] == (
            by_dest["internal_worker"], False,
        )

    def test_the_callers_own_cross_cutting_flags_survive(self):
        # The parser supplies spellings, not the decision: --host and --detach
        # are the caller's.
        assert replay_flags(
            "review-plan", _argv("--host", "opencode", "--detach", "--yes")
        ) == ["--host", "opencode", "--detach"]

    def test_no_positional_is_ever_recorded(self):
        assert replay_flags("review-plan", _argv()) == []
        assert replay_flags("ask", _argv("--question", "why?")) == []
        assert replay_flags("ask", _argv("--question-file", "/tmp/q.txt")) == []
        assert replay_flags(
            "ask", _argv("--question-file=/tmp/q.txt", "--quick")
        ) == ["--quick"]

    def test_a_long_question_replays_from_the_stored_text_never_the_temp_file(
        self, tmp_path
    ):
        question = "x" * 5000
        temp = tmp_path / "multi_llm_question_gone.txt"
        flags = replay_flags("ask", _argv("--question-file", str(temp), "--quick"))
        assert flags == ["--quick"]
        view = {
            "run": {"mode": "ask", "replay_flags": flags},
            "question": question, "progress": {},
        }
        command = run_status._resume_command(view, Path(PLAN))
        tokens = _tokens(command)
        assert tokens.count(question) == 1
        assert str(temp) not in command and "--question" not in command
        assert tokens[-3:] == [question, "--quick", "--force"]

    def test_a_short_question_is_recorded_nowhere_and_replayed_once(self):
        flags = replay_flags("ask", _argv("--question", "short one", "--quick"))
        assert flags == ["--quick"]
        env = detach.child_env(replay_flags=flags)
        assert "short one" not in env[detach.REPLAY_FLAGS_ENV]
        view = {"run": {"mode": "ask", "replay_flags": flags},
                "question": "short one", "progress": {}}
        command = run_status._resume_command(view, Path(PLAN))
        assert _tokens(command).count("short one") == 1
        # Missing key: today's questionless string.
        view["question"] = None
        assert "short one" not in run_status._resume_command(view, Path(PLAN))


# ===========================================================================
# Step 3 — one builder
# ===========================================================================

class TestRetryCommand:
    def test_no_flags_is_byte_identical_to_the_pre_d2_leaf(self):
        assert retry_command("review-plan", PLAN, "--reaggregate") == (
            cli_common.reaggregate_command("review-plan", PLAN)
        )
        assert retry_command("review-code", PLAN, "--force") == (
            cli_common.dispatcher_command("review-code", PLAN, "--force")
        )
        assert retry_command("ask", PLAN, "q?", "--force") == (
            cli_common.dispatcher_command("ask", PLAN, "q?", "--force")
        )

    def test_flags_sit_behind_the_positionals_and_ahead_of_the_verb(self):
        command = retry_command("ask", PLAN, "q?", "--force", flags=["--quick"])
        assert _tokens(command)[-5:] == ["ask", PLAN, "q?", "--quick", "--force"]

    def test_a_recorded_retry_verb_never_doubles(self):
        command = retry_command(
            "review-plan", PLAN, "--force", flags=["--quick", "--force"]
        )
        assert _tokens(command).count("--force") == 1
        command = retry_command(
            "review-plan", PLAN, "--force", flags=["--reaggregate", "--quick"]
        )
        assert "--reaggregate" not in command and "--quick" in command
        command = retry_command(
            "review-plan", PLAN, "--reaggregate-consolidation",
            flags=["--consolidate", "--quick"],
        )
        assert "--consolidate " not in command and "--quick" in command

    def test_never_sanitized_against_todays_parser(self):
        stale = "--flag-no-parser-in-this-tree-knows"
        command = retry_command("review-plan", PLAN, "--reaggregate", flags=[stale])
        assert stale in _tokens(command)
        # The dispatcher then refuses it loudly (exit 3), never silently.
        import multi_llm

        assert multi_llm.main(["review-plan", PLAN, stale, "--reaggregate"]) == 3


# ===========================================================================
# Step 2 — three sources, one precedence rule
# ===========================================================================

class TestPrecedence:
    def test_live_invocation_of_the_same_mode_wins_over_the_stored_triple(self):
        _live("review-plan", "--models", "a,b")
        document = {
            "reaggregate_mode": "review-plan", "plan_file": PLAN,
            "replay_flags": ["--quick"],
        }
        for reader in (validation.resolved_reaggregate_command,
                       detach.resolved_reaggregate_command):
            command = reader(document)
            assert "--models a,b --reaggregate" in command
            assert "--quick" not in command

    def test_a_different_live_mode_never_wins(self):
        # `status`'s own `--view run` argv is not the run's.
        _live("status", "--view", "run")
        document = {
            "reaggregate_mode": "review-plan", "plan_file": PLAN,
            "replay_flags": ["--quick"],
        }
        command = detach.resolved_reaggregate_command(document)
        assert "--quick --reaggregate" in command and "--view" not in command

    def test_documents_without_the_key_print_the_bare_verb(self):
        document = {"reaggregate_mode": "review-plan", "plan_file": PLAN}
        bare = cli_common.reaggregate_command("review-plan", PLAN)
        assert detach.resolved_reaggregate_command(document) == bare
        assert validation.resolved_reaggregate_command(document) == bare
        view = {"run": {"mode": "review-plan"}, "progress": {}}
        assert run_status._resume_command(view, Path(PLAN)) == bare

    def test_malformed_stored_flags_decode_to_nothing(self):
        assert current_replay_flags("--quick", mode="review-plan") == []
        assert current_replay_flags({"a": 1}, mode="review-plan") == []
        assert detach.decode_replay_flags("not json") == []
        assert detach.decode_replay_flags('{"a": 1}') == []
        assert detach.decode_replay_flags(None) == []
        assert detach.decode_replay_flags('["--quick"]') == ["--quick"]


# ===========================================================================
# The file route, once per document, read back by a second process
# ===========================================================================

class TestFileRoute:
    @pytest.mark.parametrize("selection", [["--quick"], ["--models", "a,b"]])
    def test_status_json_triple(self, tmp_path, selection):
        _live("review-plan", *selection)
        fields = cli_common.durable_reaggregate_fields(
            "review_plan_orchestrator.py", PLAN
        )
        assert fields["replay_flags"] == selection
        status_file.write_status(str(tmp_path), fields)
        # A second process: no live invocation, only the file.
        cli_common.clear_invocation()
        document = status_file.read_status(str(tmp_path))
        command = detach.resolved_reaggregate_command(document)
        assert command == retry_command(
            "review-plan", PLAN, "--reaggregate", flags=selection
        )
        assert " ".join(selection) + " --reaggregate" in command

    @pytest.mark.parametrize("selection", [["--quick"], ["--models", "a,b"]])
    def test_validation_tasks_json_triple(self, tmp_path, selection):
        _live("review-code", *selection)
        document = validation.prepare_batched_validation_tasks(
            groups=[{
                "theme": "T", "reference": "Task 1", "models": ["m"],
                "category": "modification", "importance": "MEDIUM",
                "suggestions": [{"title": "s", "desc": "d", "type": "modification",
                                 "importance": "MEDIUM"}],
            }],
            context="plan", output_dir=str(tmp_path), plan_file=PLAN,
            orchestrator="code_review_orchestrator.py",
        )
        assert document["replay_flags"] == selection
        cli_common.clear_invocation()
        stored = json.loads(json.dumps(document))
        command = validation.resolved_reaggregate_command(stored)
        assert " ".join(selection) + " --reaggregate" in command
        assert _tokens(command)[-4 - len(selection) + 1:][:1] == ["review-code"] or (
            "review-code" in _tokens(command)
        )

    def test_consolidation_tasks_json_composes_a_command_not_a_bare_verb(
        self, tmp_path, capsys
    ):
        import review_plan_orchestrator as rpo
        from utils.consolidation import prepare_consolidation_tasks

        _live("review-plan", "--quick")
        groups = [
            {"theme": f"T{i}", "reference": "Task 1", "models": ["m"],
             "category": "modification", "importance": "MEDIUM",
             "suggestions": [{"title": f"s{i}", "desc": "d",
                              "type": "modification", "importance": "MEDIUM"}]}
            for i in range(2)
        ]
        document = prepare_consolidation_tasks(
            groups, {"task 1": [0, 1]}, str(tmp_path), PLAN
        )
        assert document["reaggregate_command"] == "--reaggregate-consolidation"
        assert document["replay_flags"] == ["--quick"]

        # `consolidate_mode` is that document's writer *and* the printer of
        # its follow-up: it composes the runnable string through the builder
        # from the stored bare verb, the document's plan_file and the flags —
        # never printing the flag fragment alone.
        (tmp_path / "grouped.json").write_text(json.dumps(groups), encoding="utf-8")
        (tmp_path / "validation.json").write_text(json.dumps(
            [{"group_index": i, "status": "valid"} for i in range(2)]
        ), encoding="utf-8")
        rpo.consolidate_mode(str(tmp_path), PLAN, args=None)
        out = capsys.readouterr().out
        printed = [line.strip() for line in out.splitlines()
                   if "--reaggregate-consolidation" in line]
        assert printed, out
        assert printed[-1] == retry_command(
            "review-plan", PLAN, "--reaggregate-consolidation", flags=["--quick"]
        )
        assert printed[-1] != "--reaggregate-consolidation"
        stored = json.loads(
            (tmp_path / "consolidation_tasks.json").read_text(encoding="utf-8")
        )
        assert stored["replay_flags"] == ["--quick"]
        assert stored["reaggregate_command"] == "--reaggregate-consolidation"
        # A second process reading that document composes the same command
        # from the stored list, and never a bare verb.
        cli_common.clear_invocation()
        assert retry_command(
            "review-plan", stored["plan_file"], stored["reaggregate_command"],
            flags=current_replay_flags(stored["replay_flags"], mode="review-plan"),
        ) == printed[-1]

    def test_generate_tasks_status_document_carries_the_triple(self):
        _live("generate-tasks", "--models", "a,b")
        assert cli_common.current_replay_flags(mode="generate-tasks") == [
            "--models", "a,b",
        ]

    def test_a_stale_stored_spelling_is_emitted_verbatim_on_the_file_route(self):
        stale = "--flag-a-newer-plugin-dropped"
        document = {
            "reaggregate_mode": "review-plan", "plan_file": PLAN,
            "replay_flags": [stale, "--quick"],
        }
        assert stale in detach.resolved_reaggregate_command(document)
        assert stale in validation.resolved_reaggregate_command(document)


# ===========================================================================
# The Goal's selector-fidelity line, on all three routes
# ===========================================================================

class TestSelectorFidelity:
    """Edit the config between the failure and the replay: the replayed argv is
    token-identical, still the selector, never a resolved ``provider:model``."""

    def _config(self, tmp_path, monkeypatch, quick, defaults):
        path = tmp_path / "providers.yaml"
        path.write_text(
            "defaults:\n  models:\n" + "".join(f"    - {m}\n" for m in defaults)
            + "  quick_models:\n" + "".join(f"    - {m}\n" for m in quick),
            encoding="utf-8",
        )
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(path))
        from utils import provider_registry

        provider_registry._config = None
        provider_registry._config_key = None
        return path

    @pytest.mark.parametrize("selection", [["--quick"], []])
    def test_token_identical_across_a_config_edit(self, tmp_path, monkeypatch,
                                                  selection):
        self._config(tmp_path, monkeypatch, ["codex:gpt-5.5"], ["codex:gpt-5.5"])
        _live("review-plan", *selection)
        document = cli_common.durable_reaggregate_fields(
            "review_plan_orchestrator.py", PLAN
        )
        foreground_before = validation.resolved_reaggregate_command(document)
        record = {"mode": "review-plan", "replay_flags": document["replay_flags"]}
        status_before = run_status._resume_command(
            {"run": record, "progress": {}}, Path(PLAN)
        )
        cli_common.clear_invocation()
        file_before = detach.resolved_reaggregate_command(document)

        # The repair D3/D4 exist to make possible: a wider quick panel, and a
        # wider default panel for the flag-less run.
        self._config(
            tmp_path, monkeypatch,
            ["codex:gpt-5.5", "opencode:opencode/big-pickle", "claude-code:sonnet"],
            ["codex:gpt-5.5", "opencode:opencode/big-pickle", "claude-code:sonnet"],
        )
        _live("review-plan", *selection)
        assert validation.resolved_reaggregate_command(document) == foreground_before
        cli_common.clear_invocation()
        assert run_status._resume_command(
            {"run": record, "progress": {}}, Path(PLAN)
        ) == status_before
        assert detach.resolved_reaggregate_command(document) == file_before

        for command in (foreground_before, status_before, file_before):
            tokens = _tokens(command)
            assert ("--quick" in tokens) is bool(selection)
            assert "--models" not in tokens
            assert not any(":" in t and "/" not in t for t in tokens[-3:]), (
                f"a resolved spec leaked into the replay: {command}"
            )


# ===========================================================================
# Step 4 — the shape, per site
# ===========================================================================

class TestRetryShape:
    def test_takes_values_never_paths(self):
        import inspect

        params = inspect.signature(retry_shape).parameters
        assert set(params) == {"reason_code", "statuses", "artifacts_ok"}
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values())

    def test_the_three_shapes(self):
        assert retry_shape(reason_code="runner_invalid_model") == SHAPE_REPAIR
        assert retry_shape(statuses=[STATUS_INVALID_MODEL, STATUS_OK]) == SHAPE_REPAIR
        assert retry_shape(statuses=[STATUS_TIMEOUT, STATUS_RATE_LIMITED]) == SHAPE_REAGGREGATE
        assert retry_shape() == SHAPE_REAGGREGATE
        assert retry_shape(statuses=[STATUS_TIMEOUT], artifacts_ok=False) == SHAPE_FORCE

    def test_neither_a_count_nor_report_ok_decides(self):
        # 18 done, 10 invalid_model is shape 1; 0 done, 28 timeouts is shape 2.
        assert retry_shape(statuses=[STATUS_OK] * 18 + [STATUS_INVALID_MODEL] * 10) == SHAPE_REPAIR
        assert retry_shape(statuses=[STATUS_TIMEOUT] * 28) == SHAPE_REAGGREGATE


def _report(*statuses):
    return executor.BatchReport(
        kind="validation",
        outcomes=[
            executor.TaskOutcome(
                task_id=f"validation-batch-{i}", kind="validation",
                runner="codex:gpt-5.5", status=status,
            )
            for i, status in enumerate(statuses)
        ],
    )


class TestContinuePastMarkerShapes:
    """The site this incident printed: shaped from the report *it* was
    returned, artifacts from the document's grouped file."""

    ARGS = type("Args", (), {"exec_subagents": True, "yes": True, "host": "claude-code"})()
    RETRY = "uv run … multi_llm.py review-plan p.md --quick --reaggregate"

    def _block(self, tmp_path, report, document=None):
        return "\n".join(executor._shaped_retry_block(
            self.ARGS, report, phase_dir=str(tmp_path), document=document,
            failed_ids="validation-batch-0", retry_command=self.RETRY,
        ))

    def test_fatal_status_names_the_repair_above_the_command(self, tmp_path):
        text = self._block(tmp_path, _report(STATUS_INVALID_MODEL, STATUS_OK))
        assert text.index("Repair first:") < text.index(self.RETRY)
        assert "invalid_model" in text
        assert "--force is not needed" in text
        assert text.count(self.RETRY) == 1
        assert "--force`" not in text and " --force" not in text.replace(
            "--force is not needed", ""
        )

    def test_transient_status_offers_the_call_and_its_cost(self, tmp_path):
        text = self._block(tmp_path, _report(STATUS_TIMEOUT, STATUS_OK, STATUS_OK))
        assert "Repair first" not in text
        assert self.RETRY in text
        assert "re-dispatches the 1 failed task(s) and keeps the 2 valid result(s)" in text

    def test_partial_fatal_batch_still_offers_reaggregate_under_the_repair(
        self, tmp_path
    ):
        text = self._block(
            tmp_path, _report(*([STATUS_OK] * 18 + [STATUS_INVALID_MODEL] * 10))
        )
        assert "Repair first:" in text and self.RETRY in text
        assert "re-dispatches the 10 failed task(s) and keeps the 18 valid" in text

    def test_unusable_artifacts_make_force_the_command(self, tmp_path):
        document = {"grouped_file": str(tmp_path / "grouped.json")}  # absent
        text = self._block(tmp_path, _report(STATUS_RUNNER_ERROR), document)
        assert "review-plan p.md --quick --force" in text
        assert "--reaggregate" not in text
        (tmp_path / "grouped.json").write_text("[]", encoding="utf-8")
        text = self._block(tmp_path, _report(STATUS_RUNNER_ERROR), document)
        assert self.RETRY in text and "--force`" not in text

    def test_foreground_quick_hint_carries_quick_reaggregate(
        self, tmp_path, runner_env, monkeypatch, capsys
    ):
        """The foreground route end to end: a ``--quick`` run whose validation
        runner fails prints ``--quick --reaggregate`` in its exit-3 block."""
        from utils.review_orchestrator_base import dispatch_subagent_tasks

        runner_env.use("codex:gpt-5.5")
        monkeypatch.setenv("MOCK_LLM_FAIL", "1")
        task_file = _write_validation_task_file(
            tmp_path, [_validation_batch(tmp_path)]
        )
        _live("review-plan", "--quick", "--exec-subagents", "--yes",
              plan=str(tmp_path / "p.md"))
        document = json.loads(task_file.read_text(encoding="utf-8"))
        document.update(cli_common.durable_reaggregate_fields(
            "review_plan_orchestrator.py", str(tmp_path / "p.md")
        ))
        retry = validation.resolved_reaggregate_command(document)
        with pytest.raises(SystemExit) as excinfo:
            dispatch_subagent_tasks(
                self.ARGS, task_file=str(task_file), phase_dir=str(tmp_path),
                phase="review-plan", document=document, label="validation",
                retry_command=retry,
            )
        assert excinfo.value.code == 3
        err = capsys.readouterr().err
        assert "--quick --exec-subagents --yes --reaggregate" in err
        assert retry in err
        assert "--force`" not in err
        lines = [l for l in err.splitlines() if "--reaggregate" in l and "uv run" in l]
        assert lines and all("--quick" in l for l in lines)

    def test_precedence_a_models_run_resuming_a_quick_document(
        self, tmp_path, runner_env, monkeypatch, capsys
    ):
        from utils.review_orchestrator_base import dispatch_subagent_tasks

        runner_env.use("codex:gpt-5.5")
        monkeypatch.setenv("MOCK_LLM_FAIL", "1")
        task_file = _write_validation_task_file(
            tmp_path, [_validation_batch(tmp_path)]
        )
        document = json.loads(task_file.read_text(encoding="utf-8"))
        document.update({
            "reaggregate_mode": "review-plan", "plan_file": PLAN,
            "replay_flags": ["--quick"],
        })
        _live("review-plan", "--models", "a,b")
        retry = validation.resolved_reaggregate_command(document)
        with pytest.raises(SystemExit):
            dispatch_subagent_tasks(
                self.ARGS, task_file=str(task_file), phase_dir=str(tmp_path),
                phase="review-plan", document=document, label="validation",
                retry_command=retry,
            )
        err = capsys.readouterr().err
        assert "--models a,b --reaggregate" in err
        assert "--quick" not in err


class TestRefusalHandlerShape:
    def test_a_refusal_is_shape_one_with_the_recorded_flags(self, tmp_path, capsys):
        from utils.review_orchestrator_base import _print_executor_refusal

        failure = executor.RunnerRejected(
            "refused", reason_code="runner_invalid_model", key="driver.subagent_runner",
            spec="opencode:nope/nope", detail="Did you mean: x?",
        )
        retry = retry_command("review-plan", PLAN, "--reaggregate", flags=["--quick"])
        _print_executor_refusal(failure, phase_dir=str(tmp_path), retry_command=retry)
        err = capsys.readouterr().err
        assert err.index("Repair first:") < err.index(retry)
        assert "--quick --reaggregate" in err
        assert "--force is not needed" in err
        assert err.count("uv run") == 1
        status = status_file.read_status(str(tmp_path))
        assert status["state"] == "runner_rejected"
        assert status["reason_code"] == "runner_invalid_model"


# ===========================================================================
# The status route — reads the document, never the batch report
# ===========================================================================

def _status_out(plan: Path, capsys) -> str:
    code = run_status.main(["--plan-file", str(plan), "--view", "run"])
    assert code == 3
    return capsys.readouterr().out


def _failed_run(phase: Path, **extra):
    record = {
        "run_id": "run-d2-000001", "state": "failed", "exit_code": 3,
        "error": "needs_human (exit 3)", "detached": True, "mode": "review-plan",
        "replay_flags": ["--quick"],
    }
    record.update(extra)
    status_file.claim_run(str(phase), record)


class TestStatusRoute:
    def _plan(self, tmp_path):
        plan = tmp_path / "a plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        phase = tmp_path / plan.stem.replace(" ", "_") / "review-plan"
        phase.mkdir(parents=True)
        return plan, phase

    def test_runner_rejected_record_prints_the_repair_above_reaggregate(
        self, tmp_path, capsys
    ):
        plan, phase = self._plan(tmp_path)
        _failed_run(phase)
        status_file.update_status(str(phase), {
            "state": "runner_rejected", "reason_code": "runner_invalid_model",
        })
        out = _status_out(plan, capsys)
        assert out.index("repair first:") < out.index("resume with:")
        assert "runner_invalid_model" in out
        resume = [l for l in out.splitlines() if "resume with:" in l][0]
        assert "--quick --reaggregate" in resume and "--force" not in resume

    def test_mirror_with_an_invalid_model_row_is_shape_one(self, tmp_path, capsys):
        plan, phase = self._plan(tmp_path)
        _failed_run(phase)
        status_file.update_status(str(phase), {"progress": {"subagents": {
            "counts": {STATUS_INVALID_MODEL: 10, STATUS_OK: 18}, "ok": 18,
            "total": 28, "failed_task_ids": [f"t{i}" for i in range(10)],
        }}})
        out = _status_out(plan, capsys)
        assert "repair first:" in out and "invalid_model" in out
        assert "--quick --reaggregate" in out
        assert "re-dispatches the 10 failed task(s) and keeps the 18" in out

    def test_purely_retryable_mirror_is_shape_two(self, tmp_path, capsys):
        plan, phase = self._plan(tmp_path)
        _failed_run(phase)
        status_file.update_status(str(phase), {"progress": {"subagents": {
            "counts": {STATUS_TIMEOUT: 2, STATUS_OK: 3}, "ok": 3, "total": 5,
            "failed_task_ids": ["a", "b"],
        }}})
        out = _status_out(plan, capsys)
        assert "repair first:" not in out
        assert "--quick --reaggregate" in out

    def test_an_empty_phase_dir_is_shape_two(self, tmp_path, capsys):
        plan, phase = self._plan(tmp_path)
        _failed_run(phase)
        out = _status_out(plan, capsys)
        assert "repair first:" not in out
        assert "--quick --reaggregate" in out

    def test_a_stale_batch_report_is_never_read(self, tmp_path, capsys):
        plan, phase = self._plan(tmp_path)
        _failed_run(phase)
        (phase / executor.BATCH_REPORT_FILENAME).write_text(json.dumps({
            "tasks": [{"task_id": "STALE-TASK", "status": STATUS_INVALID_MODEL,
                       "reason": "STALE REASON"}] * 28,
        }), encoding="utf-8")
        status_file.update_status(str(phase), {"progress": {"subagents": {
            "counts": {STATUS_TIMEOUT: 1}, "ok": 0, "total": 1,
            "failed_task_ids": ["a"],
        }}})
        out = _status_out(plan, capsys)
        assert "repair first:" not in out
        assert "STALE" not in out

    def test_a_record_without_replay_flags_prints_the_bare_verb(self, tmp_path, capsys):
        plan, phase = self._plan(tmp_path)
        _failed_run(phase, replay_flags=None)
        status_file.update_status(str(phase), {})
        run = status_file.read_run(str(phase))
        run.pop("replay_flags", None)
        status = status_file.read_status(str(phase))
        status["run"] = run
        status_file.write_status(str(phase), status)
        out = _status_out(plan, capsys)
        assert cli_common.reaggregate_command("review-plan", str(plan)) in out

    def test_ask_replays_the_stored_question_once(self, tmp_path, capsys):
        plan = tmp_path / "p.md"
        plan.write_text("# p\n", encoding="utf-8")
        from utils.output_handler import slugify_question

        question = "what is the risk?"
        phase = tmp_path / "p" / "ask" / slugify_question(question)
        phase.mkdir(parents=True)
        status_file.write_status(str(phase), {"phase": "ask", "question": question})
        _failed_run(phase, mode="ask", replay_flags=["--quick"])
        code = run_status.main(["--plan-file", str(plan), "--view", "run"])
        out = capsys.readouterr().out
        assert code == 3, out
        resume = [l for l in out.splitlines() if "resume with:" in l][0]
        assert _tokens(resume.split("resume with:", 1)[1]).count(question) == 1
        assert "--question" not in resume and "--quick --force" in resume


class TestResumeVerbAgreesWithThePhase:
    """`status`'s `resume with:` and the phase's own retry hint are two
    renderings of one command. Measured 2026-09-14 on a failed
    `generate-tasks`: the hint said re-invoke plain, the status line said
    `--reaggregate`, the session model followed the status line, and the run
    ended on `--reaggregate found no task_generation_result.json` with nothing
    re-dispatched."""

    def _resume(self, mode, flags=("--host", "opencode", "--detach")):
        view = {"run": {"mode": mode, "replay_flags": list(flags)}, "progress": {}}
        return run_status._resume_command(view, Path(PLAN)).split()

    def test_generate_tasks_resumes_with_the_plain_re_invoke(self):
        tokens = self._resume("generate-tasks")
        assert "--reaggregate" not in tokens and "--force" not in tokens
        assert tokens[-4:] == ["--host", "opencode", "--detach", "--yes"]

    def test_a_recorded_reaggregate_flag_is_not_replayed_on_it(self):
        tokens = self._resume(
            "generate-tasks", ("--host", "opencode", "--skip-prerequisites", "--reaggregate")
        )
        assert "--reaggregate" not in tokens
        assert tokens[-4:] == ["--host", "opencode", "--skip-prerequisites", "--yes"]

    @pytest.mark.parametrize("mode", ["review-plan", "review-tasks", "review-code"])
    def test_the_review_phases_keep_reaggregate_which_re_dispatches_there(self, mode):
        assert self._resume(mode)[-1] == "--reaggregate"

    def test_it_matches_the_phases_own_retry_hint(self):
        """Same builder, same verb: the hint `generate_tasks_orchestrator`
        prints after a failed dispatch."""
        cli_common.clear_invocation()
        flags = ["--host", "opencode", "--skip-prerequisites"]
        hint = cli_common.retry_command("generate-tasks", PLAN, "--yes", flags=flags)
        assert self._resume("generate-tasks", flags) == hint.split()

    def test_the_detached_workers_own_hint_matches_it_too(self):
        """Racing re-run 2026-10-04 (codex #4): the worker's hint read
        `generate-tasks <plan> --host codex --yes`. Its argv carries
        `--internal-worker` where the launcher had `--detach`, the replay
        dropped both and the pair `--detach` implies, and the printed retry —
        run as printed — stops at TASK_GENERATION_PENDING and exits 0 having
        dispatched nothing. The worker now replays the `--detach` it was
        launched with, so its hint is the status line's command."""
        launcher = _argv("--host", "codex", "--detach", "--yes", "--exec-subagents")
        record = replay_flags("generate-tasks", launcher)
        worker = detach.build_worker_argv("generate-tasks", launcher, run_id="u")
        cli_common.record_invocation("generate-tasks", worker)
        hint = cli_common.retry_command(
            "generate-tasks", PLAN, "--yes",
            flags=cli_common.current_replay_flags(mode="generate-tasks"),
        )
        assert self._resume("generate-tasks", record) == hint.split()
        assert "--detach" in _tokens(hint)


class TestTheWritePhasesResumeWithResume:
    """`status`'s `resume with:` for the four write phases is `--resume`.

    That is the verb their own exit-3 block names ("re-invoke with --resume,
    which re-dispatches only the unfinished batches"); `--force` is the apply
    parsers' alias for `--yes`, under which the apply base re-ingests every
    human decision instead of replaying the recorded ones. Measured 2026-09-15
    on a failed `apply-task-suggestions`: the log said `--resume`, the status
    line said `--force`, and the session model ran `--claude-decide --force`
    twice before a human typed `--resume` by hand.
    """

    WRITE_MODES = ("apply-suggestions", "apply-task-suggestions",
                   "apply-code-fixes", "implement")

    def _resume(self, mode, flags):
        view = {"run": {"mode": mode, "replay_flags": list(flags)}, "progress": {}}
        return run_status._resume_command(view, Path(PLAN)).split()

    @pytest.mark.parametrize("mode", WRITE_MODES)
    def test_the_verb_is_resume_and_never_force(self, mode):
        tokens = self._resume(mode, ["--host", "opencode", "--detach"])
        assert tokens[-1] == "--resume"
        assert "--force" not in tokens and "--reaggregate" not in tokens
        assert tokens[-4:-1] == ["--host", "opencode", "--detach"]

    def test_the_selection_flags_survive_ahead_of_the_verb(self):
        tokens = self._resume(
            "apply-task-suggestions", ["--host", "opencode", "--claude-decide"]
        )
        assert tokens[-4:] == ["--host", "opencode", "--agent-decide", "--resume"]

    @pytest.mark.parametrize("recorded", [
        ["--claude-decide", "--agent-decide"],
        ["--let-claude-decide", "--claude-decide"],
        ["--agent-decide", "--agent-decide"],
    ])
    def test_every_judge_spelling_replays_once_as_agent_decide(self, recorded):
        tokens = self._resume("apply-suggestions", ["--host", "opencode", *recorded])
        assert tokens[-4:] == ["--host", "opencode", "--agent-decide", "--resume"]
        assert not {"--claude-decide", "--let-claude-decide"} & set(tokens)

    def test_a_judge_verb_absorbs_a_recorded_legacy_spelling(self):
        command = cli_common.retry_command(
            "apply-suggestions", PLAN, "--agent-decide",
            flags=["--claude-decide", "--quick"],
        ).split()
        assert command.count("--agent-decide") == 1
        assert "--claude-decide" not in command

    @pytest.mark.parametrize("mode", [
        "apply-suggestions", "apply-task-suggestions", "apply-code-fixes"])
    @pytest.mark.parametrize("unattended", [["--yes"], ["--detach"]])
    def test_the_judge_flag_yes_implies_is_not_replayed(self, mode, unattended):
        """OpenCode racing run 4: the tool route composes `--yes`, which
        already means `--agent-decide` in an apply phase, and `--resume`
        replays the judge's recorded verdicts from state.json. Replayed, the
        flag made `status`'s `resume with:` (offered as the tool call
        `--agent-decide --resume`) disagree with `resume_command`
        (`--resume`)."""
        launcher = _argv("--yes", "--agent-decide", "--host", "opencode",
                         *unattended)
        recorded = replay_flags(mode, launcher)
        assert "--agent-decide" not in recorded
        tokens = self._resume(mode, recorded)
        assert "--agent-decide" not in tokens
        assert tokens[-1] == "--resume"

    def test_a_judge_flag_recorded_beside_detach_is_dropped_when_read_back(self):
        """A run recorded before the drop reads the same as one after it."""
        tokens = self._resume(
            "apply-suggestions", ["--agent-decide", "--host", "opencode", "--detach"])
        assert tokens[-4:] == ["--host", "opencode", "--detach", "--resume"]
        assert "--agent-decide" not in tokens

    def test_without_yes_the_judge_flag_is_the_callers_choice(self):
        recorded = replay_flags("apply-suggestions", _argv("--agent-decide"))
        assert recorded == ["--agent-decide"]

    def test_a_recorded_resume_or_force_is_not_doubled(self):
        tokens = self._resume(
            "apply-task-suggestions", ["--host", "opencode", "--resume", "--force"]
        )
        assert tokens.count("--resume") == 1
        assert "--force" not in tokens

    @pytest.mark.parametrize("mode", WRITE_MODES[:3])
    def test_resuming_a_fresh_launch_does_not_clear_its_progress(self, mode):
        tokens = self._resume(
            mode, ["--host", "opencode", "--fresh", "--claude-decide"]
        )
        assert "--fresh" not in tokens
        assert tokens[-4:] == ["--host", "opencode", "--agent-decide", "--resume"]

    def test_the_cost_line_counts_batches_and_never_mentions_the_panel(self):
        view = {
            "run": {"mode": "apply-task-suggestions", "replay_flags": []},
            "phase": "apply-task-suggestions",
            "progress": {"write_subagents": {
                "kind": "apply_task_suggestions", "total": 12,
                "tasks": [
                    {"task_id": "apply-task-batch-0", "status": "ok"},
                    {"task_id": "apply-task-batch-1", "status": "unparseable_result"},
                ],
                "failed_task_ids": ["apply-task-batch-1"],
            }},
        }
        lines = run_status._resume_lines(view, Path(PLAN))
        assert lines[0].strip().startswith("resume with:")
        assert lines[0].rstrip().endswith("--resume")
        assert "1 failed batch(es)" in lines[1]
        assert "keeps the 1 already applied" in lines[1]
        assert "restore-batch" in lines[1]
        assert "reviewer panel" not in lines[1]

    @staticmethod
    def _refused_before_batch_one(mode="implement", dispatched=()):
        return {
            "run": {"mode": mode, "replay_flags": ["--host", "codex", "--detach"]},
            "phase": mode_table.resolve(mode).phase,
            "progress": {"write_subagents": {
                "kind": mode.replace("-", "_"), "total": 2,
                "batches_dispatched": list(dispatched),
                "tasks": [
                    {"task_id": "T001", "status": "rejected"},
                    {"task_id": "T002", "status": "rejected"},
                ],
                "failed_task_ids": ["T001", "T002"],
            }},
        }

    @pytest.mark.parametrize("mode", WRITE_MODES)
    def test_a_refusal_before_the_first_batch_offers_no_resume(self, mode):
        """Racing re-run 2026-10-04 (codex #5): implement was refused on the
        jail before batch 1, and the tool result offered `--resume`, which
        repeats the refusal. Nothing dispatched and nothing on record: the
        way forward is the remedy the refusal names, so no command."""
        view = self._refused_before_batch_one(mode)
        workflow = {"phases": {mode_table.resolve(mode).phase: {"status": "pending"}}}
        lines = run_status._resume_lines(view, Path(PLAN), workflow=workflow)
        assert not any("resume with:" in line for line in lines)
        assert not any("--resume" in line or "multi_llm.py" in line for line in lines)
        assert lines and lines[0].strip().startswith("no resume:")

    def test_progress_on_record_keeps_the_resume(self):
        """A `--resume`d run refused before its own first batch still has the
        earlier run's completed batches to skip: `in_progress` keeps it."""
        view = self._refused_before_batch_one()
        workflow = {"phases": {"implement": {"status": "in_progress"}}}
        lines = run_status._resume_lines(view, Path(PLAN), workflow=workflow)
        assert lines[0].strip().startswith("resume with:")

    def test_a_dispatched_batch_keeps_the_resume(self):
        view = self._refused_before_batch_one(dispatched=[0])
        workflow = {"phases": {"implement": {"status": "pending"}}}
        lines = run_status._resume_lines(view, Path(PLAN), workflow=workflow)
        assert lines[0].strip().startswith("resume with:")

    def test_the_status_view_of_such_a_run_names_no_resume(self, tmp_path, capsys):
        """End to end through `status --view run`, which reads the workflow
        itself (the tool routes poll only that view)."""
        plan = tmp_path / "plans" / "p.md"
        plan.parent.mkdir()
        plan.write_text("# p\n", encoding="utf-8")
        phase = plan.parent / "p" / "implement"
        phase.mkdir(parents=True)
        status_file.claim_run(str(phase), {
            "run_id": "run-refused-1", "state": "failed", "exit_code": 3,
            "error": "needs_human (exit 3)", "detached": True, "mode": "implement",
            "replay_flags": ["--host", "codex", "--detach"],
        })
        status_file.update_status(str(phase), {
            "progress": self._refused_before_batch_one()["progress"]})
        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 3
        out = capsys.readouterr().out
        assert "no resume:" in out, out
        assert "resume with:" not in out, out

    def test_the_review_phases_keep_their_own_cost_line(self):
        view = {
            "run": {"mode": "review-tasks", "replay_flags": []},
            "progress": {"subagents": {
                "kind": "validation", "total": 8, "ok": 6,
                "counts": {"ok": 6, "timeout": 2},
                "failed_task_ids": ["validation-batch-0", "validation-batch-3"],
            }},
        }
        lines = run_status._resume_lines(view, Path(PLAN))
        assert lines[0].rstrip().endswith("--reaggregate")
        assert "re-dispatches the 2 failed task(s)" in lines[1]
        assert "reviewer panel" in lines[1]
