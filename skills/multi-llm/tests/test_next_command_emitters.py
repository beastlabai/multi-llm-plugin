"""The three emitters of the next-phase command (plans/next-phase-command.md T5).

``multi_llm.main`` (foreground, or as the detached worker), ``run_status``'s
terminal poll and ``util next-command`` each pass the one builder's answer
through as the ``[NEXT_COMMAND]`` block and ``MULTI_LLM_RESULT.user_next_step``
(§2.2). What is pinned here:

* the gate and the exclusions (§2.1, §2.2(1)): which invocations carry the key
  and the block, and which carry neither (§5 criterion 3);
* where ``plan_arg`` comes from in each emitter (§2.1, "Where each emitter gets
  its inputs"): the typed spelling, relative when it was typed relative, even
  where the emitting process holds it absolute;
* the streams: the block on stdout, or on stderr when the invocation asked for
  JSON output, with the ``--json`` payload unchanged;
* §5 criterion 2: on all six host/route rows the ``[NEXT_COMMAND]`` line, the
  result line's ``user_next_step.command`` and the builder's output are one
  string.

Every dispatcher run here is in-process against a fixture ``state.json`` built
by ``tests/test_next_phase.py``'s helpers; the phase targets are stubbed, since
the builder reads only the recorded state.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import check_workflow_prerequisites as prereq  # noqa: E402
import multi_llm  # noqa: E402
from utils import cli_common, detach, next_phase, process_control  # noqa: E402
from utils import run_status, status_file  # noqa: E402
from utils.next_phase import (  # noqa: E402
    LOOKUP_FAILED_PREFIX,
    NEXT_COMMAND_MARKER,
    NEXT_COMMAND_OPTIONAL_MARKER,
    NEXT_COMMAND_OPTIONAL_WHY_MARKER,
    WORKFLOW_COMPLETE_MARKER,
    NextPhase,
    OptionalStep,
)
from utils.output_handler import output_dir_path  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402

from tests.test_next_phase import (  # noqa: E402
    ALL_REQUIRED,
    CR_CLEAN,
    CR_FINDING,
    NOTICE_ONE_GROUP,
    P,
    ROUTES,
    ROUTE_IDS,
    RP_VALID,
    RT_ACTIONABLE,
    RT_CLEAN,
    UP_TO_CODE_REVIEW,
    UP_TO_IMPLEMENT,
    UP_TO_REVIEW_TASKS,
    UP_TO_TASKS,
    WHY_REVIEW_TASKS,
    S,
    _build,
    _merge,
)

SKILL_ROOT = Path(__file__).parent.parent
REAL_LOAD_TARGET = multi_llm._load_target
PLUGIN = "/multi-llm:multi-llm "
BLOCK_MARKERS = (
    NEXT_COMMAND_MARKER, NEXT_COMMAND_OPTIONAL_WHY_MARKER,
    NEXT_COMMAND_OPTIONAL_MARKER, WORKFLOW_COMPLETE_MARKER,
)
EXEC = "--exec-subagents"

#: One fixture per mode: the state a finished run of that mode leaves, the
#: flags that put a write phase on its executor leg, and the answer (args after
#: the route prefix, with any optional entries; or "complete").
MODE_ROWS = [
    ("review-plan", S(("review-plan",), files=RP_VALID), (),
     (f"--apply-suggestions {P}", [])),
    ("apply-suggestions", S(("review-plan", "apply-suggestions"), files=RP_VALID),
     (EXEC,), (f"--generate-tasks {P}", [])),
    ("generate-tasks", S(UP_TO_TASKS), (),
     (f"--implement {P}", [("review-tasks", f"--review-tasks {P}", WHY_REVIEW_TASKS)])),
    ("review-tasks", S(UP_TO_REVIEW_TASKS, files=RT_ACTIONABLE), (),
     (f"--implement {P}",
      [("apply-task-suggestions", f"--apply-task-suggestions {P}", NOTICE_ONE_GROUP)])),
    ("apply-task-suggestions",
     S(UP_TO_REVIEW_TASKS + ("apply-task-suggestions",), files=RT_ACTIONABLE),
     (EXEC,), (f"--implement {P}", [])),
    ("implement", S(UP_TO_IMPLEMENT), (EXEC,), (f"--review-code {P}", [])),
    ("review-code", S(UP_TO_CODE_REVIEW, files=CR_FINDING), (),
     (f"--apply-code-fixes {P}", [])),
    ("apply-code-fixes", S(ALL_REQUIRED, files=CR_FINDING), (EXEC,), "complete"),
    ("full",
     S(ALL_REQUIRED + ("review-tasks",), skipped=("apply-task-suggestions",),
       files=_merge(RT_CLEAN, CR_FINDING)),
     (), "complete"),
]
MODE_IDS = [row[0] for row in MODE_ROWS]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """cwd at the fixture root, so ``plans/foo.md`` is a relative plan path."""
    monkeypatch.chdir(tmp_path)
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE", detach.PLAN_ARG_ENV,
                 detach.WORKER_ENV):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def build(workdir, spec):
    """``tests/test_next_phase._build``, from inside the repository.

    A plan's first ``StateManager`` records ``git rev-parse HEAD`` from the
    process's cwd, which the ``workdir`` fixture moved out of any repository.
    """
    here = os.getcwd()
    os.chdir(SKILL_ROOT)
    try:
        return _build(workdir, spec)
    finally:
        os.chdir(here)


def _target(monkeypatch, body=None, code=0):
    """Stub every mode's target: run *body(argv)* (if any), return *code*."""
    seen = {}

    def loader(module_path, attr):
        def target(argv):
            seen["argv"] = list(argv)
            if body is not None:
                result = body(argv)
                if result is not None:
                    return result
            return code

        return target

    monkeypatch.setattr(multi_llm, "_load_target", loader)
    return seen


def _run(argv, capsys):
    """``multi_llm.main(argv)`` → (exit code, stdout, stderr, result payload)."""
    code = multi_llm.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err, cli_common.parse_result_line(captured.out)


def _block(text):
    return [
        line for line in text.splitlines()
        if line.startswith(BLOCK_MARKERS)
    ]


def _expected_payload(expect, prefix):
    if expect == "complete":
        return {"phase": None, "command": None, "complete": True}
    primary_args, optional = expect
    import modes

    payload = {
        "phase": modes.phase_for(primary_args.split()[0][2:]),
        "command": prefix + primary_args,
    }
    if optional:
        payload["optional_before"] = [
            {"phase": phase, "command": prefix + args, "why": why}
            for phase, args, why in optional
        ]
    payload["complete"] = False
    return payload


def _assert_neither(out, err, result):
    assert "user_next_step" not in result, result
    assert _block(out) == [] and _block(err) == [], (out, err)


def _assert_block_matches_key(block, key):
    """§2.2(2): primary first, why directly above its command, commands verbatim."""
    if key["complete"]:
        assert block == [WORKFLOW_COMPLETE_MARKER]
        return
    assert block[0] == f"{NEXT_COMMAND_MARKER} {key['command']}"
    optional = key.get("optional_before", [])
    assert len(block) == 1 + 2 * len(optional)
    for index, step in enumerate(optional):
        assert block[1 + 2 * index] == f"{NEXT_COMMAND_OPTIONAL_WHY_MARKER} {step['why']}"
        assert block[2 + 2 * index] == f"{NEXT_COMMAND_OPTIONAL_MARKER} {step['command']}"


def _link_skill_root(tmp_path, monkeypatch, *parts):
    """Point the dispatcher's skill root at a symlink named *parts* (judged by
    its own path, never its target), so a route can be chosen by location."""
    link = tmp_path.joinpath("roots", *parts)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(SKILL_ROOT, target_is_directory=True)
    monkeypatch.setattr(cli_common, "_SKILL_DIR", link)
    return link


def _select_route(tmp_path, monkeypatch, host, route):
    """Put this process on (host, route): the skill root on claude-code, the
    tool servers' ``MULTI_LLM_ROUTE`` on the other two."""
    if host == "claude-code":
        if route == "skill":
            _link_skill_root(tmp_path, monkeypatch, "project", ".claude", "skills",
                             "multi-llm")
        else:
            _link_skill_root(tmp_path, monkeypatch, "plugins", "cache", "multi-llm")
    elif route:
        monkeypatch.setenv("MULTI_LLM_ROUTE", route)


def _terminal_record(plan, phase_dir_name, **run):
    """A finished detached run's checkpoint, without spawning anything."""
    phase_dir = output_dir_path(plan) / phase_dir_name
    phase_dir.mkdir(parents=True, exist_ok=True)
    status_file.write_status(str(phase_dir), {
        "phase": phase_dir_name, "state": "complete",
    })
    record = {"run_id": "run-t5-0001", "state": "done", "exit_code": 0,
              "detached": True}
    record.update(run)
    status_file.claim_run(str(phase_dir), record)
    return phase_dir


# ---------------------------------------------------------------------------
# multi_llm.main: each mode, the completion case, exit 2
# ---------------------------------------------------------------------------

class TestEachModeEndsWithTheBlockAndTheKey:
    @pytest.mark.parametrize("mode,spec,flags,expect", MODE_ROWS, ids=MODE_IDS)
    def test_a_finished_run_carries_both(self, workdir, capsys, monkeypatch,
                                         mode, spec, flags, expect):
        build(workdir, spec)
        _target(monkeypatch)
        code, out, err, result = _run([mode, P, *flags], capsys)
        assert code == 0, err
        key = result["user_next_step"]
        assert key == _expected_payload(expect, PLUGIN)
        _assert_block_matches_key(_block(out), key)
        # The block sits directly above the result line.
        lines = out.splitlines()
        assert lines[-1].startswith(cli_common.RESULT_MARKER)
        assert lines[-1 - len(_block(out)):-1] == _block(out)

    @pytest.mark.parametrize("mode,spec,flags", [
        ("apply-code-fixes", S(ALL_REQUIRED, files=CR_FINDING), (EXEC,)),
        ("review-code", S(UP_TO_CODE_REVIEW, files=CR_CLEAN), ()),
        ("full", S(ALL_REQUIRED + ("review-tasks",),
                   skipped=("apply-task-suggestions",),
                   files=_merge(RT_CLEAN, CR_FINDING)), ()),
    ], ids=["apply-code-fixes", "clean-review-code", "full"])
    def test_completing_the_workflow_prints_workflow_complete(
        self, workdir, capsys, monkeypatch, mode, spec, flags
    ):
        build(workdir, spec)
        _target(monkeypatch)
        code, out, err, result = _run([mode, P, *flags], capsys)
        assert code == 0, err
        assert result["user_next_step"] == {
            "phase": None, "command": None, "complete": True,
        }
        assert _block(out) == [WORKFLOW_COMPLETE_MARKER]

    @pytest.mark.parametrize("mode,spec,expect", [
        ("review-plan", S(("review-plan",), files=RP_VALID),
         f"--apply-suggestions {P}"),
        ("review-code", S(UP_TO_CODE_REVIEW, files=CR_FINDING),
         f"--apply-code-fixes {P}"),
    ], ids=["review-plan", "review-code"])
    def test_an_exit_2_rerun_of_a_completed_review_carries_both(
        self, workdir, capsys, monkeypatch, mode, spec, expect
    ):
        """The phase-complete guard's `sys.exit(2)`, not argparse's usage exit."""
        build(workdir, spec)

        def guard(argv):
            raise SystemExit(2)

        _target(monkeypatch, guard)
        code, out, err, result = _run([mode, P], capsys)
        assert code == 2
        assert result["state"] == "nothing-to-do"
        assert result["user_next_step"]["command"] == PLUGIN + expect
        assert _block(out) == [f"{NEXT_COMMAND_MARKER} {PLUGIN}{expect}"]

    def test_the_mode_word_is_the_resolved_one_so_an_alias_changes_nothing(
        self, workdir, capsys, monkeypatch
    ):
        build(workdir, S(UP_TO_CODE_REVIEW, files=CR_FINDING))
        _target(monkeypatch)
        _code, out, _err, result = _run(["code-review", P], capsys)
        assert result["mode"] == "review-code"
        assert result["user_next_step"]["command"] == f"{PLUGIN}--apply-code-fixes {P}"

    def test_a_relative_plan_prints_relative_and_an_absolute_one_absolute(
        self, workdir, capsys, monkeypatch
    ):
        plan = build(workdir, S(("review-plan",), files=RP_VALID))
        _target(monkeypatch)
        _code, out, _err, result = _run(["review-plan", P], capsys)
        assert result["user_next_step"]["command"] == f"{PLUGIN}--apply-suggestions {P}"
        _code, out, _err, result = _run(["review-plan", str(plan)], capsys)
        assert result["user_next_step"]["command"] == (
            f"{PLUGIN}--apply-suggestions {plan}"
        )


# ---------------------------------------------------------------------------
# multi_llm.main: the invocations that never carry it (§5 criterion 3)
# ---------------------------------------------------------------------------

class TestNoCommandWhereNoneBelongs:
    @pytest.mark.parametrize("mode,spec", [
        ("apply-task-suggestions", S(UP_TO_REVIEW_TASKS, files=RT_ACTIONABLE)),
        ("apply-code-fixes", S(UP_TO_CODE_REVIEW, files=CR_FINDING)),
    ], ids=["apply-task-suggestions", "apply-code-fixes"])
    def test_a_native_first_call_of_an_apply_phase(self, workdir, capsys,
                                                   monkeypatch, mode, spec):
        build(workdir, spec)
        _target(monkeypatch, lambda argv: print('{"batches": []}'))
        code, out, err, result = _run([mode, P], capsys)
        assert code == 0
        _assert_neither(out, err, result)

    @pytest.mark.parametrize("mode,spec", [
        ("apply-suggestions", S(("review-plan",), files=RP_VALID)),
        ("implement", S(UP_TO_TASKS)),
    ], ids=["apply-suggestions", "implement"])
    def test_a_native_write_leg_whose_record_is_written_at_emit_time(
        self, workdir, capsys, monkeypatch, mode, spec
    ):
        """The gate passes here — both phases mark themselves completed when
        they emit their batch document — so the exclusion is what holds."""
        plan = build(workdir, spec)

        def emit_batches(argv):
            state = StateManager(plan)
            state.mark_phase_completed(mode)
            state.save()
            print('{"batches": [{"batch_index": 0}]}')

        _target(monkeypatch, emit_batches)
        code, out, err, result = _run([mode, P], capsys)
        assert code == 0
        _assert_neither(out, err, result)
        # The builder alone would have answered: the invocation is excluded.
        assert next_phase.next_phase_commands(P, "claude-code", None, mode) is not None

    @pytest.mark.parametrize("mode,spec,marker", [
        ("review-plan", S(), "[VALIDATION_PENDING]"),
        ("review-tasks", S(UP_TO_TASKS), "[VALIDATION_PENDING]"),
        ("review-code", S(UP_TO_IMPLEMENT), "[SALVAGE_NEEDED]"),
    ], ids=["review-plan", "review-tasks", "review-code"])
    def test_a_review_stopped_at_a_pending_marker(self, workdir, capsys,
                                                  monkeypatch, mode, spec, marker):
        """Exit 0 with the phase still pending on the native leg: no answer,
        although review-tasks leaves `suggested_next` at implement."""
        build(workdir, spec)
        _target(monkeypatch, lambda argv: print(f"{marker} dispatch the batch"))
        code, out, err, result = _run([mode, P], capsys)
        assert code == 0
        _assert_neither(out, err, result)

    @pytest.mark.parametrize("spec", [
        S(),
        S(("review-plan",), files=RP_VALID),
    ], ids=["fresh", "already-on-record"])
    def test_a_detach_invoker(self, workdir, capsys, monkeypatch, spec):
        """Fresh (review-plan not on record) or re-launching a recorded phase:
        the invoker never answers; its run's terminal poll does. The stub
        records the poll command as `next`, as `detach.launch` does on both of
        its success paths (TestTheDetachedRoute drives the real one)."""
        build(workdir, spec)

        def launched(argv):
            cli_common.record_next(detach.status_command_for(P, phase="review-plan"))

        seen = _target(monkeypatch, launched)
        code, out, err, result = _run(["review-plan", P, "--detach"], capsys)
        assert code == 0
        assert "--detach" in seen["argv"] and EXEC in seen["argv"]
        _assert_neither(out, err, result)

    def test_a_detach_invoker_whose_phase_is_on_record_and_exits_2(
        self, workdir, capsys, monkeypatch
    ):
        """The completed-phase guard exits 2 before the detach boundary, so no
        run is launched and no poll will ever answer: the invoker must. The tool
        routes append `--detach` to every launch, so this is their re-run."""
        build(workdir, S(("review-plan",), files=RP_VALID))

        def guard(argv):
            raise SystemExit(2)

        _target(monkeypatch, guard)
        code, out, err, result = _run(["review-plan", P, "--detach"], capsys)
        assert code == 2
        expect = f"{PLUGIN}--apply-suggestions {P}"
        assert result["user_next_step"]["command"] == expect
        assert _block(out) == [f"{NEXT_COMMAND_MARKER} {expect}"]

    @pytest.mark.parametrize("code", [1, 3])
    def test_a_failed_or_needs_human_exit(self, workdir, capsys, monkeypatch, code):
        build(workdir, S(("review-plan",), files=RP_VALID))
        _target(monkeypatch, code=code)
        exit_code, out, err, result = _run(["review-plan", P], capsys)
        assert exit_code == code
        _assert_neither(out, err, result)

    def test_a_mechanical_continuation_still_owed(self, workdir, capsys, monkeypatch):
        """A non-null `next` is the caller's follow-up, never the user's step."""
        build(workdir, S(("review-plan",), files=RP_VALID))
        _target(
            monkeypatch,
            lambda argv: cli_common.record_next("multi_llm.py review-plan p --reaggregate"),
        )
        code, out, err, result = _run(["review-plan", P], capsys)
        assert code == 0 and result["next"]
        _assert_neither(out, err, result)

    @pytest.mark.parametrize("argv", [
        ["ask", P, "why?"],
        ["init"],
        ["doctor"],
    ])
    def test_a_mode_that_is_no_phase(self, workdir, capsys, monkeypatch, argv):
        build(workdir, S(("review-plan",), files=RP_VALID))
        _target(monkeypatch)
        _code, out, err, result = _run(argv, capsys)
        _assert_neither(out, err, result)

    def test_a_plan_with_nothing_recorded_creates_nothing(
        self, workdir, capsys, monkeypatch
    ):
        """No state.json: the gate could only refuse, and nothing is read, so a
        run on an unrecorded plan does not grow an output dir."""
        _target(monkeypatch)
        code, out, err, result = _run(["review-plan", "plans/none.md"], capsys)
        assert code == 0
        _assert_neither(out, err, result)
        assert not (workdir / "plans").exists()

    def test_a_builder_raise_costs_one_stderr_line_and_nothing_else(
        self, workdir, capsys, monkeypatch
    ):
        build(workdir, S(("review-plan",), files=RP_VALID))
        _target(monkeypatch)

        def boom(*args, **kwargs):
            raise ValueError("no template for\nthat route")

        monkeypatch.setattr(next_phase, "next_phase_commands", boom)
        code, out, err, result = _run(["review-plan", P], capsys)
        assert code == 0
        assert result["state"] == "done"
        _assert_neither(out, err, result)
        assert err.splitlines() == [
            f"{LOOKUP_FAILED_PREFIX} (ValueError: no template for that route); "
            "no next command printed"
        ]


# ---------------------------------------------------------------------------
# multi_llm.main: JSON output, and the block's layout
# ---------------------------------------------------------------------------

class TestStreamsAndLayout:
    def test_an_apply_phase_under_output_format_json_keeps_stdout_clean(
        self, workdir, capsys, monkeypatch
    ):
        build(workdir, S(("review-plan", "apply-suggestions"), files=RP_VALID))
        document = {"batches": [{"batch_index": 0}], "status": "applied"}
        _target(monkeypatch, lambda argv: print(json.dumps(document, indent=2)))
        code, out, err, result = _run(
            ["apply-suggestions", P, "--output-format", "json", EXEC], capsys
        )
        assert code == 0
        body = out.rsplit(cli_common.RESULT_MARKER, 1)[0]
        assert json.loads(body) == document
        assert _block(out) == []
        assert _block(err) == [f"{NEXT_COMMAND_MARKER} {PLUGIN}--generate-tasks {P}"]
        assert result["user_next_step"]["command"] == f"{PLUGIN}--generate-tasks {P}"

    def test_primary_first_why_above_its_command_commands_verbatim(
        self, workdir, capsys, monkeypatch
    ):
        """Two optional entries (a shape no fixture state reaches through the
        walk today), passed through untouched by the emitter."""
        build(workdir, S(UP_TO_TASKS))
        answer = NextPhase(
            phase="implement",
            command=f'{PLUGIN}--implement "plans/my plan.md"',
            optional_before=(
                OptionalStep("review-tasks", f"{PLUGIN}--review-tasks x", "why one"),
                OptionalStep("apply-task-suggestions",
                             f"{PLUGIN}--apply-task-suggestions x --resume", "why two"),
            ),
        )
        monkeypatch.setattr(next_phase, "next_phase_commands", lambda *a, **k: answer)
        _target(monkeypatch)
        _code, out, _err, result = _run(["generate-tasks", P], capsys)
        key = result["user_next_step"]
        assert key == answer.to_dict()
        block = _block(out)
        assert block == [
            f'{NEXT_COMMAND_MARKER} {PLUGIN}--implement "plans/my plan.md"',
            f"{NEXT_COMMAND_OPTIONAL_WHY_MARKER} why one",
            f"{NEXT_COMMAND_OPTIONAL_MARKER} {PLUGIN}--review-tasks x",
            f"{NEXT_COMMAND_OPTIONAL_WHY_MARKER} why two",
            f"{NEXT_COMMAND_OPTIONAL_MARKER} {PLUGIN}--apply-task-suggestions x --resume",
        ]
        _assert_block_matches_key(block, key)


# ---------------------------------------------------------------------------
# §5 criterion 2: one string on all six routes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("host,route,prefix", ROUTES, ids=ROUTE_IDS)
@pytest.mark.parametrize("mode,spec,flags,expect", MODE_ROWS, ids=MODE_IDS)
def test_criterion_2_block_key_and_builder_are_one_string(
    workdir, capsys, monkeypatch, host, route, prefix, mode, spec, flags, expect
):
    build(workdir, spec)
    _select_route(workdir, monkeypatch, host, route)
    _target(monkeypatch)
    code, out, err, result = _run([mode, P, "--host", host, *flags], capsys)
    assert code == 0, err
    assert err == ""  # a valid route signal is honoured silently
    built = next_phase.next_phase_commands(P, host, route, mode)
    key = result["user_next_step"]
    assert key == built.to_dict() == _expected_payload(expect, prefix)
    block = _block(out)
    assert block == next_phase.render_next_phase_lines(built)
    _assert_block_matches_key(block, key)
    if not built.complete:
        assert block[0][len(NEXT_COMMAND_MARKER) + 1:] == key["command"] == built.command
        for step, line in zip(built.optional_before, block[2::2]):
            assert line[len(NEXT_COMMAND_OPTIONAL_MARKER) + 1:] == step.command


# ---------------------------------------------------------------------------
# The detached route: the worker, the launch transport, the terminal poll
# ---------------------------------------------------------------------------

def _stub_launch_prereqs(tmp_path, monkeypatch):
    uv = tmp_path / "fake-uv"
    uv.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    uv.chmod(0o755)
    monkeypatch.setenv(detach.UV_ENV, str(uv))
    monkeypatch.setattr(detach, "missing_provider_binaries", lambda *a, **k: {})
    monkeypatch.setattr(detach, "missing_runner_binaries", lambda *a, **k: {})


class TestTheDetachedRoute:
    def _launch_through_the_dispatcher(self, workdir, monkeypatch, capsys):
        """A real `detach.launch` behind `review-plan plans/foo.md --detach`,
        with the fork stubbed: returns (payload, the child's env, run id)."""
        plan = workdir / P
        phase_dir = output_dir_path(plan) / "review-plan"
        phase_dir.mkdir(parents=True, exist_ok=True)
        _stub_launch_prereqs(workdir, monkeypatch)
        spawned = {}

        def fake_spawn(command, *, cwd, env, log_path):
            spawned["env"] = dict(env)
            spawned["command"] = list(command)
            spawned["run_id"] = command[command.index("--internal-run-id") + 1]
            status_file.claim_run(str(phase_dir), {
                "run_id": spawned["run_id"], "state": "running",
                "pid": os.getpid(), "heartbeat_at": status_file.now_iso(),
            })
            return os.getpid()

        monkeypatch.setattr(process_control, "spawn_detached", fake_spawn)

        def launch(argv):
            record = cli_common.invocation()
            return detach.launch(
                mode="review-plan", argv=record["argv"], phase_dir=str(phase_dir),
                plan_file=str(plan), phase="review-plan",
                args=argparse.Namespace(force=False, if_not_running=False),
            )

        _target(monkeypatch, launch)
        code, out, err, result = _run(["review-plan", P, "--detach"], capsys)
        assert code == 0, err
        return result, out, err, spawned

    @pytest.mark.parametrize("spec", [
        S(),
        S(("review-plan",), files=RP_VALID),
    ], ids=["fresh", "already-on-record"])
    def test_the_launch_ships_the_typed_plan_and_answers_nothing(
        self, workdir, capsys, monkeypatch, spec
    ):
        build(workdir, spec)
        result, out, err, spawned = self._launch_through_the_dispatcher(
            workdir, monkeypatch, capsys
        )
        _assert_neither(out, err, result)
        assert spawned["env"][detach.PLAN_ARG_ENV] == P
        # The worker's own argv can only carry the absolute path.
        command = spawned["command"]
        assert command[command.index("--plan-file") + 1] == str(workdir / P)

    def test_the_whole_route_prints_the_plan_as_typed(self, workdir, capsys,
                                                      monkeypatch):
        """Launch with a relative plan, run the worker in-process from the
        shipped environment, then poll: the worker's block and the terminal
        poll both print the plan relative although each holds it absolute."""
        plan = build(workdir, S())
        _result, _out, _err, spawned = self._launch_through_the_dispatcher(
            workdir, monkeypatch, capsys
        )
        phase_dir = output_dir_path(plan) / "review-plan"
        (phase_dir / "validation.json").write_text(
            json.dumps(RP_VALID[("review-plan", "validation.json")]), encoding="utf-8"
        )
        run_id = spawned["run_id"]

        # -- the worker ----------------------------------------------------
        monkeypatch.setenv(detach.PLAN_ARG_ENV, spawned["env"][detach.PLAN_ARG_ENV])

        def worker(argv):
            plan_file = argv[argv.index("--plan-file") + 1]
            detach.begin_worker_run(
                phase_dir=str(phase_dir), mode="review-plan", plan_file=plan_file,
                args=argparse.Namespace(internal_worker=True,
                                        internal_run_id=run_id, host=None),
            )
            assert detach.PLAN_ARG_ENV not in os.environ  # popped
            state = StateManager(Path(plan_file))
            state.mark_phase_completed("review-plan")
            state.save()

        _target(monkeypatch, worker)
        worker_argv = [
            "review-plan", str(plan), "--yes", EXEC,
            "--internal-worker", "--internal-run-id", run_id,
        ]
        code, out, err, result = _run(worker_argv, capsys)
        assert code == 0, err
        expected = f"{PLUGIN}--apply-suggestions {P}"
        assert _block(out) == [f"{NEXT_COMMAND_MARKER} {expected}"]
        assert result["user_next_step"]["command"] == expected
        record = status_file.read_run(str(phase_dir))
        assert record["plan_arg"] == P
        assert record["plan_file"] == str(plan)
        assert record["state"] == "done"

        # -- the terminal poll (its own argv absolute) --------------------
        monkeypatch.setattr(multi_llm, "_load_target", REAL_LOAD_TARGET)
        code, out, err, result = _run(
            ["status", str(plan), "--view", "run", "--phase", "review-plan"], capsys
        )
        assert code == 0, err
        assert result["user_next_step"]["command"] == expected
        assert _block(out) == [f"{NEXT_COMMAND_MARKER} {expected}"]

    def test_a_worker_with_no_shipped_plan_falls_back_to_plan_file(
        self, workdir, capsys, monkeypatch
    ):
        plan = build(workdir, S(("review-plan",), files=RP_VALID))
        phase_dir = output_dir_path(plan) / "review-plan"

        def worker(argv):
            detach.begin_worker_run(
                phase_dir=str(phase_dir), mode="review-plan", plan_file=str(plan),
                args=argparse.Namespace(internal_worker=True,
                                        internal_run_id="run-noarg", host=None),
            )

        _target(monkeypatch, worker)
        code, out, err, result = _run([
            "review-plan", str(plan), "--yes", EXEC,
            "--internal-worker", "--internal-run-id", "run-noarg",
        ], capsys)
        assert code == 0, err
        assert "plan_arg" not in status_file.read_run(str(phase_dir))
        assert result["user_next_step"]["command"] == (
            f"{PLUGIN}--apply-suggestions {plan}"
        )

    def test_child_env_carries_this_launchs_value_or_none(self, monkeypatch):
        monkeypatch.setenv(detach.PLAN_ARG_ENV, "stale/inherited.md")
        assert detach.child_env(plan_arg="plans/x.md")[detach.PLAN_ARG_ENV] == (
            "plans/x.md"
        )
        assert detach.PLAN_ARG_ENV not in detach.child_env(plan_arg=None)

    def test_the_invoker_value_falls_back_to_the_launch_argv(self):
        cli_common.clear_invocation()
        assert detach._invoker_plan_arg(["--plan-file", "plans/a b.md"]) == "plans/a b.md"
        cli_common.record_invocation("review-plan", ["--plan-file", "typed.md"])
        try:
            assert detach._invoker_plan_arg(["--plan-file", "/abs/typed.md"]) == "typed.md"
        finally:
            cli_common.clear_invocation()


# ---------------------------------------------------------------------------
# run_status's terminal poll
# ---------------------------------------------------------------------------

def _poll(plan_arg, *flags):
    return ["status", plan_arg, "--phase", "review-plan", *flags]


@pytest.fixture
def finished_review(workdir):
    """review-plan completed and polled as a finished detached run."""
    plan = build(workdir, S(("review-plan",), files=RP_VALID))
    _terminal_record(plan, "review-plan", mode="review-plan",
                     plan_file=str(plan), plan_arg=P)
    return plan


EXPECTED_POLL = f"{PLUGIN}--apply-suggestions {P}"


class TestTheTerminalPoll:
    @pytest.mark.parametrize("worker_exit", [0, 2])
    def test_a_terminal_poll_carries_both(self, workdir, capsys, worker_exit):
        plan = build(workdir, S(("review-plan",), files=RP_VALID))
        _terminal_record(plan, "review-plan", mode="review-plan",
                         plan_file=str(plan), plan_arg=P, exit_code=worker_exit)
        code, out, err, result = _run(_poll(str(plan), "--view", "run"), capsys)
        assert code == 0, err
        assert result["mode"] == "review-plan"
        assert result["user_next_step"]["command"] == EXPECTED_POLL
        assert _block(out) == [f"{NEXT_COMMAND_MARKER} {EXPECTED_POLL}"]
        # After the rendered text, directly above the result line.
        lines = out.splitlines()
        assert lines[-2] == f"{NEXT_COMMAND_MARKER} {EXPECTED_POLL}"
        assert lines[-1].startswith(cli_common.RESULT_MARKER)

    def test_a_record_without_plan_arg_uses_its_absolute_plan_file(
        self, workdir, capsys
    ):
        plan = build(workdir, S(("review-plan",), files=RP_VALID))
        _terminal_record(plan, "review-plan", mode="review-plan", plan_file=str(plan))
        _code, out, _err, result = _run(_poll(P, "--view", "run"), capsys)
        assert result["user_next_step"]["command"] == (
            f"{PLUGIN}--apply-suggestions {plan}"
        )

    def test_a_live_poll_carries_neither(self, workdir, capsys):
        plan = build(workdir, S(("review-plan",), files=RP_VALID))
        _terminal_record(
            plan, "review-plan", mode="review-plan", plan_file=str(plan),
            plan_arg=P, state="running", exit_code=None, pid=os.getpid(),
            process_start_time=process_control.process_start_time(os.getpid()),
            heartbeat_at=status_file.now_iso(), heartbeat_interval_s=10.0,
        )
        code, out, err, result = _run(_poll(P, "--view", "run"), capsys)
        assert code == 2
        _assert_neither(out, err, result)

    @pytest.mark.parametrize("state,exit_code", [("failed", 3), ("aborted", None)])
    def test_a_failed_or_aborted_poll_carries_neither(self, workdir, capsys,
                                                      state, exit_code):
        plan = build(workdir, S(("review-plan",), files=RP_VALID))
        _terminal_record(plan, "review-plan", mode="review-plan",
                         plan_file=str(plan), plan_arg=P, state=state,
                         exit_code=exit_code)
        code, out, err, result = _run(_poll(P, "--view", "run"), capsys)
        assert code == 3
        _assert_neither(out, err, result)

    def test_a_terminal_ask_poll_carries_neither_and_reads_no_workflow(
        self, workdir, capsys, monkeypatch
    ):
        plan = build(workdir, S(("review-plan",), files=RP_VALID))
        _terminal_record(plan, "ask/why-0badc0de", mode="ask", plan_file=str(plan),
                         plan_arg=P)

        def unreadable(plan_path):
            raise AssertionError("an ask poll has no phase to answer for")

        monkeypatch.setattr(run_status, "_workflow_payload", unreadable)
        code, out, err, result = _run(
            ["status", P, "--view", "run", "--phase", "ask"], capsys
        )
        assert code == 0, err
        assert result["mode"] == "ask"
        _assert_neither(out, err, result)
        assert err == ""

    def test_an_unpinned_poll_carries_neither(self, finished_review, capsys):
        code, out, err, result = _run(["status", P, "--view", "run"], capsys)
        assert code == 0
        _assert_neither(out, err, result)

    def test_a_failed_workflow_read_costs_one_stderr_line(
        self, finished_review, capsys, monkeypatch
    ):
        _code, _out, _err, before = _run(_poll(P, "--view", "run"), capsys)

        def unreadable(plan_path):
            raise OSError("state.json unreadable")

        monkeypatch.setattr(run_status, "_workflow_payload", unreadable)
        code, out, err, result = _run(_poll(P, "--view", "run"), capsys)
        assert code == 0
        _assert_neither(out, err, result)
        assert err.splitlines() == [
            f"{LOOKUP_FAILED_PREFIX} (OSError: state.json unreadable); "
            "no next command printed"
        ]
        # Every other key as it was; F7's notice goes with the same read.
        for payload in (before, result):
            payload.pop("user_next_step", None)
            payload.pop("notice", None)
        assert result == before

    def test_a_builder_raise_on_the_poll_leaves_the_exit_code(
        self, finished_review, capsys, monkeypatch
    ):
        def boom(*args, **kwargs):
            raise ValueError("no template")

        monkeypatch.setattr(next_phase, "next_phase_commands", boom)
        code, out, err, result = _run(_poll(P, "--view", "run"), capsys)
        assert code == 0
        _assert_neither(out, err, result)
        assert len(err.splitlines()) == 1 and err.startswith(LOOKUP_FAILED_PREFIX)

    def test_the_workflow_is_read_after_the_wait(self, workdir, capsys, monkeypatch):
        """A run that goes terminal inside `--wait` answers from the state it
        left, not from the read `--view all` took before waiting."""
        plan = build(workdir, S())
        phase_dir = _terminal_record(
            plan, "review-plan", mode="review-plan", plan_file=str(plan),
            plan_arg=P, state="running", exit_code=None, pid=os.getpid(),
            process_start_time=process_control.process_start_time(os.getpid()),
            heartbeat_at=status_file.now_iso(), heartbeat_interval_s=10.0,
        )
        (phase_dir / "validation.json").write_text(
            json.dumps(RP_VALID[("review-plan", "validation.json")]), encoding="utf-8"
        )

        def finish_during_wait(plan_path, args, views):
            state = StateManager(plan)
            state.mark_phase_completed("review-plan")
            state.save()
            status_file.update_run(
                str(phase_dir), {"state": "done", "exit_code": 0},
                expect_run_id="run-t5-0001",
            )
            return run_status._collect(plan_path, args)

        monkeypatch.setattr(run_status, "_wait_for_terminal", finish_during_wait)
        monkeypatch.setattr(run_status, "_resolve_wait_timeout", lambda *a: 1.0)
        code, out, err, result = _run(_poll(P, "--wait"), capsys)
        assert code == 0, err
        assert result["user_next_step"]["command"] == EXPECTED_POLL


class TestTheViewsAndTheJsonPayload:
    @pytest.mark.parametrize("view", ["run", "all"])
    def test_text_prints_the_block_on_stdout(self, finished_review, capsys, view):
        code, out, err, result = _run(_poll(P, "--view", view), capsys)
        assert code == 0, err
        assert result["user_next_step"]["command"] == EXPECTED_POLL
        assert _block(out) == [f"{NEXT_COMMAND_MARKER} {EXPECTED_POLL}"]
        assert _block(err) == []

    @pytest.mark.parametrize("view", ["run", "all"])
    def test_json_keeps_stdout_one_document_and_the_payload_unchanged(
        self, finished_review, capsys, monkeypatch, view
    ):
        code, out, err, result = _run(_poll(P, "--view", view, "--json"), capsys)
        assert code == 0, err
        payload = json.loads(out.rsplit(cli_common.RESULT_MARKER, 1)[0])
        assert _block(out) == []
        assert _block(err) == [f"{NEXT_COMMAND_MARKER} {EXPECTED_POLL}"]
        assert result["user_next_step"]["command"] == EXPECTED_POLL
        assert "user_next_step" not in json.dumps(payload)

        # Today's payload: the same poll with the emitter switched off.
        monkeypatch.setattr(run_status, "_emit_user_next_step", lambda *a, **k: None)
        _code, today_out, _err, today = _run(
            _poll(P, "--view", view, "--json"), capsys
        )
        assert json.loads(today_out.rsplit(cli_common.RESULT_MARKER, 1)[0]) == payload
        assert "user_next_step" not in today

    def test_view_workflow_carries_neither_and_matches_the_retired_mode(
        self, finished_review, capsys
    ):
        plan = str(finished_review)
        code, out, err, result = _run(_poll(plan, "--view", "workflow"), capsys)
        assert code == 0
        _assert_neither(out, err, result)
        through_status = out.rsplit(cli_common.RESULT_MARKER, 1)[0]
        with pytest.raises(SystemExit):
            prereq.main(["--plan-file", plan, "--status"])
        assert through_status == capsys.readouterr().out


# ---------------------------------------------------------------------------
# util next-command
# ---------------------------------------------------------------------------

class TestUtilNextCommand:
    def test_before_and_after_the_completion_write(self, workdir, capsys):
        """An apply-fix batch on record and the phase not yet marked: the report
        step's lookup answers `--resume`; after `--mark-completed`, complete."""
        build(workdir, S(UP_TO_CODE_REVIEW, tasks={"apply-fix-batch-0": "completed"},
                          files=CR_FINDING))
        lookup = ["util", "next-command", P, "--after", "apply-code-fixes"]
        code, out, err, result = _run(lookup, capsys)
        assert code == 0, err
        resume = f"{PLUGIN}--apply-code-fixes {P} --resume"
        assert _block(out) == [f"{NEXT_COMMAND_MARKER} {resume}"]
        assert result["user_next_step"]["command"] == resume
        assert result["mode"] == "util next-command"

        # The native leg's completion write (it carries no key itself).
        code, out, err, result = _run(
            ["apply-code-fixes", P, "--mark-completed"], capsys
        )
        assert code == 0, err
        _assert_neither(out, err, result)

        code, out, err, result = _run(lookup, capsys)
        assert code == 0, err
        assert _block(out) == [WORKFLOW_COMPLETE_MARKER]
        assert result["user_next_step"] == {
            "phase": None, "command": None, "complete": True,
        }

    def test_a_builder_raise_is_exit_3_with_one_line_and_no_block(
        self, workdir, capsys, monkeypatch
    ):
        build(workdir, S(UP_TO_TASKS))

        def boom(*args, **kwargs):
            raise RuntimeError("state.json is corrupt")

        monkeypatch.setattr(next_phase, "next_phase_commands", boom)
        code, out, err, result = _run(
            ["util", "next-command", P, "--after", "generate-tasks"], capsys
        )
        assert code == 3
        _assert_neither(out, err, result)
        assert err.splitlines() == [
            f"{LOOKUP_FAILED_PREFIX} (RuntimeError: state.json is corrupt); "
            "no next command printed"
        ]

    def test_an_unknown_after_word_is_exit_3_naming_the_modes(self, workdir, capsys):
        build(workdir, S(UP_TO_TASKS))
        code, out, err, result = _run(
            ["util", "next-command", P, "--after", "reveiw-plan"], capsys
        )
        assert code == 3
        _assert_neither(out, err, result)
        assert len(err.splitlines()) == 1
        assert "reveiw-plan" in err and "review-plan" in err and "full" in err

    def test_a_missing_plan_is_exit_3(self, workdir, capsys):
        code, out, err, result = _run(["util", "next-command", "plans/nope.md"], capsys)
        assert code == 3
        _assert_neither(out, err, result)
        assert len(err.splitlines()) == 1

    @pytest.mark.parametrize("word", ["ask", "status", "init"])
    def test_no_answer_prints_nothing_and_exits_0(self, workdir, capsys, word):
        build(workdir, S(UP_TO_TASKS))
        code, out, err, result = _run(
            ["util", "next-command", P, "--after", word], capsys
        )
        assert code == 0
        _assert_neither(out, err, result)
        assert out.splitlines()[:-1] == []

    @pytest.mark.parametrize("word,expected", [
        ("full", f"--implement {P}"),
        ("tasks", f"--implement {P}"),          # an alias
        ("generate-tasks", f"--implement {P}"),
    ])
    def test_any_mode_word_or_alias(self, workdir, capsys, word, expected):
        build(workdir, S(UP_TO_TASKS))
        code, out, err, result = _run(
            ["util", "next-command", P, "--after", word], capsys
        )
        assert code == 0, err
        assert result["user_next_step"]["command"] == PLUGIN + expected

    def test_no_after_is_plain_suggested_next(self, workdir, capsys):
        build(workdir, S(("generate-tasks",)))
        _code, out, _err, result = _run(["util", "next-command", P], capsys)
        assert result["user_next_step"]["command"] == f"{PLUGIN}--review-plan {P}"

    @pytest.mark.parametrize("tail", [
        ["--after", "generate-tasks", "--host", "codex"],
        ["--host", "codex", "--after", "generate-tasks"],
    ])
    def test_host_is_accepted_in_any_position_after_the_plan(
        self, workdir, capsys, tail
    ):
        build(workdir, S(UP_TO_TASKS))
        code, out, err, result = _run(["util", "next-command", P, *tail], capsys)
        assert code == 0, err
        assert result["user_next_step"]["command"] == (
            f"Use the multi-llm skill with exactly these arguments: --implement {P}"
        )
        assert _block(out)[0] == f"{NEXT_COMMAND_MARKER} {result['user_next_step']['command']}"

    def test_the_relative_plan_is_printed_as_received(self, workdir, capsys):
        plan = build(workdir, S(UP_TO_TASKS))
        _code, _out, _err, rel = _run(
            ["util", "next-command", P, "--after", "generate-tasks"], capsys
        )
        _code, _out, _err, absolute = _run(
            ["util", "next-command", str(plan), "--after", "generate-tasks"], capsys
        )
        assert rel["user_next_step"]["command"] == f"{PLUGIN}--implement {P}"
        assert absolute["user_next_step"]["command"] == f"{PLUGIN}--implement {plan}"

    def test_it_writes_nothing(self, workdir, capsys):
        plan = build(workdir, S(UP_TO_TASKS))
        state_file = output_dir_path(plan) / "state.json"
        before = state_file.read_bytes()
        listing = sorted(p.name for p in output_dir_path(plan).iterdir())
        _run(["util", "next-command", P, "--after", "generate-tasks"], capsys)
        assert state_file.read_bytes() == before
        assert sorted(p.name for p in output_dir_path(plan).iterdir()) == listing
