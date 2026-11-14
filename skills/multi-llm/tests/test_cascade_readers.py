"""``--cascade``'s other readers: ``util next-command``, ``status`` and the help surfaces.

Between two stages of a cascade every step of the composite is on record, so
the plain rules (``improve_workflow.skip_reason``) read "everything complete":
``--blocked`` would print ``[ALREADY_COMPLETE]`` and ``--after`` would name the
next workflow phase. While the pointer's cascade is unfinished, every reader
that answers for the composite names the composite's own continuation with
``--cascade`` instead; once it is finished, or with none, the answers are the
ones they always were. ``status`` shows the stage of a composite run's
unfinished cascade, in the text and in ``--json``. The pointer and the record
are written here as ``utils/improve_cascade.py`` writes them, and read back
through its own ``load_current`` so every fixture passes the schema.

The contract: docs/development/workflows.md, "A cascade (``--cascade``)".

Usage:
    uv run --project skills/multi-llm -- pytest skills/multi-llm/tests/test_cascade_readers.py -q
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import modes  # noqa: E402
import multi_llm  # noqa: E402
from utils import cli_common, status_file, status_text, status_views  # noqa: E402
from utils import improve_cascade as ic  # noqa: E402
from utils import improve_workflow as iw  # noqa: E402
from utils import next_phase as np_  # noqa: E402
from utils.cascade_provenance import HIDDEN_FLAG  # noqa: E402
from utils.output_handler import output_dir_path  # noqa: E402
from tests.support.next_phase_world import (  # noqa: E402
    CR_FINDING,
    P,
    RP_VALID,
    S,
    UP_TO_CODE_REVIEW,
    _build,
)
from tests.test_result_contract import RESULT_LINE, STATUS_JSON, assert_valid  # noqa: E402

CID = "c0ffee00" + "1" * 24
OTHER_CID = "deadbeef" + "2" * 24
MODELS = ("codex:gpt-a", "codex:gpt-b", "codex:gpt-c")
PLUGIN = "/multi-llm:multi-llm"

#: Each composite with the state its two steps leave on record (between two
#: stages, and after a finished cascade's last stage).
DONE = {
    "improve-plan": S(("review-plan", "apply-suggestions"), files=RP_VALID),
    "improve-code": S(UP_TO_CODE_REVIEW + ("apply-code-fixes",), files=CR_FINDING),
}
#: A stop inside a stage: the review completed, the apply part-way.
APPLY_PART_WAY = {
    "improve-plan": S(("review-plan",), tasks={"apply-batch-0": "completed"}, files=RP_VALID),
    "improve-code": S(UP_TO_CODE_REVIEW, tasks={"apply-fix-batch-0": "completed"},
                      files=CR_FINDING),
}
COMPOSITES = list(DONE)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _stage(index: int, model: str, state: str) -> Dict[str, Any]:
    closed = state in ("done", "skipped")
    return {"index": index, "model": model, "state": state, "bound_round": None,
            "scope_policy": None,
            "started_at": "2026-11-13T12:00:00" if state != "pending" else None,
            "finished_at": "2026-11-13T12:10:00" if closed else None,
            "skip_reason": "rate limited" if state == "skipped" else None,
            "counts": None}


def _cascade(plan: Path, mode: str, *, state: str = "running", closed: int = 1,
             cascade_id: str = CID, models=MODELS) -> Path:
    """The pointer and a record whose first *closed* stages are done.

    ``state`` is the record's (``running``, ``finished``, ``abandoned``); a
    finished cascade has every stage done. Returns the cascade's directory.
    """
    if state == "finished":
        closed = len(models)
    composite = ic.composite_dir(plan, mode)
    rel = f"{ic.CASCADES_DIR}/20261113T120000-{cascade_id[:8]}"
    directory = composite / rel
    directory.mkdir(parents=True, exist_ok=True)
    stages = [
        _stage(index, model, "done" if index <= closed else (
            "running" if index == closed + 1 and state == "running" else "pending"))
        for index, model in enumerate(models, start=1)
    ]
    record = {
        "schema_version": 1, "cascade_id": cascade_id, "mode": mode, "state": state,
        "started_at": "2026-11-13T12:00:00", "scope_policy": None,
        "selectors": {"files_file": None, "base_ref": None}, "stages": stages,
        "finished_at": "2026-11-13T13:00:00" if state != "running" else None,
    }
    (directory / ic.RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")
    pointer = {"schema_version": 1, "cascade_id": cascade_id, "dir": rel}
    (composite / ic.POINTER_NAME).write_text(json.dumps(pointer), encoding="utf-8")
    current = ic.load_current(plan, mode)
    assert current is not None and current.state == state
    return directory


def _composite_run(plan: Path, composite: str, **run: Any) -> None:
    """The composite's own ``run`` record in its run dir, as a detached worker leaves it."""
    target = output_dir_path(plan) / composite
    target.mkdir(parents=True, exist_ok=True)
    status_file.write_status(str(target), {"phase": composite})
    record = {"run_id": "run-improve-0001", "state": "failed", "exit_code": 3,
              "detached": True, "mode": composite}
    record.update(run)
    status_file.claim_run(str(target), record)


def _build_outside(root: Path, spec) -> Path:
    """``_build`` from the repository: a plan's first ``StateManager`` records
    ``git rev-parse HEAD`` from the cwd."""
    here = os.getcwd()
    os.chdir(Path(__file__).parent.parent)
    try:
        return _build(root, spec)
    finally:
        os.chdir(here)


@pytest.fixture
def here(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("MULTI_LLM_HOST", modes.ROUTE_ENV):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _blocked(argv: List[str], monkeypatch, capsys):
    record: Dict[str, Any] = {}
    monkeypatch.setattr(cli_common, "record_user_resume",
                        lambda obj: record.setdefault("v", obj))
    code = np_.main(argv)
    return code, capsys.readouterr(), record.get("v")


def _after(argv: List[str], monkeypatch, capsys):
    record: Dict[str, Any] = {}
    monkeypatch.setattr(cli_common, "record_user_next_step",
                        lambda obj: record.setdefault("v", obj))
    code = np_.main(argv)
    return code, capsys.readouterr(), record.get("v")


def _why(stage: int = 2, model: str = MODELS[1]) -> str:
    return f"continues the unfinished cascade at stage {stage} of 3 ({model})"


# ---------------------------------------------------------------------------
# The shared reader
# ---------------------------------------------------------------------------

class TestUnfinishedCascade:
    @pytest.mark.parametrize("mode", COMPOSITES)
    def test_an_unfinished_cascade_is_its_status(self, here, mode):
        plan = _build_outside(here, DONE[mode])
        directory = _cascade(plan, mode, closed=1)
        assert np_.unfinished_cascade(mode, plan) == {
            "cascade_id": CID, "state": "running", "stage": 2, "of": 3,
            "model": MODELS[1], "dir": str(directory),
        }

    @pytest.mark.parametrize("state", ["finished", "abandoned"])
    def test_a_finished_or_abandoned_cascade_is_none(self, here, state):
        plan = _build_outside(here, DONE["improve-plan"])
        _cascade(plan, "improve-plan", state=state)
        assert np_.unfinished_cascade("improve-plan", plan) is None

    def test_no_pointer_and_no_composite_are_none_and_write_nothing(self, here):
        plan = _build_outside(here, DONE["improve-plan"])
        before = sorted(p.relative_to(here) for p in here.rglob("*"))
        assert np_.unfinished_cascade("improve-plan", plan) is None
        assert np_.unfinished_cascade("apply-suggestions", plan) is None
        assert np_.unfinished_cascade("review", plan) is None
        assert sorted(p.relative_to(here) for p in here.rglob("*")) == before

    def test_improve_commits_never_reads_a_cascade(self, here):
        plan = _build_outside(here, DONE["improve-code"])
        commits = output_dir_path(plan) / "improve-commits"
        commits.mkdir(parents=True)
        (commits / ic.POINTER_NAME).write_text("not json", encoding="utf-8")
        assert np_.unfinished_cascade("improve-commits", plan) is None

    def test_an_unreadable_record_raises_naming_the_file(self, here):
        plan = _build_outside(here, DONE["improve-plan"])
        directory = _cascade(plan, "improve-plan")
        (directory / ic.RECORD_NAME).write_text("{", encoding="utf-8")
        with pytest.raises(ic.CascadeRecordError) as raised:
            np_.unfinished_cascade("improve-plan", plan)
        assert raised.value.path == str(directory / ic.RECORD_NAME)


# ---------------------------------------------------------------------------
# util next-command --blocked <composite>
# ---------------------------------------------------------------------------

class TestBlocked:
    @pytest.mark.parametrize("mode", COMPOSITES)
    def test_between_two_stages_it_continues_the_composite(self, here, monkeypatch,
                                                           capsys, mode):
        """Every step on record, the cascade at stage 2: never
        `[ALREADY_COMPLETE]`, the composite's continuation with `--cascade`."""
        plan = _build_outside(here, DONE[mode])
        _cascade(plan, mode, closed=1)
        code, out, key = _blocked([P, "--blocked", mode, "--host", "claude-code"],
                                  monkeypatch, capsys)
        assert code == 0, out.err
        command = f"{PLUGIN} --{mode} {P} --cascade --resume"
        assert out.out == f"{np_.RESUME_COMMAND_MARKER} {command}\n"
        assert "[ALREADY_COMPLETE]" not in out.out
        assert key == {
            "phase": mode, "mode": mode, "command": command,
            "flags": ["--cascade", "--resume"], "why": _why(),
            "markdown": np_.render_resume_markdown(mode, command, _why(), []),
        }

    @pytest.mark.parametrize("mode", COMPOSITES)
    def test_a_stop_inside_a_stage_continues_the_composite_not_the_apply(
        self, here, monkeypatch, capsys, mode
    ):
        """An apply part-way through stage 2: its own resume would finish the
        apply and leave the cascade where it is; the composite goes on."""
        plan = _build_outside(here, APPLY_PART_WAY[mode])
        _cascade(plan, mode, closed=1)
        code, out, key = _blocked([P, "--blocked", mode], monkeypatch, capsys)
        assert code == 0, out.err
        assert out.out == (f"{np_.RESUME_COMMAND_MARKER} "
                           f"{PLUGIN} --{mode} {P} --cascade --resume\n")
        assert key["mode"] == mode and key["why"] == _why()

    def test_the_run_records_selectors_ride_along_without_force(self, here, monkeypatch,
                                                                 capsys):
        from utils.cli_common import replay_flags

        plan = _build_outside(here, DONE["improve-code"])
        _cascade(plan, "improve-code", closed=2)
        launched = ["--plan-file", str(plan), "--host", "codex", "--detach", "--force",
                    "--cascade", "--models", *MODELS, "--base-ref", "origin/release"]
        _composite_run(plan, "improve-code", host="codex", route="package",
                       replay_flags=replay_flags("improve-code", launched))
        code, out, key = _blocked([P, "--blocked", "improve-code", "--host", "codex"],
                                  monkeypatch, capsys)
        assert code == 0, out.err
        flags = ["--models", *MODELS, "--base-ref", "origin/release", "--cascade", "--resume"]
        assert key["flags"] == flags
        assert out.out == (f"{np_.RESUME_COMMAND_MARKER} $multi-llm:multi-llm "
                           f"--improve-code {P} {' '.join(flags)}\n")
        assert key["why"] == _why(3, MODELS[2])

    @pytest.mark.parametrize("state", ["finished", "abandoned", None])
    @pytest.mark.parametrize("mode", COMPOSITES)
    def test_with_no_unfinished_cascade_the_answer_is_todays(self, here, monkeypatch,
                                                             capsys, mode, state):
        """A finished, abandoned or absent cascade: `[ALREADY_COMPLETE]`, exit 2,
        byte for byte the answer with no cascade files at all."""
        plan = _build_outside(here, DONE[mode])
        if state is not None:
            _cascade(plan, mode, state=state)
        got = _blocked([P, "--blocked", mode], monkeypatch, capsys)
        shutil.rmtree(ic.composite_dir(plan, mode), ignore_errors=True)
        want = _blocked([P, "--blocked", mode], monkeypatch, capsys)
        assert got[0] == want[0] == 2
        assert (got[1].out, got[1].err, got[2]) == (want[1].out, want[1].err, want[2])
        assert got[1].out.startswith("[ALREADY_COMPLETE] ")

    def test_a_finished_cascade_leaves_an_inner_resume_as_it_was(self, here, monkeypatch,
                                                                  capsys):
        plan = _build_outside(here, APPLY_PART_WAY["improve-plan"])
        _cascade(plan, "improve-plan", state="finished")
        code, out, key = _blocked([P, "--blocked", "improve-plan"], monkeypatch, capsys)
        assert code == 0, out.err
        assert out.out == (f"{np_.RESUME_COMMAND_MARKER} "
                           f"{PLUGIN} --apply-suggestions {P} --resume\n")
        assert "why" not in key or "cascade" not in key["why"]

    def test_an_unreadable_record_is_a_failed_lookup(self, here, monkeypatch, capsys):
        plan = _build_outside(here, DONE["improve-plan"])
        directory = _cascade(plan, "improve-plan")
        (directory / ic.RECORD_NAME).write_text("[]", encoding="utf-8")
        code, out, key = _blocked([P, "--blocked", "improve-plan"], monkeypatch, capsys)
        assert (code, out.out, key) == (3, "", None)
        assert len(out.err.splitlines()) == 1
        assert out.err.startswith(np_.LOOKUP_FAILED_PREFIX)
        assert "record.json" in out.err


# ---------------------------------------------------------------------------
# util next-command --after <composite>, and the builder under the gate
# ---------------------------------------------------------------------------

class TestAfter:
    @pytest.mark.parametrize("mode", COMPOSITES)
    def test_between_two_stages_it_continues_the_composite(self, here, monkeypatch,
                                                           capsys, mode):
        plan = _build_outside(here, DONE[mode])
        _cascade(plan, mode, closed=1)
        code, out, key = _after([P, "--after", mode, "--host", "claude-code"],
                                monkeypatch, capsys)
        assert code == 0, out.err
        command = f"{PLUGIN} --{mode} {P} --cascade --resume"
        lines = out.out.splitlines()
        assert lines[:2] == [f"{np_.NEXT_COMMAND_WHY_MARKER} {_why()}",
                             f"{np_.NEXT_COMMAND_MARKER} {command}"]
        assert (key["phase"], key["command"], key["complete"], key["why"]) == (
            mode, command, False, _why())
        assert "--generate-tasks" not in out.out

    @pytest.mark.parametrize("mode", COMPOSITES)
    def test_the_gated_builder_answers_the_same(self, here, mode):
        """A composite's own exit (`finished_mode`) reads the cascade first too."""
        plan = _build_outside(here, DONE[mode])
        _cascade(plan, mode, closed=2)
        answer = np_.next_phase_commands(P, "opencode", None, mode, plan_path=plan)
        assert answer is not None and not answer.complete
        assert answer.command == f"/multi-llm --{mode} {P} --cascade --resume"
        assert answer.why == _why(3, MODELS[2])

    @pytest.mark.parametrize("state", ["finished", "abandoned", None])
    @pytest.mark.parametrize("mode", COMPOSITES)
    def test_with_no_unfinished_cascade_the_answer_is_todays(self, here, monkeypatch,
                                                             capsys, mode, state):
        plan = _build_outside(here, DONE[mode])
        if state is not None:
            _cascade(plan, mode, state=state)
        got = _after([P, "--after", mode], monkeypatch, capsys)
        shutil.rmtree(ic.composite_dir(plan, mode), ignore_errors=True)
        want = _after([P, "--after", mode], monkeypatch, capsys)
        assert got[0] == want[0] == 0
        assert (got[1].out, got[2]) == (want[1].out, want[2])
        assert "--cascade" not in got[1].out

    def test_an_inner_phase_word_is_not_the_composite(self, here):
        """`--after apply-suggestions` walks on as today: only the composite's
        word continues a cascade."""
        plan = _build_outside(here, DONE["improve-plan"])
        _cascade(plan, "improve-plan")
        answer = np_.next_phase_commands(P, "claude-code", None, None,
                                         after_mode="apply-suggestions", plan_path=plan)
        assert answer.command == f"{PLUGIN} --generate-tasks {P}"


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def _status(argv: List[str], capsys):
    capsys.readouterr()
    code = multi_llm.main(["status", *argv])
    out = capsys.readouterr().out
    payload = cli_common.parse_result_line(out)
    assert payload is not None
    return code, out.rsplit(cli_common.RESULT_MARKER, 1)[0], payload


class TestStatus:
    LINE = f"      cascade:        stage 2 of 3 ({MODELS[1]})"

    def _stopped(self, here, state="running"):
        plan = _build_outside(here, DONE["improve-plan"])
        _cascade(plan, "improve-plan", state=state, closed=1)
        _composite_run(plan, "improve-plan",
                       replay_flags=["--models", *MODELS, "--cascade"])
        return plan

    def test_a_stopped_composite_run_shows_its_stage_and_resumes_the_cascade(
        self, here, capsys
    ):
        plan = self._stopped(here)
        code, text, _payload = _status([str(plan), "--view", "run"], capsys)
        assert code == cli_common.EXIT_NEEDS_HUMAN
        lines = text.splitlines()
        head = next(i for i, line in enumerate(lines) if line.startswith("  improve-plan"))
        assert lines[head + 3] == self.LINE, lines[head:head + 5]
        resume = next(line for line in lines if "resume with:" in line)
        assert resume.rstrip().endswith(
            f"--models {' '.join(MODELS)} --cascade --resume"), resume
        assert HIDDEN_FLAG not in text

    def test_a_cascade_run_record_without_the_flag_still_resumes_the_cascade(
        self, here, capsys
    ):
        """A continuation typed bare (an unfinished cascade continues with
        nothing typed) leaves no `--cascade` in the record; the line adds it."""
        plan = _build_outside(here, DONE["improve-plan"])
        _cascade(plan, "improve-plan", closed=1)
        _composite_run(plan, "improve-plan", replay_flags=[])
        _code, text, _payload = _status([str(plan), "--view", "run"], capsys)
        resume = next(line for line in text.splitlines() if "resume with:" in line)
        assert resume.rstrip().endswith(" --cascade --resume"), resume

    def test_the_json_view_carries_the_cascade_and_matches_the_schema(self, here, capsys):
        plan = self._stopped(here)
        _code, text, payload = _status([str(plan), "--json"], capsys)
        document = json.loads(text)
        assert_valid(STATUS_JSON, document)
        assert_valid(RESULT_LINE, payload)
        (view,) = [v for v in document["runs"] if v.get("phase") == "improve-plan"]
        assert view["cascade"] == {
            "cascade_id": CID, "state": "running", "stage": 2, "of": 3,
            "model": MODELS[1], "dir": str(ic.load_current(plan, "improve-plan").cascade_dir),
        }

    def test_a_pinned_poll_relays_the_stage_first(self, here, capsys):
        plan = self._stopped(here)
        _code, _text, payload = _status(
            [str(plan), "--view", "run", "--run-id", "run-improve"], capsys)
        assert payload["progress_lines"][0] == f"cascade: stage 2 of 3 ({MODELS[1]})"

    @pytest.mark.parametrize("state", ["finished", "abandoned"])
    def test_a_finished_or_abandoned_cascade_shows_nothing(self, here, capsys, state):
        plan = self._stopped(here, state=state)
        _code, text, _payload = _status([str(plan), "--view", "run"], capsys)
        assert "cascade:" not in text
        resume = next(line for line in text.splitlines() if "resume with:" in line)
        assert resume.rstrip().endswith(f"--models {' '.join(MODELS)} --cascade --resume")
        _code, text, _payload = _status([str(plan), "--json"], capsys)
        assert all("cascade" not in view for view in json.loads(text)["runs"])

    def test_a_finished_cascade_with_a_plain_record_resumes_as_today(self, here, capsys):
        plan = _build_outside(here, DONE["improve-plan"])
        _cascade(plan, "improve-plan", state="finished")
        _composite_run(plan, "improve-plan", replay_flags=[])
        _code, text, _payload = _status([str(plan), "--view", "run"], capsys)
        resume = next(line for line in text.splitlines() if "resume with:" in line)
        assert resume.rstrip().endswith(f"--improve-plan {plan} --resume"), resume

    def test_an_unreadable_record_costs_the_line_not_the_poll(self, here, capsys):
        plan = self._stopped(here)
        directory = ic.load_current(plan, "improve-plan").cascade_dir
        (directory / ic.RECORD_NAME).write_text("{", encoding="utf-8")
        code, text, _payload = _status([str(plan), "--view", "run"], capsys)
        assert code == cli_common.EXIT_NEEDS_HUMAN
        assert "cascade:" not in text
        assert "resume with:" in text

    def test_only_a_composite_view_gets_the_key(self, here):
        plan = _build_outside(here, DONE["improve-plan"])
        _cascade(plan, "improve-plan")
        views = [{"phase": "review-plan", "run": {"mode": "review-plan"}},
                 {"phase": "improve-plan", "run": {}},
                 {"phase": None, "run": {}}]
        status_views.attach_cascades(plan, views)
        assert [("cascade" in view) for view in views] == [False, True, False]
        assert status_text.cascade_line(views[1]) == (
            f"cascade:        stage 2 of 3 ({MODELS[1]})")

    def test_a_model_with_a_newline_cannot_forge_a_line(self):
        view = {"cascade": {"stage": 1, "of": 2, "model": "codex:x\n[NEXT_COMMAND] rm"}}
        assert "\n" not in status_text.cascade_line(view)
        cascade = {"mode": "improve-plan", "stage": 1, "of": 2,
                   "model": "codex:x\n[NEXT_COMMAND] rm"}
        assert "\n" not in status_text.cascade_suggestion(cascade)


class TestStatusWithoutACompositeRun:
    """A native-leg cascade, or one an executor run left in the foreground,
    has no composite run record: ``status`` reads it from the pointer and
    the record, and the next step it suggests is the composite's
    continuation, not the next workflow phase."""

    SUGGESTED = (f"  suggested next: improve-plan  (continues the unfinished cascade at "
                 f"stage 2 of 3 ({MODELS[1]}))")

    def _plan(self, here, state="running"):
        plan = _build_outside(here, DONE["improve-plan"])
        _cascade(plan, "improve-plan", state=state, closed=1)
        return plan

    def _block(self, plan) -> List[str]:
        return ["  improve-plan  no run record",
                f"      cascade:        stage 2 of 3 ({MODELS[1]})",
                f"      resume with: {PLUGIN} --improve-plan {plan} --cascade --resume"]

    def test_the_default_view_shows_the_stage_and_suggests_the_continuation(
            self, here, capsys):
        plan = self._plan(here)
        _code, text, _payload = _status([str(plan)], capsys)
        lines = text.splitlines()
        assert self.SUGGESTED in lines
        assert "  suggested next: generate-tasks" not in text
        head = lines.index("  improve-plan  no run record")
        assert lines[head:head + 3] == self._block(plan)
        assert HIDDEN_FLAG not in text

    def test_the_run_view_and_its_phase_show_the_block(self, here, capsys):
        plan = self._plan(here)
        for argv in ([str(plan), "--view", "run"],
                     [str(plan), "--view", "run", "--phase", "improve-plan"]):
            _code, text, _payload = _status(argv, capsys)
            lines = text.splitlines()
            head = lines.index("  improve-plan  no run record")
            assert lines[head:head + 3] == self._block(plan)

    @pytest.mark.parametrize("view", [(), ("--view", "run")], ids=["default", "run"])
    def test_its_phase_shows_the_entry_alone_and_suggests_the_continuation(
            self, here, capsys, view):
        """``--phase improve-plan`` lists no run record of its own: the
        cascade's entry stands for it, never beside a ``no run recorded``
        line saying the same."""
        plan = self._plan(here)
        _code, text, _payload = _status([str(plan), *view, "--phase", "improve-plan"], capsys)
        lines = text.splitlines()
        assert "no run recorded" not in text
        runs = lines.index("runs ({phase}/.status.json) — is a launched run still alive:")
        assert lines[runs + 1:runs + 4] == self._block(plan)
        if not view:
            assert self.SUGGESTED in lines
        # Without the cascade, the phase's lines are today's.
        shutil.rmtree(ic.composite_dir(plan, "improve-plan"))
        _code, text, _payload = _status([str(plan), *view, "--phase", "improve-plan"], capsys)
        assert "  no run recorded for phase improve-plan" in text.splitlines()

    @pytest.mark.parametrize("phase", ["review-plan", "apply-suggestions", "improve-code"])
    def test_another_phase_suggests_neither_the_cascade_nor_the_next_phase(
            self, here, capsys, phase):
        """The cascade's continuation is not another phase's to suggest, and
        the workflow's own suggestion reads every step done between two
        stages, so under ``--phase`` naming another phase no suggestion line
        prints while the cascade is unfinished; without one it is today's."""
        plan = self._plan(here)
        _code, text, _payload = _status([str(plan), "--phase", phase], capsys)
        assert "suggested next:" not in text
        shutil.rmtree(ic.composite_dir(plan, "improve-plan"))
        _code, text, _payload = _status([str(plan), "--phase", phase], capsys)
        assert "  suggested next: generate-tasks" in text.splitlines()

    def test_another_phase_or_a_run_id_shows_no_block(self, here, capsys):
        plan = self._plan(here)
        for argv in ([str(plan), "--view", "run", "--phase", "review-plan"],
                     [str(plan), "--view", "run", "--run-id", "nothing"]):
            _code, text, _payload = _status(argv, capsys)
            assert "improve-plan  no run record" not in text and "cascade:" not in text

    def test_the_json_view_carries_it_and_matches_the_schema(self, here, capsys):
        plan = self._plan(here)
        _code, text, payload = _status([str(plan), "--json"], capsys)
        document = json.loads(text)
        assert_valid(STATUS_JSON, document)
        assert_valid(RESULT_LINE, payload)
        assert document["cascades"] == [{
            "mode": "improve-plan", "cascade_id": CID, "state": "running", "stage": 2,
            "of": 3, "model": MODELS[1],
            "dir": str(ic.load_current(plan, "improve-plan").cascade_dir)}]
        # The workflow payload is state.json's view, unchanged.
        assert document["workflow"]["suggested_next"] == "generate-tasks"

    def test_the_workflow_view_keeps_its_bytes(self, here, capsys):
        plan = self._plan(here)
        capsys.readouterr()
        assert multi_llm.main(["status", str(plan), "--view", "workflow"]) == 0
        with_cascade = capsys.readouterr().out
        shutil.rmtree(ic.composite_dir(plan, "improve-plan"))
        assert multi_llm.main(["status", str(plan), "--view", "workflow"]) == 0
        assert capsys.readouterr().out == with_cascade
        assert "cascade" not in with_cascade

    @pytest.mark.parametrize("state", ["finished", "abandoned"])
    def test_a_finished_or_abandoned_cascade_reads_as_none(self, here, capsys, state):
        plan = self._plan(here, state=state)
        _code, text, _payload = _status([str(plan)], capsys)
        with_record = text
        shutil.rmtree(ic.composite_dir(plan, "improve-plan"))
        _code, text, _payload = _status([str(plan)], capsys)
        assert with_record == text
        assert "  suggested next: generate-tasks" in text and "cascade" not in text
        _code, text, _payload = _status([str(plan), "--json"], capsys)
        assert "cascades" not in json.loads(text)

    def test_with_a_composite_run_view_the_block_is_not_repeated(self, here, capsys):
        plan = self._plan(here)
        _composite_run(plan, "improve-plan", replay_flags=["--models", *MODELS, "--cascade"])
        _code, text, _payload = _status([str(plan)], capsys)
        assert "improve-plan  no run record" not in text
        assert text.count("cascade:        stage 2 of 3") == 1
        assert self.SUGGESTED in text.splitlines()
        _code, text, _payload = _status([str(plan), "--json"], capsys)
        assert "cascades" not in json.loads(text)

    def test_an_unreadable_record_costs_the_block_not_the_poll(self, here, capsys):
        plan = self._plan(here)
        directory = ic.load_current(plan, "improve-plan").cascade_dir
        (directory / ic.RECORD_NAME).write_text("{", encoding="utf-8")
        _code, text, _payload = _status([str(plan)], capsys)
        assert "improve-plan  no run record" not in text and "cascade" not in text
        assert "  suggested next: generate-tasks" in text

    def test_the_pointer_name_is_the_cores(self):
        assert status_views.CASCADE_POINTER_NAME == ic.POINTER_NAME

    def test_no_pointer_loads_no_cascade_core(self, here, monkeypatch):
        plan = _build_outside(here, DONE["improve-plan"])

        def refuse(*_args, **_kwargs):
            raise AssertionError("the cascade core was asked with no pointer on disk")

        monkeypatch.setattr(np_, "unfinished_cascade", refuse)
        assert status_views.unfinished_cascades(plan) == []


# ---------------------------------------------------------------------------
# Help and the argument hint
# ---------------------------------------------------------------------------

def _help(argv: List[str]) -> Optional[int]:
    """The dispatcher's exit code for a ``--help``, whether it returns or raises."""
    try:
        return multi_llm.main(argv)
    except SystemExit as exc:
        return exc.code or 0


class TestHelpSurfaces:
    @pytest.mark.parametrize("argv", [
        ["improve-plan", "--help"],
        ["improve-code", "--help"],
        ["util", "improve-steps", "--help"],
    ], ids=["improve-plan", "improve-code", "improve-steps"])
    def test_the_composites_show_cascade_and_never_the_hidden_flag(self, argv, capsys):
        assert _help(argv) == 0
        out = capsys.readouterr().out
        assert "--cascade" in out
        assert HIDDEN_FLAG not in out

    def test_improve_commits_takes_no_cascade(self, capsys):
        assert _help(["improve-commits", "--help"]) == 0
        out = capsys.readouterr().out
        assert "--cascade" not in out and HIDDEN_FLAG not in out

    def test_the_argument_hint_names_cascade_and_never_the_hidden_flag(self):
        hint = modes.render_argument_hint()
        assert "[--cascade (--improve-plan/--improve-code only)]" in hint
        assert HIDDEN_FLAG not in hint

    def test_the_recorded_continuation_adds_cascade_before_resume(self):
        assert iw.recorded_continuation("improve-plan", None) == ["--resume"]
        assert iw.recorded_continuation("improve-plan", None, cascade=True) == [
            "--cascade", "--resume"]
        assert iw.recorded_continuation(
            "improve-code", ["--force", "--cascade", "--models", "a"], cascade=True,
        ) == ["--models", "a", "--cascade", "--resume"]
