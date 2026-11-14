"""``--cascade`` on the native leg: ``util improve-steps``' hand-off and ``util improve-cascade next``.

The native cores plan with ``util improve-steps``, which stays read-only: a
cascade gets ``[IMPROVE_CASCADE_NEXT] <command>`` and ``[IMPROVE_RERUN]`` in
place of a step block. The command it names, ``util improve-cascade next``,
is the cascade's one writer on this leg: each call closes out the stage just
run and prints the next stage's step block with ``util improve-steps``' own
printers, then ``[IMPROVE_CASCADE_CONTINUE]`` and ``[IMPROVE_RERUN]``.

These tests drive the dispatcher as a core does, running each printed
command's own tokens, and fabricate what each stage's review and apply leave
on disk between the calls (``tests/support/cascade_world.py``), so the real
``improve_cascade.advance``, close-out, archive and ledger run.

Contract: docs/development/workflows.md, "A cascade (``--cascade``)".

Usage:
    uv run --project skills/multi-llm -- pytest skills/multi-llm/tests/test_improve_cascade_native.py -q
"""

import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import multi_llm  # noqa: E402
from utils import cascade_provenance as provenance  # noqa: E402
from utils import detach, status_file  # noqa: E402
from utils import improve_cascade as ic  # noqa: E402
from utils import improve_cascade_next as icn  # noqa: E402
from utils import improve_steps  # noqa: E402
from utils import improve_workflow as iw  # noqa: E402
from utils import report_check  # noqa: E402
from utils.cli_common import _quote, dispatcher_prefix  # noqa: E402

from tests.support.cascade_world import CascadeWorld, groups_of, ids, member  # noqa: E402

PLAN, CODE = "improve-plan", "improve-code"
MODES = [PLAN, CODE]
M1, M2, M3 = "codex:gpt-a", "codex:gpt-b", "codex:gpt-c"
TYPED = "plans/export.md"
HOST = ("--host", "claude-code")

NEXT = improve_steps.CASCADE_NEXT_MARKER
RERUN = improve_steps.RERUN_MARKER
CONTINUE = icn.CONTINUE_MARKER
COMPLETE = icn.COMPLETE_MARKER
STAGE_LINE = re.compile(r"^\[IMPROVE_CASCADE\] stage (\d+) of (\d+): (\S+)$", re.M)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A git repository holding the plan, as the cwd, with no host declared."""
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE", detach.PLAN_ARG_ENV,
                 detach.WORKER_ENV):
        monkeypatch.delenv(name, raising=False)
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "init"], check=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def marker(out: str, name: str) -> str:
    """The text after *name* on the one line it opens."""
    found = [line[len(name) + 1:] for line in out.splitlines() if line.startswith(name + " ")]
    assert len(found) == 1, (name, out)
    return found[0]


def has(out: str, name: str) -> bool:
    return any(line.startswith(name) for line in out.splitlines())


def stage_of(out: str) -> Tuple[int, int, str]:
    (found,) = STAGE_LINE.findall(out)
    return int(found[0]), int(found[1]), found[2]


def files_under(directory: Path) -> Dict[str, bytes]:
    """Every file below *directory* with its bytes, the phase locks left out:
    the planner takes the composite run dir's lock, which records its holder
    in ``.lock``, the lock's file and not the cascade's state."""
    if not directory.exists():
        return {}
    return {p.relative_to(directory).as_posix(): p.read_bytes()
            for p in sorted(directory.rglob("*"))
            if p.is_file() and not p.name.endswith(".lock")}


class Native:
    """One plan driven as a core drives it: the planner, the printed commands,
    and each stage's phases fabricated between the calls."""

    def __init__(self, root: Path, mode: str, capsys, models=(M1, M2, M3)):
        self.world = CascadeWorld(root, mode, of=len(models), models=models)
        # The world lays out a cascade dir of its own; advance makes the real one.
        shutil.rmtree(self.world.run_dir)
        self.mode = mode
        self.capsys = capsys
        self.clock = 0.0

    # -- the dispatcher, as a host's Bash call runs it ----------------------
    def dispatch(self, argv: List[str]) -> Tuple[int, str, str]:
        code = multi_llm.main(list(argv))
        captured = self.capsys.readouterr()
        out = "\n".join(line for line in captured.out.splitlines()
                        if not line.startswith("MULTI_LLM_RESULT:"))
        return code, out, captured.err

    def steps(self, *typed: str) -> Tuple[int, str, str]:
        """Section 1: ``util improve-steps`` with the user's flags."""
        return self.dispatch(["util", "improve-steps", TYPED, "--mode", self.mode,
                              *HOST, *typed])

    def run_printed(self, command: str) -> Tuple[int, str, str]:
        """A printed command, exactly its tokens: the dispatcher prefix, then the rest."""
        tokens = shlex.split(command)
        prefix = shlex.split(dispatcher_prefix())
        assert tokens[:len(prefix)] == prefix, command
        return self.dispatch(tokens[len(prefix):])

    def start(self, *typed: str) -> Tuple[int, str, str]:
        """Section 1's hand-off, then the command after ``[IMPROVE_CASCADE_NEXT]``, once."""
        code, out, err = self.steps(*typed)
        assert code == 0, err
        if out.lstrip().startswith("{"):
            return self.run_printed(json.loads(out)["cascade_next"])
        return self.run_printed(marker(out, NEXT))

    # -- what a stage's phases leave on disk --------------------------------
    @staticmethod
    def review_flags(out: str) -> List[str]:
        return shlex.split(marker(out, f"{improve_steps.FLAGS_MARKER} 1"))

    def _manifest(self, out: str) -> Tuple[Path, str]:
        flags = self.review_flags(out)
        return (Path(flags[flags.index(provenance.HIDDEN_FLAG) + 1]),
                flags[flags.index("--models") + 1])

    def _bind(self, manifest: Path) -> None:
        path = self.world.review_dir / ".status.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data[provenance.STATUS_KEY] = provenance.block_from_manifest(manifest)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def run_stage(self, out: str) -> None:
        """Steps 1 and 2 of the stage *out* printed: one finding, reviewed and applied."""
        stage, _of, model = stage_of(out)
        manifest, flagged = self._manifest(out)
        assert flagged == model == self.world.models[stage - 1]
        code = self.mode == CODE
        groups = groups_of([member(
            f"Stage {stage} finding", f"A gap stage {stage}'s model found.", "medium",
            file="src/a.py" if code else None, model=model)], code=code, model=model)
        begin = self.clock
        self.clock += 10
        sha = self.world.write_review(stage, groups, verdicts=[("valid", "a real gap")],
                                      started=begin, completed=begin + 2,
                                      round_id=f"{stage:016x}")
        self._bind(manifest)
        self.world.write_apply(
            stage, sha, batches=[(f"apply-batch-{stage}", ids(groups),
                                  ["src/a.py"] if code else [str(self.world.plan)])],
            outcomes={ids(groups)[0]: ("applied", "")}, at=begin + 5,
            plan_edit=None if code else f"\n## Stage {stage}\nEdited by {model}.\n")

    def fail_review(self, out: str, reason: str = "quota exhausted") -> None:
        """Step 1 of the stage *out* printed ran and no model produced a result."""
        stage, _of, model = stage_of(out)
        manifest, _flagged = self._manifest(out)
        if self.world.review_dir.exists():
            shutil.rmtree(self.world.review_dir)
        self.world.review_dir.mkdir(parents=True)
        status_file.write_status(str(self.world.review_dir), {
            "phase": self.world.names.review_phase, "state": "complete",
            "round_id": f"{stage:016x}", "models_requested": [model],
            "models_completed": [], "models_failed": {model: reason},
            provenance.STATUS_KEY: provenance.block_from_manifest(manifest),
        })
        state = self.world.state()
        state.get("phases_completed", {}).pop(self.world.names.review_phase, None)
        self.world.save_state(state)

    # -- the cascade on record ----------------------------------------------
    def record(self) -> Dict[str, Any]:
        current = ic.load_current(self.world.plan, self.mode)
        assert current is not None
        return current.record

    def cascades(self) -> List[Path]:
        root = self.world.run_dir / ic.CASCADES_DIR
        return sorted(root.iterdir()) if root.exists() else []


# ---------------------------------------------------------------------------
# 1. util improve-steps hands a cascade off, writing nothing
# ---------------------------------------------------------------------------

class TestTheHandOff:
    @pytest.mark.parametrize("mode", MODES)
    def test_cascade_typed_writes_nothing_and_prints_the_hand_off(self, repo, capsys, mode):
        run = Native(repo, mode, capsys)
        before = files_under(run.world.out)
        code, out, err = run.steps("--cascade", "--models", M1, M2)
        assert code == 0, err
        assert files_under(run.world.out) == before
        assert out.splitlines() == [
            f"{NEXT} {marker(out, NEXT)}", f"{RERUN} {marker(out, RERUN)}"]
        assert not has(out, improve_steps.STEP_MARKER)
        # Every token the util was given, after `util improve-cascade next`.
        tokens = shlex.split(marker(out, NEXT))
        prefix = shlex.split(dispatcher_prefix())
        assert tokens[len(prefix):] == [
            "util", "improve-cascade", "next", TYPED, "--mode", mode, *HOST,
            "--cascade", "--models", M1, M2]
        rerun = shlex.split(marker(out, RERUN))
        assert rerun[-5:] == ["--models", M1, M2, "--cascade", "--resume"]
        assert f"--{mode}" in rerun and "--force" not in rerun

    def test_a_missing_host_is_added_so_the_next_call_names_the_same_one(
            self, repo, capsys, monkeypatch):
        run = Native(repo, PLAN, capsys)
        monkeypatch.setenv("MULTI_LLM_HOST", "claude-code")
        code, out, err = run.dispatch(["util", "improve-steps", TYPED, "--mode", PLAN,
                                       "--cascade", "--models", M1])
        assert code == 0, err
        tokens = shlex.split(marker(out, NEXT))
        assert tokens[-2:] == ["--host", "claude-code"]

    def test_the_json_object_adds_the_hand_off_and_renames_nothing(self, repo, capsys):
        run = Native(repo, PLAN, capsys)
        code, out, err = run.steps("--cascade", "--models", M1, M2, "--json")
        assert code == 0, err
        payload = json.loads(out)
        assert tuple(payload) == improve_steps.JSON_KEYS
        assert payload["steps"] == [] and payload["already_complete"] is None
        assert "--json" in shlex.split(payload["cascade_next"])
        assert "--cascade" in payload["rerun_argv"]

    def test_a_plan_without_a_cascade_keeps_its_object_with_a_null_hand_off(
            self, repo, capsys):
        run = Native(repo, PLAN, capsys)
        code, out, _err = run.steps("--models", M1, "--json")
        assert code == 0
        payload = json.loads(out)
        assert payload["cascade_next"] is None and payload["steps"]

    @pytest.mark.parametrize("mode", MODES)
    def test_an_unfinished_cascade_hands_off_without_the_flag(self, repo, capsys, mode):
        run = Native(repo, mode, capsys)
        code, _out, err = run.start("--cascade", "--models", M1, M2)
        assert code == 0, err
        before = files_under(run.world.out)
        code, out, err = run.steps()
        assert code == 0, err
        assert files_under(run.world.out) == before
        assert has(out, NEXT) and not has(out, improve_steps.STEP_MARKER)
        assert shlex.split(marker(out, RERUN))[-2:] == ["--cascade", "--resume"]

    def test_a_conflicting_model_list_is_refused_on_stderr_alone(self, repo, capsys):
        run = Native(repo, PLAN, capsys)
        run.start("--cascade", "--models", M1, M2)
        before = files_under(run.world.out)
        code, out, err = run.steps("--models", M3)
        assert code == 3
        assert out == ""
        assert err.startswith(f"{ic.CONFLICT_MARKER} --models")
        assert files_under(run.world.out) == before

    def test_a_finished_cascade_typed_again_is_already_complete(self, repo, capsys):
        run = Native(repo, PLAN, capsys, models=(M1,))
        code, out, err = run.start("--cascade", "--models", M1)
        run.run_stage(out)
        code, out, err = run.run_printed(marker(out, CONTINUE))
        assert code == 2, err
        before = files_under(run.world.out)
        code, out, err = run.steps("--cascade", "--models", M1)
        assert code == 2, err
        assert out.startswith(f"{improve_steps.ALREADY_COMPLETE_MARKER} cascade finished at ")
        assert len(out.splitlines()) == 1
        assert files_under(run.world.out) == before
        code, out, _err = run.steps("--cascade", "--models", M1, "--json")
        payload = json.loads(out)
        assert code == 2 and payload["already_complete"].startswith("cascade finished at")
        assert payload["steps"] == [] and payload["rerun"] is None

    @pytest.mark.parametrize("mode", MODES)
    def test_the_printed_continuation_typed_after_the_finish_is_already_complete(
            self, repo, capsys, monkeypatch, mode):
        """A continue call's ``[IMPROVE_RERUN]`` is ``--cascade --resume`` with
        no model flag; typed after the cascade finished it is
        ``[ALREADY_COMPLETE]``, never a new cascade of today's defaults (here
        another list, M3)."""
        monkeypatch.setattr(ic, "resolve_stage_models",
                            lambda plan, m, args: (list(args.models or [M3]), None))
        run = Native(repo, mode, capsys, models=(M1, M2))
        code, out, err = run.start("--cascade", "--models", M1, M2)
        reruns = []
        while code == 0:
            reruns.append(marker(out, RERUN))
            run.run_stage(out)
            code, out, err = run.run_printed(marker(out, CONTINUE))
        assert code == 2 and has(out, COMPLETE), err
        later = shlex.split(reruns[-1])
        assert later[-2:] == ["--cascade", "--resume"] and "--models" not in later
        before, cascades = files_under(run.world.out), run.cascades()
        typed = later[later.index(TYPED) + 1:]
        code, out, err = run.steps(*typed)
        assert code == 2, err
        assert out.startswith(f"{improve_steps.ALREADY_COMPLETE_MARKER} cascade finished at ")
        assert not has(out, NEXT)
        assert files_under(run.world.out) == before and run.cascades() == cascades
        # The first call of a hand-off made before the finish, run after it,
        # answers the same.
        code, out, err = run.dispatch(["util", "improve-cascade", "next", TYPED, "--mode", mode,
                                       *HOST, *typed])
        assert code == 2, err
        assert out.startswith(f"{improve_steps.ALREADY_COMPLETE_MARKER} cascade finished at ")
        assert files_under(run.world.out) == before and run.cascades() == cascades


# ---------------------------------------------------------------------------
# 2. util improve-cascade next prints the stage with improve-steps' printers
# ---------------------------------------------------------------------------

def _expected_stage_lines(mode: str, typed: List[str], plan: Path, stage: int) -> List[str]:
    """``plan_steps`` on the stage's namespace, rendered line by line, independently."""
    args = icn.parse_next_args([TYPED, "--mode", mode, *HOST, *typed])
    current = ic.load_current(plan, mode)
    manifest = current.cascade_dir / f"stage-{stage}" / "inputs" / provenance.MANIFEST_NAME
    stage_args = ic.derive_stage_args(args, mode, current, stage, force=True,
                                      manifest=manifest)
    records = iw.read_records(mode, plan, stage_args)
    lines = []
    for step in iw.plan_steps(mode, stage_args, records):
        head = f"{improve_steps.STEP_MARKER} {step.index} {step.step} {step.decision}"
        lines.append(head if step.reason is None else f"{head} ({step.reason})")
        if step.fresh:
            lines.append(f"{improve_steps.FRESH_MARKER} {step.index}")
        lines.append(" ".join([improve_steps.FLAGS_MARKER, step.index,
                               *(_quote(t) for t in step.flags)]))
    return lines


class TestNext:
    @pytest.mark.parametrize("mode", MODES)
    def test_the_first_stage_prints_the_step_markers_plan_steps_gives_it(
            self, repo, capsys, mode):
        run = Native(repo, mode, capsys)
        typed = ["--cascade", "--models", M1, M2]
        code, out, err = run.start(*typed)
        assert code == 0, err
        assert stage_of(out) == (1, 2, M1)
        lines = out.splitlines()
        first = lines.index(f"{ic.CASCADE_MARKER} stage 1 of 2: {M1}")
        assert any(line.startswith(f"{ic.CASCADE_MARKER} new cascade ")
                   for line in lines[:first])
        body = lines[first + 1:-2]
        assert body == _expected_stage_lines(mode, typed, run.world.plan, 1)
        assert lines[-2].startswith(f"{CONTINUE} ") and lines[-1].startswith(f"{RERUN} ")
        flags = run.review_flags(out)
        assert flags[flags.index("--models") + 1] == M1 and flags.count("--models") == 1
        assert "--force" in flags and "--rerun-all" in flags
        assert flags[-2] == provenance.HIDDEN_FLAG and Path(flags[-1]).is_absolute()
        if mode == PLAN:
            assert f"{improve_steps.STEP_MARKER} 1b consolidate skip (cascade)" in lines

    def test_the_continue_line_carries_the_id_and_no_selector(self, repo, capsys):
        run = Native(repo, PLAN, capsys)
        code, out, err = run.start("--force", "--cascade", "--models", M1, M2,
                                   "--scope-policy", "preserve", "--yes",
                                   "--exec-subagents")
        assert code == 0, err
        tokens = shlex.split(marker(out, CONTINUE))
        tail = tokens[len(shlex.split(dispatcher_prefix())):]
        assert tail == ["util", "improve-cascade", "next", TYPED, "--mode", PLAN, *HOST,
                        icn.CASCADE_ID_FLAG, run.record()["cascade_id"],
                        icn.AFTER_STAGE_FLAG, "1", "--yes", "--exec-subagents"]
        # The step flags lines keep the cross-cutting tail, as improve-steps'.
        flags = run.review_flags(out)
        assert flags[-1] == "--exec-subagents" and flags[-3] == provenance.HIDDEN_FLAG

    def test_the_json_object_is_improve_steps_plus_cascade(self, repo, capsys):
        run = Native(repo, PLAN, capsys)
        code, out, err = run.start("--cascade", "--models", M1, M2, "--json")
        assert code == 0, err
        payload = json.loads(out)
        assert tuple(payload) == icn.JSON_KEYS
        assert tuple(payload["cascade"]) == icn.CASCADE_KEYS
        cascade = payload["cascade"]
        assert (cascade["state"], cascade["stage"], cascade["of"], cascade["model"]) == (
            "stage", 1, 2, M1)
        assert cascade["cascade_id"] == run.record()["cascade_id"]
        assert "--json" in cascade["continue_argv"]
        assert payload["cascade_next"] is None
        assert [s["index"] for s in payload["steps"]] == ["1", "1b", "2"]
        assert f"{ic.CASCADE_MARKER} stage 1 of 2: {M1}" in payload["notices"]
        assert "--cascade" in payload["rerun_argv"]

    @pytest.mark.parametrize("mode", MODES)
    def test_the_cascade_runs_to_complete_one_stage_per_continue_call(
            self, repo, capsys, mode):
        run = Native(repo, mode, capsys)
        code, out, err = run.start("--cascade", "--models", M1, M2, M3)
        seen = []
        while code == 0:
            seen.append(stage_of(out))
            run.run_stage(out)
            code, out, err = run.run_printed(marker(out, CONTINUE))
        assert code == 2, err
        assert seen == [(1, 3, M1), (2, 3, M2), (3, 3, M3)]
        summary = run.record()
        table = marker(out, COMPLETE)
        assert table.startswith("3 stages; table: ")
        assert Path(table.split("table: ", 1)[1]).resolve() == (
            ic.load_current(run.world.plan, mode).summary_path.resolve())
        assert summary["state"] == ic.FINISHED
        assert [s["state"] for s in summary["stages"]] == [ic.DONE] * 3
        cascade_dir = ic.load_current(run.world.plan, mode).cascade_dir
        for k in (1, 2, 3):
            assert (cascade_dir / f"stage-{k}" / "artifacts" / ".complete.json").is_file()
        assert (cascade_dir / "summary.md").is_file()
        # Exit 2 prints no continuation: section 4's lookup gives the next step.
        assert not has(out, RERUN) and not has(out, CONTINUE)

    def test_stage_2s_review_reads_the_ledger_stage_1_left(self, repo, capsys):
        run = Native(repo, PLAN, capsys)
        code, out, _err = run.start("--cascade", "--models", M1, M2)
        run.run_stage(out)
        code, out, err = run.run_printed(marker(out, CONTINUE))
        assert code == 0, err
        manifest = Path(run.review_flags(out)[-1])
        ledger = (manifest.parent / provenance.REVIEWER_LEDGER).read_text(encoding="utf-8")
        assert "Stage 1 finding" in ledger
        # The stage's own round starts fresh, and only once.
        flags = run.review_flags(out)
        assert "--force" in flags and f"{improve_steps.FRESH_MARKER} 1" in out


# ---------------------------------------------------------------------------
# 3. Restarts, abandons and conflicts
# ---------------------------------------------------------------------------

class TestRestartAndAbandon:
    @pytest.mark.parametrize("mode", MODES)
    def test_force_cascade_then_the_continue_commands_run_each_stage_once(
            self, repo, capsys, mode):
        run = Native(repo, mode, capsys)
        # An unfinished cascade with stage 1 done is on record.
        code, out, _err = run.start("--cascade", "--models", M1, M2, M3)
        run.run_stage(out)
        run.run_printed(marker(out, CONTINUE))
        old_id = run.record()["cascade_id"]

        code, out, err = run.start("--force", "--cascade", "--models", M1, M2, M3)
        assert code == 0, err
        assert any(line.startswith("NOTE: --force --cascade abandons the unfinished cascade")
                   for line in out.splitlines())
        new_id = run.record()["cascade_id"]
        assert new_id != old_id
        seen, ids_named = [], []
        while code == 0:
            seen.append(stage_of(out))
            command = marker(out, CONTINUE)
            tokens = shlex.split(command)
            assert "--force" not in tokens and "--cascade" not in tokens
            assert "--models" not in tokens
            ids_named.append(tokens[tokens.index(icn.CASCADE_ID_FLAG) + 1])
            flags = run.review_flags(out)
            assert "--force" in flags and "--rerun-all" in flags  # each stage starts once
            run.run_stage(out)
            code, out, err = run.run_printed(command)
        assert code == 2, err
        assert seen == [(1, 3, M1), (2, 3, M2), (3, 3, M3)]
        assert set(ids_named) == {new_id}
        assert len(run.cascades()) == 2
        old = json.loads(next(p for p in run.cascades() if p.name.endswith(old_id[:8]))
                         .joinpath(ic.RECORD_NAME).read_text(encoding="utf-8"))
        assert old["state"] == ic.ABANDONED
        assert [s["state"] for s in run.record()["stages"]] == [ic.DONE] * 3

    @pytest.mark.parametrize("mode", MODES)
    def test_force_without_cascade_abandons_and_prints_a_plain_block(
            self, repo, capsys, mode):
        run = Native(repo, mode, capsys)
        run.start("--cascade", "--models", M1, M2)
        pointer = ic.pointer_path(run.world.plan, mode)
        record_path = ic.load_current(run.world.plan, mode).record_path
        before = files_under(run.world.out)
        code, out, err = run.steps("--force", "--models", M1, M2)
        assert code == 0, err
        assert files_under(run.world.out) == before  # the planner still writes nothing
        assert "--cascade" not in shlex.split(marker(out, RERUN))
        code, out, err = run.run_printed(marker(out, NEXT))
        assert code == 0, err
        assert out.splitlines()[0].startswith(
            "NOTE: --force without --cascade abandons the unfinished cascade")
        assert not has(out, CONTINUE) and not has(out, ic.CASCADE_MARKER)
        flags = run.review_flags(out)
        assert flags[flags.index("--models") + 1:][:2] == [M1, M2]
        assert "--force" in flags and provenance.HIDDEN_FLAG not in flags
        rerun = shlex.split(marker(out, RERUN))
        assert "--cascade" not in rerun and rerun[-1] == "--resume"
        if mode == PLAN:
            assert not has(out, f"{improve_steps.STEP_MARKER} 1b consolidate skip")
        assert not pointer.exists()
        assert json.loads(record_path.read_text(encoding="utf-8"))["state"] == ic.ABANDONED

    @pytest.mark.parametrize("mode", MODES)
    def test_then_the_stale_cascade_rerun_is_refused_on_both_planner_calls(
            self, repo, capsys, monkeypatch, mode):
        """The ``[IMPROVE_RERUN]`` the cascade printed (``--cascade --resume``,
        no model flag), typed after ``--force`` alone abandoned it: section 1
        and a hand-off's first call both refuse it at exit 3 on stderr alone,
        writing nothing, never a new cascade of today's defaults (M3 here)."""
        monkeypatch.setattr(ic, "resolve_stage_models",
                            lambda plan, m, args: (list(args.models or [M3]), None))
        run = Native(repo, mode, capsys)
        code, out, err = run.start("--cascade", "--models", M1, M2)
        run.run_stage(out)
        code, out, err = run.run_printed(marker(out, CONTINUE))
        assert code == 0 and stage_of(out)[0] == 2, err
        stale = shlex.split(marker(out, RERUN))
        assert stale[-2:] == ["--cascade", "--resume"] and "--models" not in stale
        code, out, err = run.steps("--force", "--models", M1)
        code, out, err = run.run_printed(marker(out, NEXT))
        assert code == 0, err
        assert ic.load_current(run.world.plan, mode) is None
        before, cascades = files_under(run.world.out), run.cascades()
        typed = stale[stale.index(TYPED) + 1:]
        for argv in (["util", "improve-steps", TYPED, "--mode", mode, *HOST, *typed],
                     ["util", "improve-cascade", "next", TYPED, "--mode", mode, *HOST,
                      *typed]):
            code, out, err = run.dispatch(argv)
            assert code == 3, (argv, out, err)
            assert out == ""
            assert err.startswith(f"{ic.CONFLICT_MARKER} --resume: there is no unfinished "
                                  f"{mode} cascade to resume ("), err
            assert "--cascade without --resume, or with --models or --quick, starts one." in err
            assert files_under(run.world.out) == before and run.cascades() == cascades

    def test_a_continue_call_naming_another_cascade_is_refused(self, repo, capsys):
        run = Native(repo, PLAN, capsys)
        code, out, _err = run.start("--cascade", "--models", M1, M2)
        tokens = shlex.split(marker(out, CONTINUE))
        position = tokens.index(icn.CASCADE_ID_FLAG) + 1
        tokens[position] = "f" * 32
        before = files_under(run.world.out)
        code, out, err = run.run_printed(" ".join(_quote(t) for t in tokens))
        assert code == 3
        assert out == ""
        assert err.startswith(f"{ic.CONFLICT_MARKER} {icn.CASCADE_ID_FLAG}: ")
        assert files_under(run.world.out) == before


# ---------------------------------------------------------------------------
# 4. Zero-result stages
# ---------------------------------------------------------------------------

class TestZeroResultStages:
    def test_a_middle_stage_is_skipped_by_the_continue_call(self, repo, capsys):
        run = Native(repo, PLAN, capsys, models=(M1, M2))
        code, out, _err = run.start("--cascade", "--models", M1, M2)
        run.fail_review(out, "quota exhausted")
        code, out, err = run.run_printed(marker(out, CONTINUE))
        assert code == 0, err
        assert f"{ic.SKIPPED_MARKER} 1 {M1}: quota exhausted" in out.splitlines()
        assert stage_of(out) == (2, 2, M2)
        assert [s["state"] for s in run.record()["stages"]] == [ic.SKIPPED, ic.STAGE_RUNNING]

    def test_the_last_stage_stops_with_the_rerun_and_the_rerun_starts_it_over(
            self, repo, capsys):
        run = Native(repo, PLAN, capsys, models=(M1, M2))
        code, out, _err = run.start("--cascade", "--models", M1, M2)
        run.run_stage(out)
        code, out, _err = run.run_printed(marker(out, CONTINUE))
        assert stage_of(out) == (2, 2, M2)
        run.fail_review(out, "rate limited")
        code, out, err = run.run_printed(marker(out, CONTINUE))
        assert code == 3
        assert "stage 2 of 2" in err and "rate limited" in err
        assert out.splitlines() == [f"{RERUN} {marker(out, RERUN)}"]
        rerun = shlex.split(marker(out, RERUN))
        assert rerun[-2:] == ["--cascade", "--resume"]
        # The user's command: section 1 again, which hands off; the first call
        # starts stage 2 over.
        code, out, err = run.start(*rerun[rerun.index(TYPED) + 1:])
        assert code == 0, err
        assert stage_of(out) == (2, 2, M2)
        assert "--force" in run.review_flags(out)

    def test_the_stopped_last_stage_as_json(self, repo, capsys):
        run = Native(repo, PLAN, capsys, models=(M1,))
        code, out, _err = run.start("--cascade", "--models", M1, "--json")
        payload = json.loads(out)
        command = payload["cascade"]["continue"]
        manifest = Path(payload["steps"][0]["argv"][
            payload["steps"][0]["argv"].index(provenance.HIDDEN_FLAG) + 1])
        run.world.review_dir.mkdir(parents=True)
        status_file.write_status(str(run.world.review_dir), {
            "phase": "review-plan", "state": "complete", "round_id": "1" * 16,
            "models_requested": [M1], "models_completed": [], "models_failed": {M1: "x"},
            provenance.STATUS_KEY: provenance.block_from_manifest(manifest)})
        code, out, err = run.run_printed(command)
        assert code == 3, err
        payload = json.loads(out)
        assert payload["cascade"]["state"] == "stopped"
        assert payload["cascade"]["stage"] == payload["cascade"]["of"] == 1
        assert "--cascade" in payload["rerun_argv"]


class TestAContinueCallFollowsItsStage:
    """``[IMPROVE_CASCADE_CONTINUE]`` carries ``--after-stage <k>``: when stage
    k is still open the call stops at exit 3 with the user's continuation,
    and never prints stage k's block again at exit 0 (a host following the
    printed block would run the paid review over and over)."""

    def _open_round(self, run: "Native", out: str) -> None:
        """Step 1 ran, opened its round with the stage's block, and stopped
        for another reason than no result (a validation decision owed)."""
        stage, _of, model = stage_of(out)
        manifest, _flagged = run._manifest(out)
        run.world.review_dir.mkdir(parents=True, exist_ok=True)
        status_file.write_status(str(run.world.review_dir), {
            "phase": run.world.names.review_phase, "state": "models_complete",
            "round_id": f"{stage:016x}", "models_requested": [model],
            "models_completed": [model], "models_failed": {},
            provenance.STATUS_KEY: provenance.block_from_manifest(manifest),
        })

    @pytest.mark.parametrize("mode", MODES)
    def test_a_review_stop_with_results_then_the_continue_call_is_exit_3(
            self, repo, capsys, mode):
        run = Native(repo, mode, capsys, models=(M1, M2))
        code, out, _err = run.start("--cascade", "--models", M1, M2)
        assert stage_of(out) == (1, 2, M1)
        command = marker(out, CONTINUE)
        assert shlex.split(command)[-2:] == [icn.AFTER_STAGE_FLAG, "1"]
        self._open_round(run, out)
        for _attempt in range(3):
            code, out2, err = run.run_printed(command)
            assert code == 3, (out2, err)
            assert not has(out2, ic.CASCADE_MARKER) and not has(out2, improve_steps.STEP_MARKER)
            assert out2.splitlines() == [f"{RERUN} {marker(out2, RERUN)}"]
            assert shlex.split(marker(out2, RERUN))[-2:] == ["--cascade", "--resume"]
            assert f"stage 1 of 2 ({M1}) did not finish" in err
        assert [s["state"] for s in run.record()["stages"]] == [ic.STAGE_RUNNING, ic.PENDING]
        # The user's continuation picks stage 1 up again, without a new round.
        rerun = shlex.split(marker(out2, RERUN))
        code, out, err = run.start(*rerun[rerun.index(TYPED) + 1:])
        assert code == 0, err
        assert stage_of(out) == (1, 2, M1)
        assert "--force" not in run.review_flags(out)

    def test_a_continue_call_before_the_stage_ran_is_exit_3(self, repo, capsys):
        run = Native(repo, PLAN, capsys, models=(M1, M2))
        code, out, _err = run.start("--cascade", "--models", M1, M2)
        before = files_under(run.world.out)
        code, out2, err = run.run_printed(marker(out, CONTINUE))
        assert code == 3
        assert out2.splitlines() == [f"{RERUN} {marker(out2, RERUN)}"]
        assert "stage 1 of 2" in err and "no review round of this stage is on record" in err
        assert files_under(run.world.out) == before

    def test_the_stop_as_json_is_stopped_with_the_rerun(self, repo, capsys):
        run = Native(repo, PLAN, capsys, models=(M1, M2))
        code, out, _err = run.start("--cascade", "--models", M1, M2, "--json")
        command = json.loads(out)["cascade"]["continue"]
        assert json.loads(out)["cascade"]["continue_argv"][-3:] == [
            icn.AFTER_STAGE_FLAG, "1", "--json"]
        code, out, err = run.run_printed(command)
        assert code == 3, err
        payload = json.loads(out)
        assert (payload["cascade"]["state"], payload["cascade"]["stage"]) == ("stopped", 1)
        assert payload["steps"] == [] and "--cascade" in payload["rerun_argv"]

    def test_a_zero_result_middle_stage_is_still_skipped_at_exit_0(self, repo, capsys):
        run = Native(repo, PLAN, capsys, models=(M1, M2))
        code, out, _err = run.start("--cascade", "--models", M1, M2)
        run.fail_review(out)
        code, out, err = run.run_printed(marker(out, CONTINUE))
        assert code == 0, err
        assert has(out, ic.SKIPPED_MARKER) and stage_of(out) == (2, 2, M2)
        assert shlex.split(marker(out, CONTINUE))[-2:] == [icn.AFTER_STAGE_FLAG, "2"]


# ---------------------------------------------------------------------------
# 5. The Stop hook reads the real lines
# ---------------------------------------------------------------------------

class TestTheReportCheckReadsTheRealLines:
    def test_every_printed_cascade_command_is_a_planner_call(self, repo, capsys):
        """The hand-off's and each stage's command match the planner pattern, so
        the `[IMPROVE_RERUN]` each prints is read as the run's continuation."""
        run = Native(repo, PLAN, capsys)
        code, out, _err = run.steps("--cascade", "--models", M1, M2)
        assert report_check._PLANNER_RE.search(marker(out, NEXT))
        code, out, err = run.run_printed(marker(out, NEXT))
        assert code == 0, err
        assert report_check._PLANNER_RE.search(marker(out, CONTINUE))
        blocks = report_check.find_blocks(out, continuations=True)
        assert [(b.command, b.continuation) for b in blocks] == [(marker(out, RERUN), True)]
        # Without continuations (the wakeup check) a stage block holds no block.
        assert report_check.find_blocks(out) == []


# ---------------------------------------------------------------------------
# 6. The dispatcher's tables know the util
# ---------------------------------------------------------------------------

class TestTheDispatcherTables:
    def test_it_is_a_util_command(self):
        module, attr, _summary = multi_llm.UTIL_COMMANDS["improve-cascade"]
        assert (module, attr) == ("utils.improve_cascade_next", "main")
        assert "util improve-cascade next" in multi_llm.render_util_help()

    def test_the_plan_is_the_positional_after_next(self):
        assert multi_llm._UTIL_PATH_POSITIONAL_INDEX["improve-cascade"] == 1
        argv = ["next", "--models", M1, M2, "--mode", PLAN, TYPED, "--cascade-id", "x"]
        assert multi_llm._first_positional(
            argv, multi_llm._UTIL_PATH_POSITIONAL["improve-cascade"],
            multi_llm._UTIL_VARIADIC_OPTIONS["improve-cascade"],
            skip=multi_llm._UTIL_PATH_POSITIONAL_INDEX["improve-cascade"]) == TYPED

    @pytest.mark.parametrize("mode", [None, PLAN, CODE])
    def test_the_option_tables_are_the_parsers_own(self, mode):
        parser = icn.build_next_parser(mode)
        variadic = {option for action in parser._actions
                    if action.nargs in ("+", "*") for option in action.option_strings}
        valued = {action.option_strings[0] for action in parser._actions
                  if action.option_strings and action.nargs != 0}
        assert variadic == set(multi_llm._UTIL_VARIADIC_OPTIONS["improve-cascade"])
        assert valued <= set(multi_llm._UTIL_PATH_POSITIONAL["improve-cascade"])
        assert "--files-from" in multi_llm._UTIL_INPUT_PATH_OPTIONS

    def test_the_namespace_is_improve_steps_own(self):
        typed = [TYPED, "--mode", CODE, *HOST, "--cascade", "--models", M1, "--files",
                 "a.py", "--base-ref", "main", "--scope-policy", "preserve"]
        ours = vars(icn.parse_next_args(typed))
        theirs = vars(improve_steps.parse_args(typed))
        assert ours.pop("cascade_id") is None
        assert ours.pop("after_stage") is None
        assert ours == theirs

    def test_after_stage_goes_with_a_cascade_id(self, capsys):
        with pytest.raises(SystemExit):
            icn.parse_next_args([TYPED, "--mode", PLAN, *HOST, icn.AFTER_STAGE_FLAG, "1"])
        assert f"{icn.AFTER_STAGE_FLAG} goes with {icn.CASCADE_ID_FLAG}" in capsys.readouterr().err

    def test_improve_commits_takes_no_cascade(self, capsys):
        with pytest.raises(SystemExit):
            icn.parse_next_args([TYPED, "--mode", "improve-commits"])
        assert "invalid choice" in capsys.readouterr().err
