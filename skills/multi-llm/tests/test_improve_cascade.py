"""``--cascade``'s core: the pointer and record, the stage list, binding, and the resume table.

``utils/improve_cascade.py`` decides what a cascade runs; the archive
(``utils/cascade_archive.py``) and the ledger (``utils/cascade_ledger.py``)
are replaced here by fakes that keep their call signatures, so every test is a
pure function or a file layout in ``tmp_path``. A stage's review and apply are
simulated by writing what they leave: the review dir's ``.status.json`` (its
round id, model lists and provenance block) and ``state.json``'s completions.

The contract: docs/development/workflows.md, "A cascade (``--cascade``)".

Usage:
    uv run --project skills/multi-llm -- pytest skills/multi-llm/tests/test_improve_cascade.py -q
"""

import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import utils  # noqa: E402
from utils import cascade_ledger as real_ledger  # noqa: E402
from utils import cascade_provenance as provenance  # noqa: E402
from utils import improve_cascade as ic  # noqa: E402
from utils import improve_steps  # noqa: E402
from utils import improve_workflow as iw  # noqa: E402
from utils import interactive, provider_registry, status_file  # noqa: E402
from utils.output_handler import phase_dir_path  # noqa: E402
from code_review_orchestrator import REVIEW_SCOPE_FILES_KEY  # noqa: E402

from tests.support.improve_world import (  # noqa: E402
    APPLY_PHASE,
    CODE,
    PLAN,
    REVIEW_PHASE,
    T,
    _out,
    _plan,
    _write_state,
    improve_repo,  # noqa: F401 - the `repo` fixture
)

SKILL_ROOT = Path(__file__).parent.parent
SCHEMA_PATH = SKILL_ROOT / "schemas" / "cascade_record.schema.json"

M1, M2, M3 = "codex:gpt-a", "codex:gpt-b", "codex:gpt-c"
ROUND_1, ROUND_2, ROUND_3 = "1" * 16, "2" * 16, "3" * 16


# ---------------------------------------------------------------------------
# Fakes of the archive and the ledger (their signatures, keyword-only)
# ---------------------------------------------------------------------------

def _counts(**overrides: Any) -> Dict[str, Any]:
    counts = dict(findings=3, valid=2, invalid=1, needs_decision=0, reraises=0,
                  applied=2, no_op=0, declined=0, skipped_by_user=0, judge_approved=0,
                  judge_salvaged=0, judge_skipped=0, nothing_to_apply=False,
                  time_s=12.5, cost=None)
    counts.update(overrides)
    return counts


class _Archive(types.ModuleType):
    """``utils.cascade_archive`` as the contract states it, recording each call."""

    class ArchiveRefusal(Exception):
        def __init__(self, message: str):
            super().__init__(message)
            self.message = message

    class CloseOutRaced(Exception):
        pass

    def __init__(self) -> None:
        super().__init__("utils.cascade_archive")
        self.close_outs: List[Dict[str, Any]] = []
        self.published: Dict[tuple, Dict[str, Any]] = {}
        self.summaries: List[Dict[str, Any]] = []
        self.race = 0
        self.refuse: Optional[str] = None

    def close_out(self, *, plan_path, mode, cascade_dir, cascade_id, stage, model,
                  round_id, recheck):
        self.close_outs.append(dict(plan_path=plan_path, mode=mode, cascade_dir=cascade_dir,
                                    cascade_id=cascade_id, stage=stage, model=model,
                                    round_id=round_id, recheck=recheck()))
        if self.refuse:
            raise self.ArchiveRefusal(self.refuse)
        if self.race:
            self.race -= 1
            raise self.CloseOutRaced()
        counts = _counts(findings=stage)
        self.published[(str(cascade_dir), stage, cascade_id)] = counts
        return counts

    def published_counts(self, cascade_dir, stage, cascade_id):
        return self.published.get((str(cascade_dir), stage, cascade_id))

    def stage_counts(self, artifacts_dir, mode):  # pragma: no cover - not called by core
        return _counts()

    def render_summary(self, cascade_dir, record):
        self.summaries.append(json.loads(json.dumps(record)))
        text = "| Stage | Model |\n" + "".join(
            f"| {s['index']} | {s['model']} |\n" for s in record["stages"])
        (Path(cascade_dir) / "summary.md").write_text(text, encoding="utf-8")
        return text


def _freeze_inputs(*, cascade_dir, cascade_id, mode, stage, of, model, plan_path):
    """``cascade_ledger.freeze_inputs``: empty ledgers, the plan, the manifest last."""
    inputs = Path(cascade_dir) / f"stage-{stage}" / "inputs"
    manifest = inputs / provenance.MANIFEST_NAME
    if manifest.exists():
        provenance.load_manifest(manifest)
        return manifest
    inputs.mkdir(parents=True, exist_ok=True)
    names = list(provenance.LEDGER_NAMES)
    for name in names:
        (inputs / name).write_text("", encoding="utf-8")
    if mode == PLAN:
        (inputs / provenance.PLAN_BEFORE).write_bytes(Path(plan_path).read_bytes())
        (inputs / provenance.CHANGES_SO_FAR).write_text("", encoding="utf-8")
        names += [provenance.PLAN_BEFORE, provenance.CHANGES_SO_FAR]
    return provenance.write_manifest(inputs, cascade_id=cascade_id, mode=mode, stage=stage,
                                     of=of, model=model, names=names)


@pytest.fixture
def archive(monkeypatch):
    fake = _Archive()
    ledger = types.ModuleType("utils.cascade_ledger")
    ledger.LEDGER_INLINE_MAX_CHARS = 8000
    ledger.freeze_inputs = _freeze_inputs
    # The real mark reader: the core asks it, before any binding, about an
    # ``artifacts/`` the archive's published_counts did not accept.
    ledger.read_complete = real_ledger.read_complete
    ledger.ArchiveRefusal = real_ledger.ArchiveRefusal
    ledger.calls = []
    for name, module in (("cascade_archive", fake), ("cascade_ledger", ledger)):
        monkeypatch.setitem(sys.modules, f"utils.{name}", module)
        monkeypatch.setattr(utils, name, module, raising=False)
    return fake


# ---------------------------------------------------------------------------
# The world: a plan, its review round, its records
# ---------------------------------------------------------------------------

def _args(mode: str, plan: Path, *argv: str):
    return iw.parse_args(mode, ["--plan-file", str(plan), *argv])


def _review_dir(plan: Path, mode: str) -> Path:
    return phase_dir_path(plan, REVIEW_PHASE[mode])


def _round(plan: Path, mode: str, *, manifest: Optional[Path], round_id: str,
           completed: Optional[List[str]] = None,
           failed: Optional[Dict[str, str]] = None) -> None:
    """A review round's checkpoint: its id, its models, and the stage's block when flagged."""
    directory = _review_dir(plan, mode)
    directory.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {"phase": REVIEW_PHASE[mode], "state": "models_complete",
                            "round_id": round_id}
    if completed is not None:
        data["models_completed"] = completed
    if failed is not None:
        data["models_failed"] = failed
    if manifest is not None:
        data[provenance.STATUS_KEY] = provenance.block_from_manifest(manifest)
    status_file.write_status(str(directory), data)


def _done(plan: Path, mode: str, review_at: float, apply_at: Optional[float],
          scope: Optional[List[str]] = None) -> None:
    """``state.json`` as a review (and its apply) completed: review-code also
    records the scope it reviewed, which the stage's close-out compares."""
    completed = {REVIEW_PHASE[mode]: T(review_at)}
    if apply_at is not None:
        completed[APPLY_PHASE[mode]] = T(apply_at)
    _write_state(plan, completed=completed,
                 extra={REVIEW_SCOPE_FILES_KEY: scope} if scope else None)


def _record(plan: Path, mode: str) -> Dict[str, Any]:
    current = ic.load_current(plan, mode)
    assert current is not None
    return current.record


def _tree(plan: Path) -> Dict[str, bytes]:
    """Every file under the plan's output dir, with its bytes, its phase locks left out.

    ``advance`` plans under the composite run dir's phase lock, and taking a
    lock records its holder in the ``.lock`` file: that file is the lock's,
    not the cascade's state (a lock file left behind means nothing,
    ``process_control.PhaseLock.release``), so "nothing written" is about
    every other file.
    """
    root = _out(plan)
    if not root.exists():
        return {}
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*")
            if p.is_file() and not p.name.endswith(".lock")}


def _run_stage(plan: Path, mode: str, stage: ic.StagePlan, *, round_id: str,
               review_at: float, apply_at: Optional[float]) -> None:
    """What stage *stage*'s review (with its hidden flag) and apply leave behind."""
    _round(plan, mode, manifest=Path(stage.stage_args.cascade_stage_file),
           round_id=round_id, completed=[stage.model], failed={})
    files_from = stage.stage_args.files_from
    scope = Path(files_from).read_text(encoding="utf-8").split() if files_from else None
    _done(plan, mode, review_at, apply_at, scope)


def _review_flags(mode: str, stage_args) -> List[str]:
    records = iw.read_records(mode, stage_args.plan_file, stage_args)
    return iw.step_flags(mode, iw._row(mode).sequence[0], stage_args, records)


def _start(plan: Path, mode: str, *argv: str) -> ic.StagePlan:
    stage = ic.advance(plan, mode, _args(mode, plan, "--cascade", *argv), initial=True)
    assert isinstance(stage, ic.StagePlan), stage
    return stage


# ---------------------------------------------------------------------------
# 1. The stage list and the frozen selectors
# ---------------------------------------------------------------------------

class TestTheStageList:
    @pytest.mark.parametrize("mode", [PLAN, CODE])
    def test_models_as_typed_keep_their_order_and_collapse_duplicates(
            self, repo, archive, mode):
        plan = _plan(repo)
        stage = _start(plan, mode, "--models", M2, M1, M2, M3)
        assert [s["model"] for s in _record(plan, mode)["stages"]] == [M2, M1, M3]
        assert (stage.stage, stage.of, stage.model) == (1, 3, M2)
        assert not any(line.startswith("[AUTO_RESOLVED]") for line in stage.notices)

    def test_quick_resolves_through_the_reviews_own_call(self, repo, archive, monkeypatch):
        plan = _plan(repo)
        seen = {}

        def resolve(**kwargs):
            seen.update(kwargs)
            return [M3, M1]

        monkeypatch.setattr(interactive, "resolve_models", resolve)
        stage = _start(plan, PLAN, "--quick")
        assert seen == dict(cli_models=None, interactive=False, quick=True,
                            mode=REVIEW_PHASE[PLAN], anchor=str(plan.resolve()),
                            unattended=True)
        assert [s["model"] for s in _record(plan, PLAN)["stages"]] == [M3, M1]
        assert not any(line.startswith("[AUTO_RESOLVED]") for line in stage.notices)

    @pytest.mark.parametrize("config, source", [
        ({"defaults": {"models": [M1, M2]}}, "defaults.models"),
        ({"defaults": {"modes": {"review-plan": [M1, M2]}}}, "defaults.modes.review-plan"),
    ])
    def test_defaults_are_recorded_once_as_an_auto_resolution(
            self, repo, archive, monkeypatch, config, source):
        plan = _plan(repo)
        monkeypatch.setattr(interactive, "resolve_models", lambda **kw: [M1, M2])
        monkeypatch.setattr(provider_registry, "load_config", lambda **kw: config)
        stage = _start(plan, PLAN)
        assert f"[AUTO_RESOLVED] models: {M1}, {M2} ({source})" in stage.notices
        # A continuation records nothing again.
        again = ic.advance(plan, PLAN, _args(PLAN, plan), initial=True)
        assert not any(line.startswith("[AUTO_RESOLVED]") for line in again.notices)

    def test_one_model_is_a_one_stage_cascade_with_its_note(self, repo, archive):
        plan = _plan(repo)
        stage = _start(plan, PLAN, "--models", M1)
        assert stage.of == 1
        assert ic.ONE_MODEL_NOTE in stage.notices

    def test_nothing_resolving_is_exit_3(self, repo, archive, monkeypatch):
        plan = _plan(repo)

        def resolve(**kwargs):
            raise ic.NeedsHuman("ERROR: no models could be resolved")

        monkeypatch.setattr(interactive, "resolve_models", resolve)
        with pytest.raises(ic.CascadeRefusal) as excinfo:
            ic.peek(plan, PLAN, _args(PLAN, plan, "--cascade"))
        assert excinfo.value.code == 3
        assert _tree(plan) == {}


class TestFrozenSelectors:
    def test_files_are_written_once_and_every_stage_reviews_them_from_the_file(
            self, repo, archive):
        plan = _plan(repo)
        stage = _start(plan, CODE, "--models", M1, M2, "--files", "src/b.py",
                       "--files", "src/a.py", "--base-ref", "HEAD~1")
        files = stage.cascade_dir / ic.FILES_NAME
        assert files.read_text(encoding="utf-8") == "src/b.py\nsrc/a.py\n"
        record = _record(plan, CODE)
        assert record["selectors"] == {"files_file": "files.txt", "base_ref": "HEAD~1"}
        assert stage.stage_args.files is None
        assert stage.stage_args.files_from == str(files)
        assert os.path.isabs(stage.stage_args.files_from)
        flags = _review_flags(CODE, stage.stage_args)
        assert flags.count("--files-from") == 1 and "--files" not in flags
        assert flags[flags.index("--files-from") + 1] == str(files)

    def test_files_from_a_file_is_frozen_as_its_repository_relative_list(self, repo, archive):
        plan = _plan(repo)
        listing = repo / "list.txt"
        listing.write_text("src/x.py\n\n./src/y.py\n", encoding="utf-8")
        stage = _start(plan, CODE, "--models", M1, "--files-from", str(listing))
        assert (stage.cascade_dir / ic.FILES_NAME).read_text(encoding="utf-8") == (
            "src/x.py\nsrc/y.py\n")

    def test_files_from_stdin_is_refused_with_the_reviews_own_words(self, repo, archive,
                                                                    capsys):
        plan = _plan(repo)
        args = _args(CODE, plan, "--cascade", "--models", M1, "--files-from", "-")
        for call in (lambda: ic.peek(plan, CODE, args),
                     lambda: ic.advance(plan, CODE, args, initial=True)):
            with pytest.raises(ic.CascadeRefusal) as excinfo:
                call()
            assert excinfo.value.code == 1
            assert excinfo.value.message == iw.files_from_stdin_refusal()
        assert iw.preflight(CODE, plan, args, lambda: iw.read_records(CODE, plan, args),
                            skip_own_run_dir=False, detached=False) == 1
        assert iw.files_from_stdin_refusal() in capsys.readouterr().err
        assert _tree(plan) == {}

    def test_a_flag_free_continuation_reproduces_the_selectors(self, repo, archive):
        plan = _plan(repo)
        first = _start(plan, CODE, "--models", M1, M2, "--files", "a.py",
                       "--base-ref", "main")
        _run_stage(plan, CODE, first, round_id=ROUND_1, review_at=1, apply_at=2)
        second = ic.advance(plan, CODE, _args(CODE, plan), initial=True)
        assert isinstance(second, ic.StagePlan) and second.stage == 2
        review = _review_flags(CODE, second.stage_args)
        assert review[review.index("--files-from") + 1] == str(
            second.cascade_dir / ic.FILES_NAME)
        assert review[review.index("--base-ref") + 1] == "main"
        assert review[review.index("--models") + 1] == M2
        records = iw.read_records(CODE, plan, second.stage_args)
        apply = iw.step_flags(CODE, "apply-code-fixes", second.stage_args, records)
        assert apply[apply.index("--base-ref") + 1] == "main"

    def test_improve_plan_keeps_the_plan_at_the_start(self, repo, archive):
        plan = _plan(repo)
        stage = _start(plan, PLAN, "--models", M1)
        assert (stage.cascade_dir / ic.START_NAME).read_bytes() == plan.read_bytes()
        assert _record(plan, PLAN)["selectors"] == {"files_file": None, "base_ref": None}


# ---------------------------------------------------------------------------
# 2. The stage namespace
# ---------------------------------------------------------------------------

class TestTheStageNamespace:
    def test_every_planning_input_is_set_and_nothing_leaks_from_the_invocation(
            self, repo, archive):
        plan = _plan(repo)
        args = _args(CODE, plan, "--cascade", "--force", "--resume", "--models", M1, M2,
                     "--files", "a.py", "--base-ref", "main", "--scope-policy", "preserve")
        stage = ic.advance(plan, CODE, args, initial=True)
        derived = stage.stage_args
        assert derived is not args
        assert (derived.models, derived.quick, derived.resume) == ([M1], False, False)
        assert derived.force is True  # a fresh round, the stage's own decision
        assert derived.scope_policy == "preserve"
        assert derived.files is None
        assert derived.base_ref == "main"
        assert derived.cascade is True
        assert Path(derived.cascade_stage_file).name == provenance.MANIFEST_NAME
        assert os.path.isabs(derived.cascade_stage_file)
        # The user's namespace is untouched.
        assert (args.models, args.files, args.force) == ([M1, M2], ["a.py"], True)

    def test_the_review_gets_the_hidden_flag_and_one_model_and_no_consolidation(
            self, repo, archive):
        plan = _plan(repo)
        stage = _start(plan, PLAN, "--models", M1, M2)
        records = iw.read_records(PLAN, plan, stage.stage_args)
        steps = {s.index: s for s in iw.plan_steps(PLAN, stage.stage_args, records)}
        review = list(steps["1"].flags)
        assert review[-2:] == [provenance.HIDDEN_FLAG, stage.stage_args.cascade_stage_file]
        assert review[review.index("--models") + 1:review.index("--models") + 2] == [M1]
        assert "--force" in review and "--rerun-all" in review and steps["1"].fresh
        assert (steps["1b"].decision, steps["1b"].reason) == (iw.SKIP, "cascade")
        assert provenance.HIDDEN_FLAG not in steps["2"].flags


# ---------------------------------------------------------------------------
# 3. The resume / re-run / --force table
# ---------------------------------------------------------------------------

def _finish_cascade(plan: Path, mode: str, *argv: str) -> ic.Complete:
    """A one-stage cascade run to the end."""
    stage = _start(plan, mode, *argv)
    _run_stage(plan, mode, stage, round_id=ROUND_1, review_at=1, apply_at=2)
    done = ic.advance(plan, mode, _args(mode, plan), initial=False,
                      cascade_id=stage.cascade_id)
    assert isinstance(done, ic.Complete)
    return done


class TestTheResumeTable:
    def test_none_on_record_and_cascade_typed_starts_at_stage_1(self, repo, archive):
        plan = _plan(repo)
        args = _args(PLAN, plan, "--cascade", "--models", M1, M2)
        assert ic.applies(plan, PLAN, args)
        assert ic.peek(plan, PLAN, args).kind == "work"
        stage = ic.advance(plan, PLAN, args, initial=True)
        assert (stage.stage, stage.stage_args.force) == (1, True)
        assert any(line.startswith(f"{ic.CASCADE_MARKER} new cascade") for line in stage.notices)
        pointer = json.loads(ic.pointer_path(plan, PLAN).read_text(encoding="utf-8"))
        assert pointer["cascade_id"] == stage.cascade_id
        assert stage.cascade_dir == ic.composite_dir(plan, PLAN) / pointer["dir"]

    def test_a_finished_cascade_typed_again_is_already_complete_and_writes_nothing(
            self, repo, archive):
        plan = _plan(repo)
        done = _finish_cascade(plan, PLAN, "--models", M1)
        record = _record(plan, PLAN)
        assert record["state"] == ic.FINISHED
        before = _tree(plan)
        args = _args(PLAN, plan, "--cascade", "--models", M1)
        peeked = ic.peek(plan, PLAN, args)
        assert peeked.kind == "complete"
        assert peeked.lines == (
            f"[ALREADY_COMPLETE] cascade finished at {record['finished_at']} (1 stage); "
            "--force --cascade starts a new one",)
        again = ic.advance(plan, PLAN, args, initial=True)
        assert isinstance(again, ic.Complete) and again.cascade_id == done.cascade_id
        assert again.notices == peeked.lines
        assert _tree(plan) == before

    def test_a_finished_cascade_with_another_stage_list_starts_a_new_one(self, repo, archive):
        plan = _plan(repo)
        done = _finish_cascade(plan, PLAN, "--models", M1)
        args = _args(PLAN, plan, "--cascade", "--models", M1, M2)
        assert ic.peek(plan, PLAN, args).kind == "work"
        stage = ic.advance(plan, PLAN, args, initial=True)
        assert stage.cascade_id != done.cascade_id and stage.stage == 1
        old = json.loads((done.cascade_dir / ic.RECORD_NAME).read_text(encoding="utf-8"))
        assert old["state"] == ic.FINISHED

    def test_a_finished_cascade_with_other_selectors_starts_a_new_one(self, repo, archive):
        plan = _plan(repo)
        _finish_cascade(plan, CODE, "--models", M1, "--files", "a.py")
        same = _args(CODE, plan, "--cascade", "--models", M1, "--files", "a.py")
        assert ic.peek(plan, CODE, same).kind == "complete"
        for other in (("--files", "b.py"), ("--base-ref", "main")):
            args = _args(CODE, plan, "--cascade", "--models", M1, *other)
            assert ic.peek(plan, CODE, args).kind == "work", other
        # An untyped selector is the record's, as an untyped model list is.
        untyped = _args(CODE, plan, "--cascade", "--models", M1)
        assert ic.peek(plan, CODE, untyped).kind == "complete"

    @pytest.mark.parametrize("mode, start", [
        (PLAN, ("--models", M1, M2)),
        (CODE, ("--models", M1, M2, "--files", "a.py", "--base-ref", "main")),
    ])
    def test_the_printed_continuation_after_the_finish_is_already_complete(
            self, repo, archive, monkeypatch, mode, start):
        """The continuation every surface prints while a cascade runs is
        ``--cascade --resume`` with no model flag or selector; typed after the
        finish it must not resolve today's default list and start a new
        cascade (the defaults here are another list, M3)."""
        plan = _plan(repo)
        monkeypatch.setattr(interactive, "resolve_models",
                            lambda **kw: list(kw["cli_models"] or [M3]))
        stage = _start(plan, mode, *start)
        while isinstance(stage, ic.StagePlan):
            _run_stage(plan, mode, stage, round_id=f"{stage.stage:016x}",
                       review_at=10 * stage.stage, apply_at=10 * stage.stage + 5)
            stage = ic.advance(plan, mode, _args(mode, plan), initial=False,
                               cascade_id=stage.cascade_id, after_stage=stage.stage)
        assert isinstance(stage, ic.Complete)
        record = _record(plan, mode)
        assert record["state"] == ic.FINISHED
        before = _tree(plan)
        continuation = iw.continuation_flags(mode, _args(mode, plan, "--cascade"))
        assert continuation[-2:] == ["--cascade", "--resume"]
        args = _args(mode, plan, *continuation)
        assert ic.peek(plan, mode, args) == ic.Peek("complete", (ic.already_complete_line(
            ic.load_current(plan, mode)),))
        again = ic.advance(plan, mode, args, initial=True)
        assert isinstance(again, ic.Complete) and again.cascade_id == stage.cascade_id
        assert again.notices[0].startswith("[ALREADY_COMPLETE] cascade finished at ")
        assert _tree(plan) == before, "no new cascade, nothing written"
        assert iw.preflight(mode, plan, args, lambda: iw.read_records(mode, plan, args),
                            skip_own_run_dir=False, detached=False) == iw.CascadeFinished(
            again.notices[0])
        # A typed list that is not the record's still starts a new one.
        other = _args(mode, plan, "--cascade", "--resume", "--models", M3)
        assert ic.peek(plan, mode, other).kind == "work"

    @pytest.mark.parametrize("mode", [PLAN, CODE])
    @pytest.mark.parametrize("gone", ["force", "moved"])
    def test_a_stale_continuation_with_no_cascade_on_record_is_refused(
            self, repo, archive, monkeypatch, mode, gone):
        """``--cascade --resume`` (the printed continuation) after ``--force``
        alone abandoned the cascade, or after its pointer was moved aside, asks
        to go on with a cascade that is not there: it refuses, exit 3, nothing
        written, instead of resolving today's default list (M3 here) and
        starting a new paid cascade."""
        plan = _plan(repo)
        monkeypatch.setattr(interactive, "resolve_models",
                            lambda **kw: list(kw["cli_models"] or [M3]))
        first = _start(plan, mode, "--models", M1, M2)
        if gone == "force":
            assert isinstance(ic.advance(plan, mode, _args(mode, plan, "--force"), initial=True),
                              ic.NotACascade)
        else:
            ic.pointer_path(plan, mode).rename(repo / "pointer.aside")
        assert ic.load_current(plan, mode) is None
        before = _tree(plan)
        continuation = iw.continuation_flags(mode, _args(mode, plan, "--cascade"))
        assert continuation[-2:] == ["--cascade", "--resume"]
        for typed in (continuation, ["--cascade", "--resume", "--scope-policy", "expand"]):
            args = _args(mode, plan, *typed)
            assert ic.applies(plan, mode, args)
            for call in (lambda: ic.peek(plan, mode, args),
                         lambda: ic.advance(plan, mode, args, initial=True)):
                with pytest.raises(ic.CascadeRefusal) as excinfo:
                    call()
                assert excinfo.value.code == 3
                message = excinfo.value.message
                assert message.startswith(
                    f"{ic.CONFLICT_MARKER} --resume: there is no unfinished "
                    f"{iw._row(mode).name} cascade to resume (")
                assert "does not exist); nothing was run." in message
                assert message.endswith("--cascade without --resume, or with --models or "
                                        "--quick, starts one.")
            assert iw.preflight(mode, plan, args, lambda: iw.read_records(mode, plan, args),
                                skip_own_run_dir=False, detached=False) == 3
        assert _tree(plan) == before, "no new cascade, nothing written"
        old = json.loads((first.cascade_dir / ic.RECORD_NAME).read_text(encoding="utf-8"))
        assert old["state"] == (ic.ABANDONED if gone == "force" else ic.RUNNING)
        # Either way out the line names starts one.
        for typed in (("--cascade",), ("--cascade", "--resume", "--models", M1),
                      ("--cascade", "--resume", "--quick")):
            assert ic.peek(plan, mode, _args(mode, plan, *typed)).kind == "work", typed

    def test_a_stale_continuation_over_an_abandoned_pointer_is_refused(self, repo, archive):
        """A pointer still naming an abandoned cascade (an abandon whose
        pointer removal did not happen) counts as no cascade to resume."""
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        current = ic.load_current(plan, PLAN)
        ic._abandon(current)
        before = _tree(plan)
        with pytest.raises(ic.CascadeRefusal) as excinfo:
            ic.peek(plan, PLAN, _args(PLAN, plan, "--cascade", "--resume"))
        assert excinfo.value.code == 3
        assert (f"names cascade {first.cascade_id[:8]}, which was abandoned); nothing was run."
                in excinfo.value.message)
        assert _tree(plan) == before
        # A typed list still starts a new cascade over it.
        stage = ic.advance(plan, PLAN, _args(PLAN, plan, "--cascade", "--resume", "--models", M1),
                           initial=True)
        assert stage.cascade_id != first.cascade_id

    def test_a_finished_cascade_without_cascade_typed_is_todays_composite(
            self, repo, archive):
        plan = _plan(repo)
        _finish_cascade(plan, PLAN, "--models", M1)
        args = _args(PLAN, plan)
        assert not ic.applies(plan, PLAN, args)
        assert ic.peek(plan, PLAN, args) == ic.Peek("none", ())
        assert ic.advance(plan, PLAN, args, initial=True) == ic.NotACascade(())
        # The last stage's review stays completed: a plain run is [ALREADY_COMPLETE].
        assert iw.preflight(PLAN, plan, args, lambda: iw.read_records(PLAN, plan, args),
                            skip_own_run_dir=False, detached=False) is iw.FINISHED

    @pytest.mark.parametrize("typed", [(), ("--cascade",), ("--resume",)],
                             ids=["bare", "cascade", "resume"])
    def test_an_unfinished_cascade_continues_at_its_stage(self, repo, archive, typed):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        args = _args(PLAN, plan, *typed)
        assert ic.applies(plan, PLAN, args)
        line = f"{ic.CASCADE_MARKER} continuing stage 1 of 2 ({M1})"
        assert ic.peek(plan, PLAN, args) == ic.Peek("work", (line,))
        again = ic.advance(plan, PLAN, args, initial=True)
        assert again.cascade_id == first.cascade_id and again.stage == 1
        assert again.notices[0] == line

    def test_force_cascade_abandons_an_unfinished_one_before_the_pointer_moves(
            self, repo, archive):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        args = _args(PLAN, plan, "--force", "--cascade", "--models", M1, M2)
        peeked = ic.peek(plan, PLAN, args)
        assert peeked.kind == "work" and "abandons the unfinished cascade" in peeked.lines[0]
        second = ic.advance(plan, PLAN, args, initial=True)
        assert second.cascade_id != first.cascade_id and second.stage == 1
        old = json.loads((first.cascade_dir / ic.RECORD_NAME).read_text(encoding="utf-8"))
        assert old["state"] == ic.ABANDONED and old["finished_at"]
        assert first.cascade_dir.is_dir(), "an old cascade stays where it is"

    def test_force_cascade_over_a_finished_one_starts_a_new_one(self, repo, archive):
        plan = _plan(repo)
        done = _finish_cascade(plan, PLAN, "--models", M1)
        stage = ic.advance(plan, PLAN, _args(PLAN, plan, "--force", "--cascade",
                                             "--models", M1), initial=True)
        assert stage.cascade_id != done.cascade_id
        old = json.loads((done.cascade_dir / ic.RECORD_NAME).read_text(encoding="utf-8"))
        assert old["state"] == ic.FINISHED

    def test_force_without_cascade_abandons_and_runs_todays_round(self, repo, archive):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        args = _args(PLAN, plan, "--force")
        assert ic.applies(plan, PLAN, args)
        peeked = ic.peek(plan, PLAN, args)
        assert peeked.kind == "none" and peeked.lines[0].startswith("NOTE: --force without")
        outcome = ic.advance(plan, PLAN, args, initial=True)
        assert isinstance(outcome, ic.NotACascade)
        assert outcome.notices == peeked.lines
        assert not ic.pointer_path(plan, PLAN).exists()
        old = json.loads((first.cascade_dir / ic.RECORD_NAME).read_text(encoding="utf-8"))
        assert old["state"] == ic.ABANDONED
        assert not ic.applies(plan, PLAN, _args(PLAN, plan))

    @pytest.mark.parametrize("mode, start, typed, flag", [
        (PLAN, ("--models", M1, M2), ("--models", M2, M1), "--models"),
        (PLAN, ("--models", M1, M2), ("--quick",), "--quick"),
        (CODE, ("--models", M1, "--files", "a.py"), ("--files", "b.py"), "--files"),
        (CODE, ("--models", M1), ("--files", "a.py"), "--files"),
        (CODE, ("--models", M1, "--files", "a.py"), ("--files-from", "LIST"), "--files-from"),
        (CODE, ("--models", M1, "--base-ref", "main"), ("--base-ref", "dev"), "--base-ref"),
        (CODE, ("--models", M1), ("--base-ref", "dev"), "--base-ref"),
    ])
    def test_a_typed_input_that_is_not_the_records_is_a_conflict(
            self, repo, archive, monkeypatch, mode, start, typed, flag):
        plan = _plan(repo)
        listing = repo / "list.txt"
        listing.write_text("c.py\n", encoding="utf-8")
        typed = tuple(str(listing) if token == "LIST" else token for token in typed)
        monkeypatch.setattr(interactive, "resolve_models",
                            lambda **kw: list(kw["cli_models"] or [M3]))
        first = _start(plan, mode, *start)
        before = _tree(plan)
        args = _args(mode, plan, *typed)
        for call in (lambda: ic.peek(plan, mode, args),
                     lambda: ic.advance(plan, mode, args, initial=True)):
            with pytest.raises(ic.CascadeRefusal) as excinfo:
                call()
            message = excinfo.value.message
            assert excinfo.value.code == 3
            assert message.startswith(f"{ic.CONFLICT_MARKER} {flag}: the unfinished cascade "
                                      f"{first.cascade_id[:8]} (stage 1 of {first.of})")
            assert f"Re-run without {flag}" in message and "--force --cascade" in message
        assert _tree(plan) == before

    def test_the_same_inputs_typed_again_are_no_conflict(self, repo, archive):
        plan = _plan(repo)
        _start(plan, CODE, "--models", M1, M2, "--files", "a.py", "--files", "b.py",
               "--base-ref", "main")
        args = _args(CODE, plan, "--models", M1, M2, "--files", "b.py", "--files", "a.py",
                     "--base-ref", "main")
        assert ic.peek(plan, CODE, args).kind == "work"

    def test_a_typed_scope_policy_becomes_the_policy_of_later_stages(self, repo, archive):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        assert _record(plan, PLAN)["stages"][0]["scope_policy"] is None
        args = _args(PLAN, plan, "--scope-policy", "preserve")
        again = ic.advance(plan, PLAN, args, initial=True)
        record = _record(plan, PLAN)
        assert record["scope_policy"] == "preserve"
        assert record["stages"][0]["scope_policy"] is None, "stage 1 started under none"
        assert again.stage_args.scope_policy == "preserve", "today's rules see the typed one"
        assert any("policy for stages 2..2" in line for line in again.notices)
        _run_stage(plan, PLAN, first, round_id=ROUND_1, review_at=1, apply_at=2)
        second = ic.advance(plan, PLAN, _args(PLAN, plan), initial=False,
                            cascade_id=first.cascade_id)
        assert second.stage == 2 and second.stage_args.scope_policy == "preserve"
        assert _record(plan, PLAN)["stages"][1]["scope_policy"] == "preserve"


class TestLaterCalls:
    def test_a_later_call_is_bound_to_its_cascade(self, repo, archive):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1)
        ic.advance(plan, PLAN, _args(PLAN, plan, "--force", "--cascade", "--models", M1),
                   initial=True)
        with pytest.raises(ic.CascadeRefusal) as excinfo:
            ic.advance(plan, PLAN, _args(PLAN, plan), initial=False,
                       cascade_id=first.cascade_id)
        assert excinfo.value.code == 3
        assert excinfo.value.message.startswith(f"{ic.CONFLICT_MARKER} --cascade-id")

    def test_a_later_call_never_restarts_and_ignores_the_typed_flags(self, repo, archive):
        plan = _plan(repo)
        args = _args(PLAN, plan, "--force", "--cascade", "--models", M1, M2)
        first = ic.advance(plan, PLAN, args, initial=True)
        _run_stage(plan, PLAN, first, round_id=ROUND_1, review_at=1, apply_at=2)
        second = ic.advance(plan, PLAN, args, initial=False, cascade_id=first.cascade_id)
        assert second.cascade_id == first.cascade_id and second.stage == 2
        assert second.stage_args.models == [M2]

    def test_a_later_call_on_a_finished_cascade_is_complete(self, repo, archive):
        plan = _plan(repo)
        done = _finish_cascade(plan, PLAN, "--models", M1)
        again = ic.advance(plan, PLAN, _args(PLAN, plan), initial=False,
                           cascade_id=done.cascade_id)
        assert isinstance(again, ic.Complete) and again.summary_path.is_file()


#: Two planners started together, each in its own process: they meet at a
#: barrier in their first, read-only decision (before the run dir's lock),
#: then advance. ``refuse``: the planner that wins the lock holds it past the
#: other's retries; ``continue``: the other waits it out.
_PLANNER = r"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from utils import improve_cascade as ic, improve_workflow as iw, interactive
interactive.resolve_models = lambda **kw: list(kw["cli_models"])
plan, me, other, variant = sys.argv[2:6]
decide, first = ic._decide, [True]
def met(*a, **kw):
    if first[0]:
        first[0] = False
        Path(me).touch()
        deadline = time.monotonic() + 30
        while not Path(other).exists():
            if time.monotonic() > deadline:
                raise SystemExit("the other planner never came")
            time.sleep(0.002)
    return decide(*a, **kw)
ic._decide = met
if variant == "refuse":
    create = ic._create
    def slow(*a, **kw):
        time.sleep(0.6)
        return create(*a, **kw)
    ic._create = slow
else:
    iw.STEP_LOCK_ATTEMPTS = 2400
args = iw.parse_args("improve-plan", ["--plan-file", plan, "--cascade", "--models",
                                      "codex:gpt-a", "codex:gpt-b"])
try:
    out = ic.advance(plan, "improve-plan", args, initial=True)
    print(json.dumps({"kind": type(out).__name__, "cascade_id": out.cascade_id,
                      "stage": out.stage, "notices": list(out.notices)}))
except ic.CascadeRefusal as exc:
    print(json.dumps({"kind": "refused", "code": exc.code, "message": exc.message}))
"""


class TestOnePlannerAtATime:
    """``advance`` reads, decides and writes under the composite run dir's lock."""

    def _race(self, tmp_path: Path, variant: str) -> List[Dict[str, Any]]:
        import subprocess

        plan = _plan(tmp_path)
        script = tmp_path / "planner.py"
        script.write_text(_PLANNER, encoding="utf-8")
        marks = [tmp_path / "a.ready", tmp_path / "b.ready"]
        procs = [
            subprocess.Popen(
                [sys.executable, str(script), str(Path(ic.__file__).parents[1]), str(plan),
                 str(me), str(other), variant],
                cwd=str(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for me, other in (marks, marks[::-1])
        ]
        results = []
        for proc in procs:
            out, err = proc.communicate(timeout=120)
            assert proc.returncode == 0, err
            results.append(json.loads(out.strip().splitlines()[-1]))
        return results

    def _cascades(self, plan: Path) -> List[str]:
        root = ic.composite_dir(plan, PLAN) / ic.CASCADES_DIR
        return sorted(p.name for p in root.iterdir())

    def test_two_planners_started_together_hand_out_one_cascade_and_one_refusal(
            self, tmp_path):
        plan = _plan(tmp_path)
        results = self._race(tmp_path, "refuse")
        handed = [r for r in results if r["kind"] == "StagePlan"]
        refused = [r for r in results if r["kind"] == "refused"]
        assert len(handed) == 1 and len(refused) == 1, results
        assert refused[0]["code"] == 3
        assert "a live run owns" in refused[0]["message"]
        assert "cascade planner" in refused[0]["message"]
        assert len(self._cascades(plan)) == 1, "never two fresh stage-1 handoffs"
        assert ic.load_current(plan, PLAN).cascade_id == handed[0]["cascade_id"]

    def test_the_planner_that_waits_continues_the_cascade_the_other_started(
            self, tmp_path):
        plan = _plan(tmp_path)
        results = self._race(tmp_path, "continue")
        assert [r["kind"] for r in results] == ["StagePlan", "StagePlan"], results
        assert results[0]["cascade_id"] == results[1]["cascade_id"]
        assert len(self._cascades(plan)) == 1, "never two fresh stage-1 handoffs"
        news = [r for r in results
                if any(n.startswith(f"{ic.CASCADE_MARKER} new cascade") for n in r["notices"])]
        assert len(news) == 1, results
        other = next(r for r in results if r is not news[0])
        assert f"{ic.CASCADE_MARKER} continuing stage 1 of 2 (codex:gpt-a)" in other["notices"]

    def test_a_run_dir_another_process_holds_refuses_at_exit_3(self, repo, archive):
        import subprocess

        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        holder = subprocess.Popen(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, sys.argv[1]);"
             "from utils import process_control as pc;"
             "lock = pc.PhaseLock(sys.argv[2]); lock.acquire({'holder': 'another run'});"
             "print('held', flush=True); sys.stdin.read()",
             str(Path(ic.__file__).parents[1]),
             str(ic.composite_dir(plan, PLAN) / ".lock")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            assert holder.stdout.readline().strip() == "held"
            before = _tree(plan)
            with pytest.raises(ic.CascadeRefusal) as excinfo:
                ic.advance(plan, PLAN, _args(PLAN, plan), initial=False,
                           cascade_id=first.cascade_id)
            assert excinfo.value.code == 3
            assert "a live run owns" in excinfo.value.message
            assert "another run" in excinfo.value.message
            with pytest.raises(ic.CascadeRefusal):
                ic.advance(plan, PLAN, _args(PLAN, plan, "--cascade"), initial=True)
            assert _tree(plan) == before
        finally:
            holder.stdin.close()
            holder.wait(timeout=30)

    def test_a_lock_this_process_holds_is_used_and_left_held(self, repo, archive):
        """The detached worker holds its run dir's lock for the whole run;
        its own planner calls go on under it, and it stays the worker's."""
        from utils import detach, process_control

        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        path = detach.lock_path(str(ic.composite_dir(plan, PLAN)))
        lock = process_control.PhaseLock(path)
        lock.acquire({"holder": "the worker"})
        try:
            _run_stage(plan, PLAN, first, round_id=ROUND_1, review_at=1, apply_at=2)
            second = ic.advance(plan, PLAN, _args(PLAN, plan), initial=False,
                                cascade_id=first.cascade_id, after_stage=1)
            assert isinstance(second, ic.StagePlan) and second.stage == 2
            assert archive.close_outs and archive.close_outs[-1]["stage"] == 1
            assert lock.held and process_control.held_phase_lock(path) is lock
        finally:
            lock.release()

    def test_a_lock_it_takes_is_released_and_registered_while_held(self, repo, archive,
                                                                   monkeypatch):
        """The close-out inside ``advance`` sees the run dir's lock as this
        process's own (``improve_workflow.holding``), never as another run."""
        from utils import detach, process_control

        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        path = detach.lock_path(str(ic.composite_dir(plan, PLAN)))
        seen = []
        close_out = archive.close_out

        def watching(**kwargs):
            seen.append((iw._held_lock(ic.composite_dir(plan, PLAN)) is not None,
                         process_control.held_phase_lock(path) is not None))
            return close_out(**kwargs)

        monkeypatch.setattr(archive, "close_out", watching)
        _run_stage(plan, PLAN, first, round_id=ROUND_1, review_at=1, apply_at=2)
        ic.advance(plan, PLAN, _args(PLAN, plan), initial=False,
                   cascade_id=first.cascade_id, after_stage=1)
        assert seen == [(True, True)]
        assert process_control.held_phase_lock(path) is None
        assert process_control.lock_is_free(path)

    def test_an_answer_that_writes_nothing_creates_no_run_dir(self, repo, archive):
        plan = _plan(repo)
        assert isinstance(ic.advance(plan, PLAN, _args(PLAN, plan), initial=True),
                          ic.NotACascade)
        assert not ic.composite_dir(plan, PLAN).exists()


class TestTheReviewTakesOnlyTheCurrentHandoff:
    """The stage's review refuses a handoff the composite's pointer no longer names."""

    def _check(self, plan: Path, stage: ic.StagePlan) -> None:
        """The review's pre-flight, given the stage's flags (a fresh round:
        ``step_flags`` hands a stage's review ``--force --rerun-all``)."""
        from utils import review_lane

        args = types.SimpleNamespace(cascade_stage_file=stage.stage_args.cascade_stage_file,
                                     rerun_all=True, reaggregate=False)
        review_lane.check_cascade_stage(args, phase_dir=str(_review_dir(plan, PLAN)))

    def test_a_superseded_cascades_stage_is_refused_writing_nothing(
            self, repo, archive, capsys):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        second = ic.advance(plan, PLAN,
                            _args(PLAN, plan, "--force", "--cascade", "--models", M1, M2),
                            initial=True)
        assert second.cascade_id != first.cascade_id
        before = _tree(plan)
        with pytest.raises(SystemExit) as stop:
            self._check(plan, first)
        assert stop.value.code == 3
        err = capsys.readouterr().err
        assert "ERROR: cascade stage handoff refused" in err and "superseded" in err
        assert _tree(plan) == before
        assert not _review_dir(plan, PLAN).exists()
        self._check(plan, second)

    def test_an_abandoned_cascades_stage_is_refused(self, repo, archive, capsys):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        ic.advance(plan, PLAN, _args(PLAN, plan, "--force"), initial=True)
        with pytest.raises(SystemExit) as stop:
            self._check(plan, first)
        assert stop.value.code == 3
        assert ic.POINTER_NAME in capsys.readouterr().err

    def test_a_closed_stages_handoff_is_refused(self, repo, archive, capsys):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        _run_stage(plan, PLAN, first, round_id=ROUND_1, review_at=1, apply_at=2)
        second = ic.advance(plan, PLAN, _args(PLAN, plan), initial=False,
                            cascade_id=first.cascade_id, after_stage=1)
        assert second.stage == 2
        with pytest.raises(SystemExit) as stop:
            self._check(plan, first)
        assert stop.value.code == 3
        assert "stage 1 of cascade" in capsys.readouterr().err
        self._check(plan, second)


class TestAfterStage:
    """A continue call names the stage it follows; it never hands that stage out again."""

    def _continue(self, plan, stage, mode=PLAN):
        return ic.advance(plan, mode, _args(mode, plan), initial=False,
                          cascade_id=stage.cascade_id, after_stage=stage.stage)

    def _refused(self, plan, stage, mode=PLAN) -> ic.StageNotFinished:
        with pytest.raises(ic.StageNotFinished) as excinfo:
            self._continue(plan, stage, mode)
        return excinfo.value

    def test_a_review_that_stopped_with_its_round_open_is_not_handed_out_again(
            self, repo, archive):
        """A review stop other than no result (a validation decision owed, a
        provenance refusal after the round opened, an interruption): the
        round carries the stage's block, the stage is not done."""
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        _round(plan, PLAN, manifest=Path(first.stage_args.cascade_stage_file),
               round_id=ROUND_1, completed=[M1], failed={})
        for _attempt in range(2):
            exc = self._refused(plan, first)
            assert (exc.code, exc.cascade_id, exc.stage, exc.of, exc.model) == (
                3, first.cascade_id, 1, 2, M1)
            assert exc.message.startswith(
                f"ERROR: cascade {first.cascade_id[:8]}: stage 1 of 2 ({M1}) did not finish: ")
            assert "run the command again to continue stage 1" in exc.message
        stage = _record(plan, PLAN)["stages"][0]
        assert (stage["state"], stage["bound_round"]) == (ic.STAGE_RUNNING, ROUND_1)
        assert archive.close_outs == []
        # The user's own command (an initial call) picks the stage up.
        again = ic.advance(plan, PLAN, _args(PLAN, plan), initial=True)
        assert (again.stage, again.stage_args.force) == (1, False)

    def test_a_continue_call_before_the_stage_ran_is_refused(self, repo, archive):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        before = _tree(plan)
        exc = self._refused(plan, first)
        assert "no review round of this stage is on record" in exc.message
        assert _tree(plan) == before, "a refused continue call writes nothing"

    def test_a_foreign_round_after_binding_is_refused_and_leaves_the_binding(
            self, repo, archive):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        _round(plan, PLAN, manifest=Path(first.stage_args.cascade_stage_file),
               round_id=ROUND_1)
        ic.advance(plan, PLAN, _args(PLAN, plan), initial=True)
        _round(plan, PLAN, manifest=None, round_id=ROUND_2, completed=[M1], failed={})
        exc = self._refused(plan, first)
        assert "no longer this stage's" in exc.message
        assert _record(plan, PLAN)["stages"][0]["bound_round"] == ROUND_1
        again = ic.advance(plan, PLAN, _args(PLAN, plan), initial=True)
        assert ("NOTE: the review on record is not this cascade's stage 1; starting "
                "stage 1 over") in again.notices

    def test_a_done_stage_closes_and_the_next_is_handed_out(self, repo, archive):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        _run_stage(plan, PLAN, first, round_id=ROUND_1, review_at=1, apply_at=2)
        second = self._continue(plan, first)
        assert second.stage == 2 and len(archive.close_outs) == 1

    def test_a_zero_result_middle_stage_is_still_skipped(self, repo, archive):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        _round(plan, PLAN, manifest=Path(first.stage_args.cascade_stage_file),
               round_id=ROUND_1, completed=[], failed={M1: "rate limited"})
        second = self._continue(plan, first)
        assert (second.stage, second.stage_args.force) == (2, True)
        assert f"{ic.SKIPPED_MARKER} 1 {M1}: rate limited" in second.notices

    def test_a_zero_result_last_stage_is_still_the_last_stage_stop(self, repo, archive):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1)
        _round(plan, PLAN, manifest=Path(first.stage_args.cascade_stage_file),
               round_id=ROUND_1, completed=[], failed={M1: "quota"})
        with pytest.raises(ic.LastStageStopped):
            self._continue(plan, first)

    def test_without_after_stage_a_continue_call_checks_nothing(self, repo, archive):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        again = ic.advance(plan, PLAN, _args(PLAN, plan), initial=False,
                           cascade_id=first.cascade_id)
        assert again.stage == 1


# ---------------------------------------------------------------------------
# 4. Binding rounds to stages
# ---------------------------------------------------------------------------

class TestBinding:
    def test_the_stages_own_round_with_results_is_bound_and_resumed_not_forced(
            self, repo, archive):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        _round(plan, PLAN, manifest=Path(first.stage_args.cascade_stage_file),
               round_id=ROUND_1, completed=[M1], failed={})
        _done(plan, PLAN, 1, None)  # reviewed, not applied
        again = ic.advance(plan, PLAN, _args(PLAN, plan), initial=True)
        assert again.stage == 1 and again.stage_args.force is False
        stage = _record(plan, PLAN)["stages"][0]
        assert (stage["bound_round"], stage["state"]) == (ROUND_1, ic.STAGE_RUNNING)
        records = iw.read_records(PLAN, plan, again.stage_args)
        assert iw.skip_reason(PLAN, "review-plan", again.stage_args, records) == "completed"
        assert iw.skip_reason(PLAN, "apply-suggestions", again.stage_args, records) is None

    @pytest.mark.parametrize("hidden_flag", [True, False], ids=["flag-given", "carried-over"])
    def test_a_second_fan_out_after_binding_stays_bound(self, repo, archive, hidden_flag):
        """A ``--force`` resume of the missing model stamps a new round id; its
        block is the flag's or the one ``open_round`` carried over: the same
        stage's either way."""
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        manifest = Path(first.stage_args.cascade_stage_file)
        _round(plan, PLAN, manifest=manifest, round_id=ROUND_1)
        ic.advance(plan, PLAN, _args(PLAN, plan), initial=True)
        assert _record(plan, PLAN)["stages"][0]["bound_round"] == ROUND_1
        block = provenance.read_block(str(_review_dir(plan, PLAN)))
        carried = provenance.next_block(
            block, provenance.block_from_manifest(manifest) if hidden_flag else None,
            rerun_all=False, models=[M1])
        status_file.write_status(str(_review_dir(plan, PLAN)), {
            "round_id": ROUND_2, "models_completed": [M1], "models_failed": {},
            provenance.STATUS_KEY: carried})
        again = ic.advance(plan, PLAN, _args(PLAN, plan), initial=True)
        assert again.stage == 1 and again.stage_args.force is False
        assert _record(plan, PLAN)["stages"][0]["bound_round"] == ROUND_2
        assert not any(line.startswith("NOTE: the review on record") for line in again.notices)

    def test_an_interruption_before_the_first_fan_out_is_a_fresh_start(self, repo, archive):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        _run_stage(plan, PLAN, first, round_id=ROUND_1, review_at=1, apply_at=2)
        second = ic.advance(plan, PLAN, _args(PLAN, plan), initial=False,
                            cascade_id=first.cascade_id)
        assert second.stage == 2
        # Stage 2 never stamped a round: stage 1's block is still on record.
        again = ic.advance(plan, PLAN, _args(PLAN, plan), initial=True)
        assert (again.stage, again.stage_args.force) == (2, True)
        assert _record(plan, PLAN)["stages"][1]["bound_round"] is None
        assert not any(line.startswith("NOTE: the review on record") for line in again.notices)

    def test_another_cascades_round_is_not_bound(self, repo, archive, tmp_path):
        plan = _plan(repo)
        _start(plan, PLAN, "--models", M1, M2)
        foreign = _freeze_inputs(cascade_dir=tmp_path / "other", cascade_id="f" * 32,
                                 mode=PLAN, stage=1, of=2, model=M1, plan_path=plan)
        _round(plan, PLAN, manifest=foreign, round_id=ROUND_1, completed=[M1], failed={})
        again = ic.advance(plan, PLAN, _args(PLAN, plan), initial=True)
        assert (again.stage, again.stage_args.force) == (1, True)
        assert _record(plan, PLAN)["stages"][0]["bound_round"] is None

    def test_a_foreign_rerun_all_with_the_same_model_is_not_bound(self, repo, archive):
        """A hand-started ``--review-plan --force --rerun-all --models <stage model>``
        clears the block, so it is never read as the stage's round, bound or not."""
        plan = _plan(repo)
        _start(plan, PLAN, "--models", M1, M2)
        _round(plan, PLAN, manifest=None, round_id=ROUND_1, completed=[M1], failed={})
        _done(plan, PLAN, 1, 2)
        again = ic.advance(plan, PLAN, _args(PLAN, plan), initial=True)
        assert (again.stage, again.stage_args.force) == (1, True)
        assert _record(plan, PLAN)["stages"][0]["bound_round"] is None
        assert archive.close_outs == []

    def test_a_foreign_round_after_binding_starts_the_stage_over(self, repo, archive):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        _round(plan, PLAN, manifest=Path(first.stage_args.cascade_stage_file),
               round_id=ROUND_1)
        ic.advance(plan, PLAN, _args(PLAN, plan), initial=True)
        _round(plan, PLAN, manifest=None, round_id=ROUND_2, completed=[M1], failed={})
        again = ic.advance(plan, PLAN, _args(PLAN, plan), initial=True)
        assert (again.stage, again.stage_args.force) == (1, True)
        assert ("NOTE: the review on record is not this cascade's stage 1; starting "
                "stage 1 over") in again.notices
        assert _record(plan, PLAN)["stages"][0]["bound_round"] is None

    def test_a_malformed_block_is_exit_3(self, repo, archive):
        plan = _plan(repo)
        _start(plan, PLAN, "--models", M1)
        _review_dir(plan, PLAN).mkdir(parents=True)
        status_file.write_status(str(_review_dir(plan, PLAN)),
                                 {"round_id": ROUND_1, provenance.STATUS_KEY: {"stage": 1}})
        with pytest.raises(ic.CascadeRefusal) as excinfo:
            ic.advance(plan, PLAN, _args(PLAN, plan), initial=True)
        assert excinfo.value.code == 3 and "cascade stage input" in excinfo.value.message

    @pytest.mark.parametrize("damage", [b"{not json", b"[1, 2]"], ids=["not-json", "not-object"])
    def test_an_unreadable_review_checkpoint_is_exit_3_naming_it(self, repo, archive, damage):
        """Read as "no block", a bound stage would take the foreign-round
        path and start over, discarding the review it completed."""
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        _round(plan, PLAN, manifest=Path(first.stage_args.cascade_stage_file),
               round_id=ROUND_1, completed=[M1], failed={})
        ic.advance(plan, PLAN, _args(PLAN, plan), initial=True)
        assert _record(plan, PLAN)["stages"][0]["bound_round"] == ROUND_1
        path = Path(status_file.status_path(str(_review_dir(plan, PLAN))))
        path.write_bytes(damage)
        before = _tree(plan)
        for call in (lambda: ic.advance(plan, PLAN, _args(PLAN, plan), initial=True),
                     lambda: ic.advance(plan, PLAN, _args(PLAN, plan), initial=False,
                                        cascade_id=first.cascade_id)):
            with pytest.raises(ic.CascadeRefusal) as excinfo:
                call()
            assert excinfo.value.code == 3
            assert str(path) in excinfo.value.message or ".status.json" in excinfo.value.message
            assert "cannot read" in excinfo.value.message
        assert _tree(plan) == before
        assert _record(plan, PLAN)["stages"][0]["bound_round"] == ROUND_1
        with pytest.raises(ic.CascadeRefusal):
            ic.zero_result_stage(plan, PLAN)


# ---------------------------------------------------------------------------
# 5. Zero-result stages
# ---------------------------------------------------------------------------

class TestZeroResultStages:
    def _failed(self, plan, stage, reason="rate limited"):
        _round(plan, PLAN, manifest=Path(stage.stage_args.cascade_stage_file),
               round_id=ROUND_1, completed=[], failed={stage.model: reason})

    def test_a_middle_stage_is_skipped_and_the_next_runs(self, repo, archive):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        self._failed(plan, first)
        assert ic.zero_result_stage(plan, PLAN)
        second = ic.advance(plan, PLAN, _args(PLAN, plan), initial=False,
                            cascade_id=first.cascade_id)
        assert (second.stage, second.stage_args.force) == (2, True)
        assert f"{ic.SKIPPED_MARKER} 1 {M1}: rate limited" in second.notices
        assert any(line.startswith("[AUTO_RESOLVED] partial-results-recovery: cascade "
                                   "stage 1 of 2") for line in second.notices)
        stage = _record(plan, PLAN)["stages"][0]
        assert (stage["state"], stage["skip_reason"]) == (ic.SKIPPED, "rate limited")
        assert archive.close_outs == [] and archive.summaries
        assert not ic.zero_result_stage(plan, PLAN)

    def test_the_last_stage_stops_and_its_rerun_starts_it_fresh(self, repo, archive):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1)
        self._failed(plan, first, "quota")
        with pytest.raises(ic.CascadeRefusal) as excinfo:
            ic.advance(plan, PLAN, _args(PLAN, plan), initial=False,
                       cascade_id=first.cascade_id)
        assert excinfo.value.code == 3
        assert "stage 1 of 1" in excinfo.value.message and "quota" in excinfo.value.message
        assert _record(plan, PLAN)["state"] == ic.RUNNING
        again = ic.advance(plan, PLAN, _args(PLAN, plan), initial=True)
        assert (again.stage, again.stage_args.force) == (1, True)

    def test_after_a_successful_last_stage_a_plain_run_is_already_complete(
            self, repo, archive):
        plan = _plan(repo)
        _finish_cascade(plan, PLAN, "--models", M1)
        args = _args(PLAN, plan)
        assert iw.preflight(PLAN, plan, args, lambda: iw.read_records(PLAN, plan, args),
                            skip_own_run_dir=False, detached=False) is iw.FINISHED


# ---------------------------------------------------------------------------
# 6. Close-out
# ---------------------------------------------------------------------------

@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_a_review_dir_swapped_for_a_link_stops_the_real_close_out(repo):
    """The real archive: stage 1's review dir replaced by a link to a folder
    outside the plan's output dir is refused at exit 3 before any lock (no
    ``.lock`` created at the link's target) and before any copy."""
    plan = _plan(repo)
    first = _start(plan, PLAN, "--models", M1, M2)
    _run_stage(plan, PLAN, first, round_id=ROUND_1, review_at=1, apply_at=2)
    review = _review_dir(plan, PLAN)
    (review / "grouped.json").write_text('{"groups": []}', encoding="utf-8")
    phase_dir_path(plan, APPLY_PHASE[PLAN]).mkdir(parents=True, exist_ok=True)
    outside = repo / "external-review"
    review.rename(outside)
    (outside / "private.txt").write_text("PRIVATE-SIBLING-DATA", encoding="utf-8")
    os.symlink(outside, review, target_is_directory=True)
    before = sorted(p.name for p in outside.iterdir())

    with pytest.raises(ic.CascadeRefusal) as excinfo:
        ic.advance(plan, PLAN, _args(PLAN, plan), initial=False,
                   cascade_id=first.cascade_id, after_stage=1)
    assert excinfo.value.code == 3
    assert f"{review} is a link" in excinfo.value.message
    assert sorted(p.name for p in outside.iterdir()) == before, "no .lock written outside"
    assert ".lock" not in before
    assert not list(first.cascade_dir.rglob("private.txt")), "nothing outside was archived"
    assert _record(plan, PLAN)["stages"][0]["state"] == ic.STAGE_RUNNING


class TestCloseOut:
    def test_a_done_stage_is_archived_with_the_contracts_arguments(self, repo, archive):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        _run_stage(plan, PLAN, first, round_id=ROUND_1, review_at=1, apply_at=2)
        second = ic.advance(plan, PLAN, _args(PLAN, plan), initial=False,
                            cascade_id=first.cascade_id)
        assert second.stage == 2
        (call,) = archive.close_outs
        assert call == dict(plan_path=plan.resolve(), mode=PLAN, cascade_dir=first.cascade_dir,
                            cascade_id=first.cascade_id, stage=1, model=M1,
                            round_id=ROUND_1, recheck=True)
        stage = _record(plan, PLAN)["stages"][0]
        assert stage["state"] == ic.DONE and stage["counts"] == _counts(findings=1)
        assert stage["finished_at"]
        assert archive.summaries[-1]["stages"][0]["state"] == ic.DONE

    def test_a_continuation_that_closes_the_last_stage_keeps_its_notices(
            self, repo, archive):
        plan = _plan(repo)
        stage = _start(plan, PLAN, "--models", M1)
        _run_stage(plan, PLAN, stage, round_id=ROUND_1, review_at=1, apply_at=2)
        done = ic.advance(plan, PLAN, _args(PLAN, plan), initial=True)
        assert isinstance(done, ic.Complete)
        assert done.notices == (f"{ic.CASCADE_MARKER} continuing stage 1 of 1 ({M1})",)
        assert _record(plan, PLAN)["state"] == ic.FINISHED

    def test_the_last_close_out_finishes_the_cascade(self, repo, archive):
        plan = _plan(repo)
        done = _finish_cascade(plan, PLAN, "--models", M1)
        record = _record(plan, PLAN)
        assert record["state"] == ic.FINISHED and record["finished_at"]
        assert done.summary_path == done.cascade_dir / ic.SUMMARY_NAME
        assert done.summary_path.is_file()

    def test_a_stage_interrupted_before_its_close_out_is_closed_by_the_next_run(
            self, repo, archive):
        """The apply completed, the close-out did not: the plain rules read the
        files as finished, the cascade does not."""
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        _run_stage(plan, PLAN, first, round_id=ROUND_1, review_at=1, apply_at=2)
        args = _args(PLAN, plan)
        assert iw.nothing_to_do(PLAN, args, iw.read_records(PLAN, plan, args))
        before = _tree(plan)
        assert iw.preflight(PLAN, plan, args, lambda: iw.read_records(PLAN, plan, args),
                            skip_own_run_dir=False, detached=False) is None
        assert _tree(plan) == before, "preflight and peek write nothing"
        again = ic.advance(plan, PLAN, args, initial=True)
        assert again.stage == 2 and len(archive.close_outs) == 1

    def test_published_artifacts_close_the_stage_without_a_second_copy(self, repo, archive):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        archive.published[(str(first.cascade_dir), 1, first.cascade_id)] = _counts(applied=9)
        # The live dirs no longer show the stage (a foreign round replaced it).
        _round(plan, PLAN, manifest=None, round_id=ROUND_2)
        second = ic.advance(plan, PLAN, _args(PLAN, plan), initial=True)
        assert second.stage == 2 and archive.close_outs == []
        assert _record(plan, PLAN)["stages"][0]["counts"]["applied"] == 9

    def test_a_raced_close_out_is_planned_again(self, repo, archive):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        _run_stage(plan, PLAN, first, round_id=ROUND_1, review_at=1, apply_at=2)
        archive.race = 1
        second = ic.advance(plan, PLAN, _args(PLAN, plan), initial=True)
        assert second.stage == 2 and len(archive.close_outs) == 2

    def test_a_close_out_that_keeps_racing_stops(self, repo, archive):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        _run_stage(plan, PLAN, first, round_id=ROUND_1, review_at=1, apply_at=2)
        archive.race = ic.CLOSE_OUT_ATTEMPTS
        with pytest.raises(ic.CascadeRefusal) as excinfo:
            ic.advance(plan, PLAN, _args(PLAN, plan), initial=True)
        assert excinfo.value.code == 3
        assert _record(plan, PLAN)["stages"][0]["state"] == ic.STAGE_RUNNING

    def test_an_archive_refusal_is_exit_3_with_its_message(self, repo, archive):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1)
        _run_stage(plan, PLAN, first, round_id=ROUND_1, review_at=1, apply_at=2)
        archive.refuse = "ERROR: stage-1/artifacts/ has a .complete.json of another cascade"
        with pytest.raises(ic.CascadeRefusal) as excinfo:
            ic.advance(plan, PLAN, _args(PLAN, plan), initial=True)
        assert (excinfo.value.code, excinfo.value.message) == (3, archive.refuse)

    @pytest.mark.parametrize("mark", [None, b"{not json", "other-cascade", "other-stage"])
    def test_a_published_archive_that_does_not_check_out_stops_before_any_binding(
            self, repo, archive, mark):
        """Seen first at the top of the walk: no binding, no hand-out, no
        close-out, so a foreign round on record never sends the stage's paid
        review round again before the close-out would refuse."""
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        artifacts = first.cascade_dir / "stage-1" / "artifacts"
        artifacts.mkdir(parents=True)
        complete = artifacts / real_ledger.COMPLETE_NAME
        if isinstance(mark, bytes):
            complete.write_bytes(mark)
        elif mark is not None:
            complete.write_text(json.dumps({
                "schema_version": real_ledger.COMPLETE_SCHEMA_VERSION,
                "cascade_id": "f" * 32 if mark == "other-cascade" else first.cascade_id,
                "stage": 2 if mark == "other-stage" else 1,
                "mode": PLAN, "model": M1, "files": {"x": "0" * 64},
            }), encoding="utf-8")
        # A foreign round is on record: without the early stop the stage
        # would be handed out fresh.
        _round(plan, PLAN, manifest=None, round_id=ROUND_2, completed=[M1], failed={})
        before = _tree(plan)
        for call in (lambda: ic.advance(plan, PLAN, _args(PLAN, plan), initial=True),
                     lambda: ic.advance(plan, PLAN, _args(PLAN, plan), initial=False,
                                        cascade_id=first.cascade_id, after_stage=1)):
            with pytest.raises(ic.CascadeRefusal) as excinfo:
                call()
            assert not isinstance(excinfo.value, ic.StageNotFinished)
            assert excinfo.value.code == 3
            assert str(artifacts) in excinfo.value.message
        assert archive.close_outs == []
        assert _tree(plan) == before

    def test_recheck_is_false_once_the_round_moved(self, repo, archive):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1, M2)
        _run_stage(plan, PLAN, first, round_id=ROUND_1, review_at=1, apply_at=2)
        seen = []
        original = archive.close_out

        def close_out(**kwargs):
            _round(plan, PLAN, manifest=None, round_id=ROUND_2)  # a standalone review
            seen.append(kwargs["recheck"]())
            raise archive.CloseOutRaced()

        archive.close_out = close_out
        again = ic.advance(plan, PLAN, _args(PLAN, plan), initial=True)
        archive.close_out = original
        assert seen == [False]
        assert again.stage == 1 and again.stage_args.force is True


# ---------------------------------------------------------------------------
# 7. The preflight split
# ---------------------------------------------------------------------------

def _preflight(mode, plan, args, **kw):
    return iw.preflight(mode, plan, args, lambda: iw.read_records(mode, plan, args),
                        skip_own_run_dir=False, detached=kw.get("detached", False))


class TestThePreflightSplit:
    @pytest.mark.parametrize("mode", [PLAN, CODE])
    def test_a_new_cascade_over_completed_phases_goes_on_and_writes_nothing(
            self, repo, archive, mode):
        plan = _plan(repo)
        _done(plan, mode, 1, 2)
        before = _tree(plan)
        plain = _args(mode, plan, "--models", M1)
        assert _preflight(mode, plan, plain) is iw.FINISHED
        cascade = _args(mode, plan, "--cascade", "--models", M1)
        assert _preflight(mode, plan, cascade) is None
        assert _tree(plan) == before
        stage = ic.advance(plan, mode, cascade, initial=True)
        assert (stage.stage, stage.stage_args.force) == (1, True)

    def test_a_finished_cascade_typed_again_is_cascade_finished(self, repo, archive):
        plan = _plan(repo)
        _finish_cascade(plan, PLAN, "--models", M1)
        before = _tree(plan)
        outcome = _preflight(PLAN, plan, _args(PLAN, plan, "--cascade", "--models", M1))
        assert isinstance(outcome, iw.CascadeFinished)
        assert outcome.code == 2
        assert outcome.line.startswith("[ALREADY_COMPLETE] cascade finished at ")
        assert _tree(plan) == before

    def test_a_conflict_is_printed_and_exit_3(self, repo, archive, capsys):
        plan = _plan(repo)
        _start(plan, PLAN, "--models", M1, M2)
        args = _args(PLAN, plan, "--models", M2)
        assert _preflight(PLAN, plan, args) == 3
        assert capsys.readouterr().err.startswith(f"{ic.CONFLICT_MARKER} --models")

    def test_improve_commits_never_reads_a_cascade(self, repo, monkeypatch):
        monkeypatch.setattr(ic, "load_current", lambda *a: pytest.fail("read"))
        assert not ic.applies(repo / "b.md", "improve-commits", object())
        assert ic.cascade_status(repo / "b.md", "improve-commits") is None


# ---------------------------------------------------------------------------
# 8. The pointer and the record
# ---------------------------------------------------------------------------

class TestThePointerAndTheRecord:
    def test_an_orphan_directory_is_read_by_nobody(self, repo, archive):
        plan = _plan(repo)
        orphan = ic.composite_dir(plan, PLAN) / ic.CASCADES_DIR / "20261113T000000-deadbeef"
        orphan.mkdir(parents=True)
        (orphan / ic.RECORD_NAME).write_text("{}", encoding="utf-8")
        assert ic.load_current(plan, PLAN) is None
        assert not ic.applies(plan, PLAN, _args(PLAN, plan))

    def test_a_crash_before_the_pointer_moves_leaves_the_old_one_abandoned(
            self, repo, archive, monkeypatch):
        plan = _plan(repo)
        first = _start(plan, PLAN, "--models", M1)
        real = ic._write_json

        def crash(path, document, definition):
            if definition == "pointer":
                raise OSError("crash")
            real(path, document, definition)

        monkeypatch.setattr(ic, "_write_json", crash)
        with pytest.raises(OSError):
            ic.advance(plan, PLAN, _args(PLAN, plan, "--force", "--cascade", "--models", M1),
                       initial=True)
        monkeypatch.setattr(ic, "_write_json", real)
        current = ic.load_current(plan, PLAN)
        assert current.cascade_id == first.cascade_id and current.state == ic.ABANDONED
        assert len(list((ic.composite_dir(plan, PLAN) / ic.CASCADES_DIR).iterdir())) == 2
        # An abandoned cascade counts as none.
        assert not ic.applies(plan, PLAN, _args(PLAN, plan))
        again = ic.advance(plan, PLAN, _args(PLAN, plan, "--cascade", "--models", M1),
                           initial=True)
        assert again.cascade_id != first.cascade_id

    @pytest.mark.parametrize("which, damage, detail", [
        ("pointer", "{not json", "Expecting property name"),
        ("pointer", '{"schema_version": 1}', "missing required field 'cascade_id'"),
        ("record", '{"schema_version": 1}', "missing required field"),
        ("record", "STATE", "not in enum"),
        ("record", "MODE", "not improve-plan's"),
        ("record", "INDEX", "stage 1 has index 2"),
        ("record", None, "it does not exist"),
    ])
    def test_a_damaged_file_is_exit_3_naming_it(self, repo, archive, capsys, which,
                                                damage, detail):
        plan = _plan(repo)
        stage = _start(plan, PLAN, "--models", M1)
        target = (ic.pointer_path(plan, PLAN) if which == "pointer"
                  else stage.cascade_dir / ic.RECORD_NAME)
        record = json.loads((stage.cascade_dir / ic.RECORD_NAME).read_text(encoding="utf-8"))
        if damage is None:
            target.unlink()
        elif damage == "STATE":
            record["state"] = "bogus"
            target.write_text(json.dumps(record), encoding="utf-8")
        elif damage == "MODE":
            record["mode"] = CODE
            target.write_text(json.dumps(record), encoding="utf-8")
        elif damage == "INDEX":
            record["stages"][0]["index"] = 2
            target.write_text(json.dumps(record), encoding="utf-8")
        else:
            target.write_text(damage, encoding="utf-8")
        with pytest.raises(ic.CascadeRecordError) as excinfo:
            ic.load_current(plan, PLAN)
        assert excinfo.value.code == 3 and detail in excinfo.value.detail
        assert excinfo.value.path == str(target)
        args = _args(PLAN, plan)
        assert _preflight(PLAN, plan, args) == 3
        err = capsys.readouterr().err
        assert "cannot read" in err and target.name in err and "nothing was run" in err

    def test_cascade_status_reads_the_current_stage(self, repo, archive):
        plan = _plan(repo)
        assert ic.cascade_status(plan, PLAN) is None
        first = _start(plan, PLAN, "--models", M1, M2)
        status = ic.cascade_status(plan, PLAN)
        assert status == {"cascade_id": first.cascade_id, "state": ic.RUNNING, "stage": 1,
                          "of": 2, "model": M1, "dir": str(first.cascade_dir)}
        _run_stage(plan, PLAN, first, round_id=ROUND_1, review_at=1, apply_at=2)
        ic.advance(plan, PLAN, _args(PLAN, plan), initial=True)
        assert ic.cascade_status(plan, PLAN)["stage"] == 2


class TestTheSchema:
    def _documents(self, repo, archive):
        plan = _plan(repo)
        done = _finish_cascade(plan, CODE, "--models", M1, "--files", "a.py",
                               "--base-ref", "main")
        pointer = json.loads(ic.pointer_path(plan, CODE).read_text(encoding="utf-8"))
        record = json.loads((done.cascade_dir / ic.RECORD_NAME).read_text(encoding="utf-8"))
        return pointer, record

    def test_written_documents_pass_jsonschema_and_the_runtime_validator(self, repo, archive):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft7Validator.check_schema(schema)
        pointer, record = self._documents(repo, archive)
        for document, definition in ((pointer, "pointer"), (record, "record")):
            jsonschema.validate(document, schema)
            jsonschema.validate(document, {"$ref": f"#/definitions/{definition}",
                                           "definitions": schema["definitions"]})
            assert ic._schema_problems(document, definition) == []

    @pytest.mark.parametrize("change", [
        lambda r: r.update(state="paused"),
        lambda r: r["stages"][0].update(state="half"),
        lambda r: r["selectors"].update(files_file="../../etc/passwd"),
        lambda r: r.update(extra=1),
        lambda r: r["stages"][0].update(bound_round="not-hex"),
        lambda r: r["stages"][0]["counts"].update(applied="2"),
        lambda r: r.update(stages=[]),
    ])
    def test_both_validators_refuse_the_same_bad_record(self, repo, archive, change):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        _pointer, record = self._documents(repo, archive)
        change(record)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(record, {"$ref": "#/definitions/record",
                                         "definitions": schema["definitions"]})
        assert ic._schema_problems(record, "record")


# ---------------------------------------------------------------------------
# 9. The composites' own bits
# ---------------------------------------------------------------------------

class TestTheComposites:
    @pytest.mark.parametrize("mode", [PLAN, CODE])
    def test_both_parsers_take_cascade(self, mode, tmp_path):
        assert _args(mode, tmp_path / "p.md", "--cascade").cascade is True
        assert _args(mode, tmp_path / "p.md").cascade is False
        assert improve_steps.parse_args(["p.md", "--mode", mode, "--cascade"]).cascade

    def test_improve_commits_refuses_it_as_an_unknown_flag(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as excinfo:
            iw.parse_args("improve-commits", ["--plan-file", str(tmp_path / "b.md"),
                                              "--cascade"])
        assert excinfo.value.code == 2
        assert "unrecognized arguments: --cascade" in capsys.readouterr().err
        assert "--cascade" not in improve_steps.build_parser(
            "improve-commits")._option_string_actions
        assert "--cascade" in improve_steps.build_parser(None)._option_string_actions

    @pytest.mark.parametrize("mode", [PLAN, CODE])
    def test_the_continuation_carries_cascade_before_resume(self, mode, tmp_path):
        plain = _args(mode, tmp_path / "p.md", "--models", M1)
        assert iw.continuation_flags(mode, plain) == ["--models", M1, "--resume"]
        cascade = _args(mode, tmp_path / "p.md", "--models", M1, "--force", "--cascade")
        assert iw.continuation_flags(mode, cascade) == ["--models", M1, "--cascade",
                                                        "--resume"]
        assert iw.recorded_continuation(mode, ["--cascade", "--quick"]) == [
            "--quick", "--cascade", "--resume"]

    @pytest.mark.parametrize("mode", [PLAN, CODE])
    def test_no_stage_file_no_hidden_flag(self, repo, mode):
        plan = _plan(repo)
        args = _args(mode, plan, "--cascade", "--models", M1)
        records = iw.read_records(mode, plan, args)
        for step in iw._row(mode).sequence:
            assert provenance.HIDDEN_FLAG not in iw.step_flags(mode, step, args, records)
        args.cascade_stage_file = "/abs/manifest.json"
        review, apply = iw._row(mode).sequence
        assert iw.step_flags(mode, review, args, records)[-2:] == [
            provenance.HIDDEN_FLAG, "/abs/manifest.json"]
        assert provenance.HIDDEN_FLAG not in iw.step_flags(mode, apply, args, records)

    def test_refusal_exit_prints_by_code(self, capsys):
        assert ic.refusal_exit(ic.CascadeRefusal(1, "ERROR: one")) == 1
        assert ic.refusal_exit(ic.CascadeRefusal(code=3, message="ERROR: three")) == 3
        assert capsys.readouterr().err == "ERROR: one\nERROR: three\n"
