"""Ask mode never reuses an answer whose safety checks failed or never finished.

The review lanes' records (``models_rejected``, ``models_unchecked``;
docs/development/review-results.md, "Withheld slot answers") on ask's question
folder: a slot is armed before its dispatch, a rejection is recorded before its
answer is set aside, the launch carries both maps over, and the resume skip,
the closing scan and ``--accept-partial`` read them. See
docs/development/workflows.md#ask-mode.

The direct rows drive :class:`ask_orchestrator.AskRun` through
``test_ask_run``'s ``slot`` fixture; the end-to-end rows run
``ask_orchestrator.main`` with ``test_ask``'s ``FakeInvoker``.
"""

import asyncio
import errno
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import ask_orchestrator  # noqa: E402
from tests.test_ask import (  # noqa: E402
    FakeInvoker,
    answer_of,
    ask_argv,
    failed_line,
    partial_of,
    qdir,
    read_answers,
    read_status,
    run_ask,
    timeout_result,
)
from tests.test_ask_run import SPEC, _ask, slot  # noqa: E402,F401
from utils import result_files, review_lane, slot_guard, status_file  # noqa: E402
from utils import review_orchestrator_base as rob  # noqa: E402
from utils.json_extractor import sanitize_model_name  # noqa: E402

REJECTED = result_files.REJECTED_MODELS_KEY
UNCHECKED = result_files.UNCHECKED_MODELS_KEY
VIOLATION = "[MODE_CHECK] simulated failure"
STEM = sanitize_model_name(SPEC)
SANDBOXED = rob.SandboxVerdict(rob.SANDBOX_DISPATCH, None)
REFUSED = rob.SandboxVerdict(rob.SANDBOX_REFUSE, "[SANDBOX_UNAVAILABLE] simulated")


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    monkeypatch.setenv("MULTI_LLM_TEST_FAST_BACKOFF", "1")


def _recorded(question_dir, key):
    return status_file.read_status(str(question_dir)).get(key) or {}


def _violating(monkeypatch, when=lambda: True):
    """The guard's checks fail every slot while ``when()`` holds."""
    real = slot_guard.SlotGuard.finish

    async def finish(self, *args, **kwargs):
        if when():
            return VIOLATION
        return await real(self, *args, **kwargs)

    monkeypatch.setattr(slot_guard.SlotGuard, "finish", finish)


def _block(path):
    """Take *path* with a directory, so ``os.replace`` onto it fails."""
    Path(path).mkdir()
    (Path(path) / "keep").write_text("", encoding="utf-8")


def _blocking_during(invoker, *paths):
    """*invoker*, taking each of *paths* with a directory during its call:
    after the dispatch removed what an earlier run left there, which it
    refuses to ask over, and before the slot sets its answer aside."""
    def call(*args, **kwargs):
        for path in paths:
            _block(path)
        return invoker(*args, **kwargs)

    return call


def _printed(capsys):
    """Both streams: a fail-closed stop prints on stderr."""
    captured = capsys.readouterr()
    return captured.out + captured.err


def _writes(slot, text):
    def invoke(**_):
        slot["answer"].write_text(text, encoding="utf-8")
        return {"success": True, "details": {}}

    return invoke


# ---------------------------------------------------------------------------
# A rejected answer
# ---------------------------------------------------------------------------

def test_a_rejected_answer_whose_set_aside_failed_is_never_resumed(slot, monkeypatch):
    """Codex's repro (U90 review, finding 4): the slot's checks fail, its
    answer cannot be set aside, and a fresh run's resume skip used to count
    the file as answered."""
    slot["invoke"] = _writes(slot, "Rejected answer")
    _violating(monkeypatch)
    monkeypatch.setattr(ask_orchestrator, "set_violated_answer_aside", lambda *a: False)
    run = slot["make"]()
    result = _ask(run)
    assert not result["success"] and result["rejected"] == VIOLATION
    assert SPEC in run.answers_left_in_place
    assert _recorded(slot["dir"], REJECTED) == {SPEC: VIOLATION}

    resumed = slot["make"]()
    assert resumed.skip_existing_answer(SPEC, 0) is None
    assert slot["answer"].read_text(encoding="utf-8") == "Rejected answer"


def test_a_rejection_is_recorded_before_the_answer_is_set_aside(slot, monkeypatch):
    slot["invoke"] = _writes(slot, "Rejected answer")
    _violating(monkeypatch)
    seen = []

    def set_aside(spec, path):
        seen.append(dict(_recorded(slot["dir"], REJECTED)))
        os.replace(path, path + ".violated")
        return True

    monkeypatch.setattr(ask_orchestrator, "set_violated_answer_aside", set_aside)
    _ask(slot["make"]())
    assert seen == [{SPEC: VIOLATION}]


def test_the_next_ask_sets_the_rejected_file_aside_and_asks_again(slot, monkeypatch):
    rejecting = [True]
    _violating(monkeypatch, when=lambda: rejecting[0])
    _block(str(slot["answer"]) + ".violated")
    slot["invoke"] = _writes(slot, "Rejected answer")
    _ask(slot["make"]())
    assert slot["answer"].read_text(encoding="utf-8") == "Rejected answer"

    # The quarantine name frees up, the checks pass, the model answers anew.
    Path(str(slot["answer"]) + ".violated", "keep").unlink()
    Path(str(slot["answer"]) + ".violated").rmdir()
    rejecting[0] = False
    slot["invoke"] = _writes(slot, "A clean answer")
    run = slot["make"]()
    assert run.skip_existing_answer(SPEC, 0) is None
    result = _ask(run)
    assert result["success"] and result["checks_passed"] is True
    assert slot["answer"].read_text(encoding="utf-8") == "A clean answer"
    assert Path(str(slot["answer"]) + ".violated").read_text(
        encoding="utf-8") == "Rejected answer"
    assert SPEC not in _recorded(slot["dir"], UNCHECKED)


def test_a_rejected_file_that_cannot_be_set_aside_is_never_asked_over(slot, monkeypatch):
    _violating(monkeypatch)
    _block(str(slot["answer"]) + ".violated")
    slot["invoke"] = _writes(slot, "Rejected answer")
    _ask(slot["make"]())
    calls = len(slot["calls"])

    result = _ask(slot["make"](rerun_all=True))
    assert result["source"] == "not-dispatched"
    assert "could not be set aside" in result["error"]
    assert len(slot["calls"]) == calls
    assert slot["answer"].read_text(encoding="utf-8") == "Rejected answer"


def _metadata_fails_after(monkeypatch, path, reads=1):
    """``os.stat`` of *path* works *reads* times, then fails with EACCES."""
    real_stat = os.stat
    seen = []

    def stat(target, *args, **kwargs):
        if not isinstance(target, int) and os.fspath(target) == os.fspath(path):
            seen.append(target)
            if len(seen) > reads:
                raise PermissionError(errno.EACCES, "simulated metadata read failure",
                                      os.fspath(path))
        return real_stat(target, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat)


def _rename_fails(monkeypatch, path):
    """``os.replace`` of *path* fails with EACCES (it is still there)."""
    real_replace = os.replace

    def replace(src, dst, *args, **kwargs):
        if os.fspath(src) == os.fspath(path):
            raise PermissionError(errno.EACCES, "simulated rename failure", os.fspath(src))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)


@pytest.mark.parametrize("key", [REJECTED, UNCHECKED])
def test_retiring_a_withheld_answer_tells_absence_from_a_failure(tmp_path, monkeypatch, key):
    """Only a confirmed absence frees the path: a metadata read or a rename
    that fails any other way leaves the earlier answer withheld."""
    answer = tmp_path / f"answer_{STEM}.md"
    why = result_files.Withheld(key, "an earlier slot")
    assert ask_orchestrator.set_withheld_answer_aside(SPEC, str(answer), why) is True

    answer.write_text("Withheld answer", encoding="utf-8")
    _metadata_fails_after(monkeypatch, answer)
    _rename_fails(monkeypatch, answer)
    assert ask_orchestrator.set_withheld_answer_aside(SPEC, str(answer), why) is False
    assert answer.read_text(encoding="utf-8") == "Withheld answer"


def test_a_rejected_answer_whose_metadata_cannot_be_read_stays_withheld(slot, monkeypatch):
    """The rejection's own set-aside: a metadata error is not an absence, so
    the slot keeps the file it could not move out of this run's scan."""
    slot["invoke"] = _writes(slot, "Rejected answer")
    _violating(monkeypatch)
    _metadata_fails_after(monkeypatch, slot["answer"], reads=0)
    _rename_fails(monkeypatch, slot["answer"])
    run = slot["make"]()
    assert _ask(run)["rejected"] == VIOLATION
    assert SPEC in run.answers_left_in_place
    assert slot["answer"].read_text(encoding="utf-8") == "Rejected answer"


# ---------------------------------------------------------------------------
# Before the dispatch: the unchecked record
# ---------------------------------------------------------------------------

def test_a_slot_is_recorded_unchecked_until_its_checks_pass(slot):
    during = []

    def invoke(**_):
        during.append(dict(_recorded(slot["dir"], UNCHECKED)))
        slot["answer"].write_text("An answer.\n", encoding="utf-8")
        return {"success": True, "details": {}}

    slot["invoke"] = invoke
    assert _ask(slot["make"]())["success"]
    assert during == [{SPEC: review_lane.ARMED_REASON}]
    assert _recorded(slot["dir"], UNCHECKED) == {}


def test_a_slot_whose_record_cannot_be_written_is_never_dispatched(slot, monkeypatch):
    slot["answer"].write_text("An earlier answer.\n", encoding="utf-8")
    slot["partial"].write_text("An earlier partial.\n", encoding="utf-8")
    monkeypatch.setattr(result_files, "record_unchecked_model", lambda *a, **k: False)
    run = slot["make"](rerun_all=True)
    result = _ask(run)
    assert result == {"model": SPEC, "success": False,
                      "error": review_lane.UNRECORDED_REASON, "source": "not-dispatched"}
    assert slot["calls"] == [] and run.dispatched_specs == set()
    assert slot["answer"].read_text(encoding="utf-8") == "An earlier answer.\n"
    assert slot["partial"].read_text(encoding="utf-8") == "An earlier partial.\n"


def test_a_slot_that_raises_after_its_dispatch_stays_unchecked(slot, monkeypatch):
    async def broken(self, *args, **kwargs):
        raise RuntimeError("the checks broke")

    monkeypatch.setattr(slot_guard.SlotGuard, "finish", broken)
    slot["invoke"] = _writes(slot, "Never checked")
    with pytest.raises(RuntimeError):
        _ask(slot["make"]())
    assert _recorded(slot["dir"], UNCHECKED) == {SPEC: review_lane.ARMED_REASON}
    assert slot["make"]().skip_existing_answer(SPEC, 0) is None


def test_a_slot_cancelled_before_its_checks_finish_stays_unchecked(slot, monkeypatch):
    async def slow(self, *args, **kwargs):
        await asyncio.sleep(30)

    monkeypatch.setattr(slot_guard.SlotGuard, "finish", slow)
    slot["invoke"] = _writes(slot, "Never checked")
    run = slot["make"]()

    async def cut():
        await asyncio.wait_for(run.ask_one(SPEC, 0), timeout=0.5)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(cut())
    assert SPEC in _recorded(slot["dir"], UNCHECKED)
    assert slot["make"]().skip_existing_answer(SPEC, 0) is None


def test_an_unreadable_checkpoint_withholds_every_answer(slot):
    slot["answer"].write_text("An earlier answer.\n", encoding="utf-8")
    (slot["dir"] / ".status.json").write_text("{not json", encoding="utf-8")
    run = slot["make"]()
    assert run.skip_existing_answer(SPEC, 0) is None
    result = _ask(run)
    assert result["source"] == "not-dispatched"
    assert result["error"] == review_lane.UNRECORDED_REASON
    assert slot["calls"] == []
    assert slot["answer"].read_text(encoding="utf-8") == "An earlier answer.\n"


def test_a_folder_with_no_checkpoint_dispatches_with_no_record(slot):
    """Only a direct call gets here: the launch writes the checkpoint and
    reads it back before any slot (``_record_launch``)."""
    (slot["dir"] / ".status.json").unlink()
    slot["invoke"] = _writes(slot, "An answer.\n")
    assert _ask(slot["make"]())["success"]
    assert _recorded(slot["dir"], UNCHECKED) == {}


# ---------------------------------------------------------------------------
# End to end: the launch, the closing scan, --accept-partial
# ---------------------------------------------------------------------------

@pytest.fixture
def plan(tmp_path, monkeypatch):
    monkeypatch.setattr(rob, "sandbox_refusal", lambda *_a, **_k: SANDBOXED)
    p = tmp_path / "my-feature.md"
    p.write_text("# My Feature\n\nBuild a widget.\n", encoding="utf-8")
    return p


def _rejected_and_left(plan, monkeypatch, *, stdout=None):
    """Run 1: the slot answers, its checks fail, the quarantine name is
    taken. Returns the question folder; the checks pass from now on."""
    d = qdir(plan, "Q")
    d.mkdir(parents=True)
    _block(str(answer_of(d, SPEC)) + ".violated")
    rejecting = [True]
    _violating(monkeypatch, when=lambda: rejecting[0])
    inv = FakeInvoker()
    if stdout is None:
        inv.configure(SPEC, write_file=True, content="Rejected answer")
    else:
        inv.configure(SPEC, write_file=False, result={
            "success": True, "details": {"answer_text": stdout}})
    assert run_ask(ask_argv(plan, SPEC), inv) == 3
    rejecting[0] = False
    return d


@pytest.mark.parametrize("rerun_all", [False, True])
def test_the_closing_scan_never_counts_a_withheld_answer(plan, monkeypatch, rerun_all):
    """The next run's slot is refused at the sandbox gate, so nothing sets the
    rejected file aside: the scan leaves it unread anyway."""
    d = _rejected_and_left(plan, monkeypatch)
    monkeypatch.setattr(rob, "sandbox_refusal", lambda *_a, **_k: REFUSED)
    inv = FakeInvoker()
    extra = ("--rerun-all",) if rerun_all else ()
    assert run_ask(ask_argv(plan, SPEC, extra=extra), inv) == 3
    assert inv.calls == []
    status = read_status(d)
    assert status["models_completed"] == []
    assert status[REJECTED] == {SPEC: VIOLATION}
    assert "Rejected answer" not in read_answers(d)
    assert "is from a slot that failed its safety checks" in failed_line(d, SPEC)
    assert answer_of(d, SPEC).read_text(encoding="utf-8") == "Rejected answer"


def test_the_launch_carries_the_records_of_models_it_did_not_select(plan, monkeypatch):
    d = _rejected_and_left(plan, monkeypatch)
    other = "gemini:gemini-2.5-pro"
    assert run_ask(ask_argv(plan, other), FakeInvoker()) == 0
    assert read_status(d)[REJECTED] == {SPEC: VIOLATION}
    assert read_status(d)["models_requested"] == [other]
    calls = FakeInvoker()
    assert run_ask(ask_argv(plan, SPEC), calls) == 3
    assert calls.calls == [], "the rejected file is still in place: not asked over"
    assert "Rejected answer" not in read_answers(d)


def test_a_checked_answer_supersedes_the_rejection(plan, monkeypatch):
    d = _rejected_and_left(plan, monkeypatch)
    blocked = Path(str(answer_of(d, SPEC)) + ".violated")
    (blocked / "keep").unlink()
    blocked.rmdir()
    inv = FakeInvoker()
    inv.configure(SPEC, write_file=True, content="A clean answer")
    assert run_ask(ask_argv(plan, SPEC), inv) == 0
    assert inv.calls == [SPEC]
    status = read_status(d)
    assert status["models_completed"] == [SPEC]
    assert REJECTED not in status or status[REJECTED] == {}
    assert UNCHECKED not in status or status[UNCHECKED] == {}
    assert "A clean answer" in read_answers(d)
    assert blocked.read_text(encoding="utf-8") == "Rejected answer"
    # From here on it is an ordinary answer: the resume skips it.
    again = FakeInvoker()
    assert run_ask(ask_argv(plan, SPEC), again) == 0
    assert again.calls == []


def test_a_launch_over_an_unreadable_checkpoint_stops_before_dispatch(plan, capsys):
    d = qdir(plan, "Q")
    d.mkdir(parents=True)
    (d / ".status.json").write_text("{not json", encoding="utf-8")
    answer_of(d, SPEC).write_text("An earlier answer.\n", encoding="utf-8")
    inv = FakeInvoker()
    assert run_ask(ask_argv(plan, SPEC), inv) == 3
    assert inv.calls == []
    assert (d / ".status.json").read_text(encoding="utf-8") == "{not json"
    out = _printed(capsys)
    assert "could not be read when this ask started" in out
    assert status_file.unreadable_checkpoint_advice(str(d)) in out


def test_a_launch_whose_checkpoint_cannot_be_read_back_stops(plan, monkeypatch, capsys):
    monkeypatch.setattr(ask_orchestrator, "write_status", lambda *_a, **_k: None)
    inv = FakeInvoker()
    assert run_ask(ask_argv(plan, SPEC), inv) == 3
    assert inv.calls == []
    out = _printed(capsys)
    assert "could not be written and read back" in out
    assert status_file.unreadable_checkpoint_advice(str(qdir(plan, "Q"))) in out


def test_a_deadline_loss_whose_file_stays_is_never_resumed(plan, monkeypatch):
    """The fan-out's deadline cuts the slot while its checks run; its answer
    file can be neither moved nor copied aside, so it stays in place."""
    d = qdir(plan, "Q")
    slow = [True]
    real = slot_guard.SlotGuard.finish

    async def finish(self, *args, **kwargs):
        if slow[0]:
            await asyncio.sleep(10)
        return await real(self, *args, **kwargs)

    monkeypatch.setattr(slot_guard.SlotGuard, "finish", finish)
    monkeypatch.setattr("utils.concurrency.PROVIDER_STAGGER_DELAY", 0.0)
    inv = FakeInvoker()
    inv.configure(SPEC, write_file=True, content="Never checked")
    blocking = _blocking_during(inv, partial_of(d, SPEC), d / f"error_{STEM}.log")
    assert run_ask(ask_argv(plan, SPEC, extra=("--timeout", "1")), blocking) == 3
    assert answer_of(d, SPEC).read_text(encoding="utf-8") == "Never checked"
    assert SPEC in read_status(d)[UNCHECKED]

    # A withheld model's partial must be removable before it is asked again.
    for blocked in (partial_of(d, SPEC), d / f"error_{STEM}.log"):
        (blocked / "keep").unlink()
        blocked.rmdir()
    slow[0] = False
    again = FakeInvoker()
    again.configure(SPEC, write_file=True, content="A checked answer")
    assert run_ask(ask_argv(plan, SPEC), again) == 0
    assert again.calls == [SPEC]
    assert Path(str(answer_of(d, SPEC)) + ".unchecked").read_text(
        encoding="utf-8") == "Never checked"
    assert "A checked answer" in read_answers(d)


def test_the_run_stops_when_nothing_withholds_a_rejected_file(plan, monkeypatch, capsys):
    """The last resort: the record the slot armed vanishes from outside, and
    neither the rejection nor its retry can be recorded."""
    d = qdir(plan, "Q")
    d.mkdir(parents=True)
    _block(str(answer_of(d, SPEC)) + ".violated")
    _violating(monkeypatch)
    broken = [False]
    real_write = status_file.write_in_plan_dir

    def write_in_plan_dir(*args, **kwargs):
        if broken[0]:
            raise OSError("simulated storage failure")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(status_file, "write_in_plan_dir", write_in_plan_dir)

    class Vanishing(FakeInvoker):
        def __call__(self, *args, **kwargs):
            result_files.clear_unchecked_model(str(d), SPEC)
            broken[0] = True
            return super().__call__(*args, **kwargs)

    inv = Vanishing()
    inv.configure(SPEC, write_file=True, content="Rejected answer")
    with pytest.raises(SystemExit) as stop:
        run_ask(ask_argv(plan, SPEC), inv)
    assert stop.value.code == 3
    out = _printed(capsys)
    assert "could not record that they are withheld" in out
    assert f"{SPEC}: answer_{STEM}.md" in out
    assert not (d / "answers.md").exists()


def _answers_nothing():
    """A provider that ends cleanly and writes no answer anywhere."""
    inv = FakeInvoker()
    inv.configure(SPEC, write_file=False, result={"success": True, "details": {}})
    return inv


def test_a_metadata_error_never_shows_a_rejected_answer_gone(plan, monkeypatch):
    """The rejected answer survives its set-aside; at the retry its first
    metadata read works and every later one fails. That retry, whose model
    writes nothing, must not read the rejected file as its own answer."""
    d = _rejected_and_left(plan, monkeypatch)
    _metadata_fails_after(monkeypatch, os.path.realpath(answer_of(d, SPEC)))
    inv = _answers_nothing()
    assert run_ask(ask_argv(plan, SPEC), inv) == 3
    assert inv.calls == []
    assert "Rejected answer" not in read_answers(d)
    assert read_status(d)[REJECTED] == {SPEC: VIOLATION}
    assert answer_of(d, SPEC).read_text(encoding="utf-8") == "Rejected answer"


def test_a_withheld_partial_that_cannot_be_removed_is_never_asked_over(plan, monkeypatch,
                                                                     capsys):
    """A rejected stdout answer survives as the partial. A retry that passes
    its checks and leaves nothing used to supersede the rejection anyway, so
    a later --accept-partial counted the rejected text without a warning."""
    d = _rejected_and_left(plan, monkeypatch, stdout="A stdout answer, rejected.")
    partial = partial_of(d, SPEC)
    real_remove = os.remove
    removable = [False]

    def remove(path, *args, **kwargs):
        if os.fspath(path) == str(partial) and not removable[0]:
            raise PermissionError(errno.EACCES, "simulated", str(partial))
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr(os, "remove", remove)
    inv = FakeInvoker()
    inv.configure(SPEC, write_file=False, result=timeout_result())
    assert run_ask(ask_argv(plan, SPEC), inv) == 3
    assert inv.calls == []
    assert "partial_" in failed_line(d, SPEC) and "could not be removed" in failed_line(d, SPEC)
    assert read_status(d)[REJECTED] == {SPEC: VIOLATION}

    removable[0] = True
    capsys.readouterr()
    assert run_ask(ask_argv(plan, SPEC, extra=("--accept-partial", SPEC)), FakeInvoker()) == 0
    assert f"WARNING: {SPEC}'s last slot failed its safety checks" in capsys.readouterr().out


def test_a_withheld_partial_is_removed_before_the_model_is_asked_again(plan, monkeypatch):
    d = _rejected_and_left(plan, monkeypatch, stdout="A stdout answer, rejected.")
    inv = FakeInvoker()
    inv.configure(SPEC, write_file=False, result=timeout_result())
    assert run_ask(ask_argv(plan, SPEC), inv) == 3
    assert inv.calls == [SPEC]
    assert not partial_of(d, SPEC).exists()
    status = read_status(d)
    assert status.get(REJECTED, {}) == {} and status.get(UNCHECKED, {}) == {}


@pytest.mark.skipif(sys.platform == "win32", reason="needs symlinks")
def test_an_answer_link_never_takes_the_record_out_of_the_question_folder(plan, monkeypatch):
    """``answer_<model>.md`` links into a folder with no checkpoint: the slot
    must still be recorded in the question folder's, or the unchecked answer
    an interruption leaves is reused by the next run."""
    d = qdir(plan, "Q")
    (d / "nested").mkdir(parents=True)
    answer_of(d, SPEC).symlink_to(d / "nested" / "answer.md")
    interrupting = [True]
    real = slot_guard.SlotGuard.finish

    async def finish(self, *args, **kwargs):
        if interrupting[0]:
            raise KeyboardInterrupt  # stops the run before its close
        return await real(self, *args, **kwargs)

    monkeypatch.setattr(slot_guard.SlotGuard, "finish", finish)
    inv = FakeInvoker()
    inv.configure(SPEC, write_file=True, content="Never checked")
    with pytest.raises(KeyboardInterrupt):
        run_ask(ask_argv(plan, SPEC), inv)
    assert (d / "nested" / "answer.md").read_text(encoding="utf-8") == "Never checked"
    assert read_status(d)[UNCHECKED] == {SPEC: review_lane.ARMED_REASON}

    interrupting[0] = False
    again = FakeInvoker()
    again.configure(SPEC, write_file=True, content="A checked answer")
    assert run_ask(ask_argv(plan, SPEC), again) == 0
    assert again.calls == [SPEC]
    assert "Never checked" not in read_answers(d)
    assert "A checked answer" in read_answers(d)


def test_a_cli_slot_with_no_checkpoint_to_record_in_is_never_dispatched(plan, monkeypatch):
    """The direct-call exemption is for a slot no launch opened: a CLI slot
    whose checkpoint is gone by its arm is not dispatched."""
    d = qdir(plan, "Q")
    real = slot_guard.SlotGuard.begin

    async def begin(self, *args, **kwargs):
        gate = await real(self, *args, **kwargs)
        (d / ".status.json").unlink()
        return gate

    monkeypatch.setattr(slot_guard.SlotGuard, "begin", begin)
    inv = FakeInvoker()
    assert run_ask(ask_argv(plan, SPEC), inv) == 3
    assert inv.calls == []
    assert review_lane.UNRECORDED_REASON in failed_line(d, SPEC)


def _unfinished_and_left(plan):
    """Run 1: the checks pass, the provider times out after writing a draft,
    and the draft can be neither moved nor copied aside."""
    d = qdir(plan, "Q")
    inv = FakeInvoker()
    inv.configure(SPEC, write_file=True, content="A draft", result=timeout_result())
    blocking = _blocking_during(inv, partial_of(d, SPEC), d / f"error_{STEM}.log")
    assert run_ask(ask_argv(plan, SPEC), blocking) == 3
    assert answer_of(d, SPEC).read_text(encoding="utf-8") == "A draft"
    for blocked in (partial_of(d, SPEC), d / f"error_{STEM}.log"):
        (blocked / "keep").unlink()
        blocked.rmdir()
    return d


def test_an_unfinished_answer_left_in_place_is_never_resumed(plan):
    d = _unfinished_and_left(plan)
    status = read_status(d)
    assert status[UNCHECKED] == {SPEC: result_files.UNFINISHED_ANSWER_REASON}
    assert status["models_completed"] == []

    again = FakeInvoker()
    again.configure(SPEC, write_file=True, content="A finished answer")
    assert run_ask(ask_argv(plan, SPEC), again) == 0
    assert again.calls == [SPEC]
    assert "A draft" not in read_answers(d)
    assert "A finished answer" in read_answers(d)
    assert Path(str(answer_of(d, SPEC)) + ".unchecked").read_text(
        encoding="utf-8") == "A draft"
    assert read_status(d).get(UNCHECKED, {}) == {}


def test_an_unfinished_answer_is_counted_only_when_accepted(plan, capsys):
    d = _unfinished_and_left(plan)
    capsys.readouterr()
    accept = ask_argv(plan, SPEC, extra=("--accept-partial", SPEC))
    assert run_ask(accept, FakeInvoker()) == 3
    assert "is from a slot that left an unfinished answer" in _printed(capsys)

    os.replace(answer_of(d, SPEC), partial_of(d, SPEC))
    assert run_ask(accept, FakeInvoker()) == 0
    out = capsys.readouterr().out
    assert f"WARNING: {SPEC}'s last slot left an unfinished answer" in out
    assert read_status(d)["models_completed"] == [SPEC]
    assert "A draft" in read_answers(d)


# ---------------------------------------------------------------------------
# --accept-partial: the operator's opt-in, never silent
# ---------------------------------------------------------------------------

def test_accepting_a_rejected_partial_warns_and_clears_the_record(plan, monkeypatch, capsys):
    d = _rejected_and_left(plan, monkeypatch, stdout="A stdout answer, rejected.")
    assert partial_of(d, SPEC).exists()
    assert read_status(d)[REJECTED] == {SPEC: VIOLATION}
    capsys.readouterr()

    inv = FakeInvoker()
    assert run_ask(ask_argv(plan, SPEC, extra=("--accept-partial", SPEC)), inv) == 0
    assert inv.calls == []
    out = capsys.readouterr().out
    assert f"WARNING: {SPEC}'s last slot failed its safety checks ({VIOLATION})" in out
    assert "counted only because --accept-partial names it" in out
    status = read_status(d)
    assert status["models_completed"] == [SPEC]
    assert status[REJECTED] == {} and status.get(UNCHECKED, {}) == {}
    assert "A stdout answer, rejected." in read_answers(d)


def test_accepting_refuses_while_the_withheld_answer_file_is_in_place(plan, monkeypatch, capsys):
    d = _rejected_and_left(plan, monkeypatch)
    partial_of(d, SPEC).write_text("Some partial.\n", encoding="utf-8")
    before = (d / ".status.json").read_bytes()
    assert run_ask(ask_argv(plan, SPEC, extra=("--accept-partial", SPEC)), FakeInvoker()) == 3
    assert "is from a slot that failed its safety checks" in _printed(capsys)
    assert (d / ".status.json").read_bytes() == before
    assert partial_of(d, SPEC).exists()


def test_accepting_refuses_over_an_unreadable_checkpoint(plan, capsys):
    d = qdir(plan, "Q")
    assert run_ask(ask_argv(plan, SPEC), _timing_out()) == 3
    (d / ".status.json").write_text("{not json", encoding="utf-8")
    assert run_ask(ask_argv(plan, SPEC, extra=("--accept-partial", SPEC)), FakeInvoker()) == 3
    assert "could not be read" in _printed(capsys)
    assert partial_of(d, SPEC).exists() and not answer_of(d, SPEC).exists()


# ---------------------------------------------------------------------------
# Controls: what the records leave as it was
# ---------------------------------------------------------------------------

def _timing_out():
    inv = FakeInvoker()
    inv.configure(SPEC, write_file=False, result=timeout_result(text="Half an answer."))
    return inv


def test_an_ordinary_failed_slot_leaves_no_record(plan, capsys):
    d = qdir(plan, "Q")
    assert run_ask(ask_argv(plan, SPEC), _timing_out()) == 3
    status = read_status(d)
    assert status.get(REJECTED, {}) == {} and status.get(UNCHECKED, {}) == {}
    capsys.readouterr()
    assert run_ask(ask_argv(plan, SPEC, extra=("--accept-partial", SPEC)), FakeInvoker()) == 0
    assert "WARNING" not in _printed(capsys)
    assert read_status(d)["models_completed"] == [SPEC]


def test_a_healthy_resume_skips_the_answer(plan):
    d = qdir(plan, "Q")
    assert run_ask(ask_argv(plan, SPEC), FakeInvoker()) == 0
    again = FakeInvoker()
    assert run_ask(ask_argv(plan, SPEC), again) == 0
    assert again.calls == []
    status = read_status(d)
    assert status["models_completed"] == [SPEC]
    assert status.get(REJECTED, {}) == {} and status.get(UNCHECKED, {}) == {}


def test_an_older_checkpoint_without_the_records_resumes_as_before(plan):
    d = qdir(plan, "Q")
    d.mkdir(parents=True)
    (d / ".status.json").write_text(json.dumps({
        "phase": "ask", "state": "completed", "question": "Q",
        "models_requested": [SPEC], "models_completed": [SPEC],
    }), encoding="utf-8")
    answer_of(d, SPEC).write_text("An answer from before the records.\n", encoding="utf-8")
    inv = FakeInvoker()
    assert run_ask(ask_argv(plan, SPEC), inv) == 0
    assert inv.calls == []
    assert "An answer from before the records." in read_answers(d)
