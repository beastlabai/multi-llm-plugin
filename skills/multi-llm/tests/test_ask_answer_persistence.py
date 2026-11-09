"""Ask mode never publishes an answer that was never completely written, and
never counts one an earlier run left as this run's.

Step 2 of ``AskRun.ask_one`` writes a stdout answer to ``answer_<model>.md``
atomically (``publish_atomically``), and ``answers.md`` is written the same
way, so a full disk leaves the path as it was, never a cut prefix that the
closing scan, a resume or a reader would take for the whole text. A slot
whose write fails goes to step 3 with that error and keeps its text as the
partial, which a file left at the answer path never displaces. Before a
dispatch, the partial an earlier run left, its origin record and, under
``--rerun-all``, its answer are removed, in that order, and one that cannot be
removed stops the dispatch with the earlier answer and its origin kept: only
``FileNotFoundError`` shows any of them absent. A ``--rerun-all`` never counts
an answer it did not produce, and a resume never asks over an earlier answer
it cannot read, nor stops over one that is not UTF-8 text. See
docs/development/workflows.md#ask-mode.

The direct rows drive :class:`ask_orchestrator.AskRun` through
``test_ask_run``'s ``slot`` fixture; the end-to-end rows run
``ask_orchestrator.main`` with ``test_ask``'s ``FakeInvoker``. The full disk
is a stream that lands a prefix and raises ``ENOSPC``, on the plain ``open()``
route and on ``atomic_io``'s ``os.fdopen`` route alike.
"""

import asyncio
import builtins
import errno
import os
import shutil
import stat
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
    origin_of,
    partial_note,
    partial_of,
    provenance_of,
    qdir,
    read_answers,
    read_status,
    run_ask,
    timeout_result,
)
from tests.test_ask_run import SPEC, _ask, slot  # noqa: E402,F401
from utils import atomic_io, result_files, slot_guard  # noqa: E402
from utils import review_orchestrator_base as rob  # noqa: E402
from utils.json_extractor import sanitize_model_name  # noqa: E402
from utils.progress_tracker import DEADLINE_ERROR  # noqa: E402

REJECTED = result_files.REJECTED_MODELS_KEY
UNCHECKED = result_files.UNCHECKED_MODELS_KEY
STEM = sanitize_model_name(SPEC)
SANDBOXED = rob.SandboxVerdict(rob.SANDBOX_DISPATCH, None)

#: The answer a full disk cuts: a write carrying MARKER lands PREFIX only.
PREFIX = "A cut prefix"
MARKER = "and the rest of the whole answer"
WHOLE = f"{PREFIX} {MARKER}."
FILE_LINE = "_answer file written by the model_"
FALLBACK_LINE = "_recovered from the model's stdout (it did not write the answer file)_"
OTHER = "claude-code:haiku"


def kept_note(spec=SPEC):
    """The note a model --rerun-all never dispatched gets beside its reason."""
    return (f"answer_{sanitize_model_name(spec)}.md, an earlier run's answer, was "
            "kept as it was (not counted: --rerun-all asks every model again)")


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    monkeypatch.setenv("MULTI_LLM_TEST_FAST_BACKOFF", "1")


@pytest.fixture
def plan(tmp_path, monkeypatch):
    monkeypatch.setattr(rob, "sandbox_refusal", lambda *_a, **_k: SANDBOXED)
    p = tmp_path / "my-feature.md"
    p.write_text("# My Feature\n\nBuild a widget.\n", encoding="utf-8")
    return p


class _ShortStream:
    """A write stream on which :class:`_FullDisk` decides each write."""

    def __init__(self, stream, disk):
        self._stream = stream
        self._disk = disk

    def write(self, data):
        if self._disk.fails(data):
            self._stream.write(data[:len(PREFIX)])
            self._stream.flush()
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        return self._stream.write(data)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._stream.close()
        return False

    def __iter__(self):
        return iter(self._stream)

    def __getattr__(self, name):
        return getattr(self._stream, name)


class _FullDisk:
    """Every write stream opened while installed, by ``open()`` or by
    ``os.fdopen``: a write carrying MARKER lands PREFIX, then raises
    ``ENOSPC``, as a disk that fills mid-write does; *times* such writes
    (``None``: every one). ``failed`` counts them."""

    def __init__(self, times=None):
        self.left = times
        self.failed = 0

    def fails(self, data):
        text = (bytes(data).decode("utf-8", "replace")
                if isinstance(data, (bytes, bytearray)) else str(data))
        if MARKER not in text or self.left == 0:
            return False
        if self.left is not None:
            self.left -= 1
        self.failed += 1
        return True

    def install(self, monkeypatch):
        real_open, real_fdopen = builtins.open, os.fdopen

        def wrap(stream, mode):
            return _ShortStream(stream, self) if set(str(mode)) & set("wax+") else stream

        monkeypatch.setattr(builtins, "open", lambda file, mode="r", *a, **k: wrap(
            real_open(file, mode, *a, **k), mode))
        monkeypatch.setattr(os, "fdopen", lambda fd, mode="r", *a, **k: wrap(
            real_fdopen(fd, mode, *a, **k), mode))
        return self


def _removal_fails(monkeypatch, path):
    """``os.remove`` of *path* raises ``EACCES``, as in a folder the user
    may not write to; every other removal goes through."""
    target = os.path.realpath(path)
    real = os.remove

    def remove(p, *args, **kwargs):
        if os.path.realpath(os.fspath(p)) == target:
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), os.fspath(p))
        return real(p, *args, **kwargs)

    monkeypatch.setattr(os, "remove", remove)


def _block(path):
    """Take *path* with a directory: a move or copy onto it fails."""
    Path(path).mkdir()
    (Path(path) / "keep").write_text("", encoding="utf-8")


def _unblock(path):
    shutil.rmtree(path)


def _blocking_during(invoker, *paths):
    """*invoker*, taking each of *paths* with a directory during its call:
    after the dispatch removed what an earlier run left there."""
    def call(*args, **kwargs):
        for path in paths:
            _block(path)
        return invoker(*args, **kwargs)

    return call


def _on_stdout(text):
    """A provider that ends cleanly with *text* on stdout and no file."""
    inv = FakeInvoker()
    inv.configure(SPEC, write_file=False, result={
        "success": True, "details": {"answer_text": text}})
    return inv


def _answers_nothing():
    inv = FakeInvoker()
    inv.configure(SPEC, write_file=False, result={"success": True, "details": {}})
    return inv


def _leftover_temps(folder):
    return sorted(p.name for p in Path(folder).iterdir()
                  if p.name.endswith(atomic_io.TEMP_SUFFIX))


def _records(question_dir):
    status = read_status(question_dir)
    return status.get(REJECTED, {}), status.get(UNCHECKED, {})


def _printed(capsys):
    captured = capsys.readouterr()
    return captured.out + captured.err


def _assert_refused_rerun(d, reason):
    """A --rerun-all whose slot was not dispatched: the model fails with
    *reason* and is not counted, while its earlier answer stays as it was,
    with no record withholding it."""
    status = read_status(d)
    assert status["state"] == "failed"
    assert status["models_completed"] == []
    assert reason in status["models_failed"][SPEC]
    assert kept_note() in status["models_failed"][SPEC]
    assert status["answer_origins"]["kept"] == 0
    assert status["progress"]["models"][SPEC]["source"] == "not-dispatched"
    assert reason in failed_line(d, SPEC)
    assert "Old answer" not in read_answers(d)
    assert answer_of(d, SPEC).read_text(encoding="utf-8") == "Old answer"
    assert _records(d) == ({}, {})


def _reads_fail(monkeypatch, path, *, once=False):
    """``open()`` of *path* for reading raises ``EACCES`` while the file is
    there (every time, or the first time only); writes and every other path
    go through."""
    target = os.path.realpath(path)
    real = builtins.open
    failed = []

    def guarded(file, mode="r", *args, **kwargs):
        if (not isinstance(file, int) and "r" in str(mode)
                and os.path.realpath(os.fspath(file)) == target
                and os.path.exists(target) and not (once and failed)):
            failed.append(file)
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), os.fspath(file))
        return real(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded)
    return failed


def _publication_fails(monkeypatch, path):
    """The rename that would publish a file at *path* fails, as a full disk
    or a vanished folder makes it; every other rename goes through."""
    target = os.path.realpath(path)
    real = os.replace

    def replace(src, dst, *args, **kwargs):
        if os.path.realpath(os.fspath(dst)) == target:
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), os.fspath(dst))
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)


# ---------------------------------------------------------------------------
# A stdout answer whose write fails
# ---------------------------------------------------------------------------

def test_a_stdout_answer_a_full_disk_cuts_is_never_published(plan, monkeypatch):
    """Codex's repro (U90b review 2, finding 1): the write lands a prefix and
    fails; the closing scan used to count the prefix, exit 0, and a resume
    reused it."""
    d = qdir(plan, "Q")
    with monkeypatch.context() as m:
        disk = _FullDisk().install(m)
        assert run_ask(ask_argv(plan, SPEC), _on_stdout(WHOLE)) == 3
    assert disk.failed == 2  # the answer, then its text as the partial
    assert not answer_of(d, SPEC).exists()
    assert not partial_of(d, SPEC).exists()
    assert _leftover_temps(d) == []
    status = read_status(d)
    assert status["models_completed"] == []
    assert status["models_failed"][SPEC].startswith(
        "Recovered answer but failed to persist it: [Errno 28]")
    assert PREFIX not in read_answers(d)

    again = FakeInvoker()
    again.configure(SPEC, write_file=True, content="The whole answer, written.")
    assert run_ask(ask_argv(plan, SPEC), again) == 0
    assert again.calls == [SPEC]
    assert "The whole answer, written." in read_answers(d)


def test_the_text_whose_write_failed_is_counted_only_when_accepted(plan, monkeypatch, capsys):
    """The disk fills for the answer only: the slot fails, its whole text is
    kept as the partial, and only --accept-partial counts it."""
    d = qdir(plan, "Q")
    with monkeypatch.context() as m:
        _FullDisk(times=1).install(m)
        assert run_ask(ask_argv(plan, SPEC), _on_stdout(WHOLE)) == 3
    assert not answer_of(d, SPEC).exists()
    assert partial_of(d, SPEC).read_text(encoding="utf-8") == WHOLE
    assert failed_line(d, SPEC).endswith(partial_note(SPEC))
    assert _records(d) == ({}, {})

    resumed = FakeInvoker()
    resumed.configure(SPEC, write_file=False, result=timeout_result())
    capsys.readouterr()
    assert run_ask(ask_argv(plan, SPEC, extra=("--accept-partial", SPEC)), resumed) == 0
    assert resumed.calls == []
    assert "WARNING" not in _printed(capsys)
    assert read_status(d)["models_completed"] == [SPEC]
    assert WHOLE in read_answers(d)


def test_a_slot_whose_write_fails_ends_failed_with_that_error(slot, monkeypatch):
    slot["invoke"] = lambda **_: {"success": True, "details": {"answer_text": WHOLE}}
    run = slot["make"]()
    with monkeypatch.context() as m:
        _FullDisk(times=1).install(m)
        result = _ask(run)
    assert result["success"] is False and result["source"] == "failed"
    assert result["error"].startswith("Recovered answer but failed to persist it:")
    assert result["checks_passed"] is True
    assert not slot["answer"].exists()
    assert slot["partial"].read_text(encoding="utf-8") == WHOLE
    assert not run.answers_left_in_place
    assert result_files.unchecked_models(slot["dir"]) == {}


def test_a_failed_write_that_leaves_text_in_place_stays_withheld(plan, monkeypatch):
    """A write that fails after part of its text landed (a stand-in for a
    writer that is not atomic), and a set-aside that fails for real: the
    text is withheld as unfinished until a later slot replaces it."""
    d = qdir(plan, "Q")
    real = ask_orchestrator.publish_atomically

    def lands_then_fails(path, text):
        if not os.path.basename(path).startswith("answer_"):
            return real(path, text)
        Path(path).write_text(text[:len(PREFIX)], encoding="utf-8")
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    blockers = (partial_of(d, SPEC), d / f"error_{STEM}.log")
    with monkeypatch.context() as m:
        m.setattr(ask_orchestrator, "publish_atomically", lands_then_fails)
        assert run_ask(ask_argv(plan, SPEC),
                       _blocking_during(_on_stdout(WHOLE), *blockers)) == 3
    assert answer_of(d, SPEC).read_text(encoding="utf-8") == PREFIX
    status = read_status(d)
    assert status["models_completed"] == []
    assert status[UNCHECKED] == {SPEC: result_files.UNFINISHED_ANSWER_REASON}
    assert "left in place" in failed_line(d, SPEC)
    assert f"\n{PREFIX}" not in read_answers(d)

    for blocker in blockers:
        _unblock(blocker)
    again = FakeInvoker()
    again.configure(SPEC, write_file=True, content="A finished answer")
    assert run_ask(ask_argv(plan, SPEC), again) == 0
    assert again.calls == [SPEC]
    assert Path(str(answer_of(d, SPEC)) + ".unchecked").read_text(
        encoding="utf-8") == PREFIX
    assert "A finished answer" in read_answers(d)
    assert _records(d) == ({}, {})


def _file_and_stdout(slot_or_dir):
    """A provider that writes ``Earlier file content`` to its answer file and
    ends cleanly with the fuller WHOLE on stdout."""
    def invoke(*args, **kwargs):
        Path(slot_or_dir()).write_text("Earlier file content", encoding="utf-8")
        return {"success": True, "details": {"answer_text": WHOLE}}

    return invoke


def test_a_file_left_beside_a_failed_publication_never_displaces_its_text(slot, monkeypatch):
    """Codex's repro (U90d review, finding 2): step 1 cannot read the file
    the model wrote, step 2's publication of the fuller stdout fails, and the
    file, readable again, used to move over the partial, losing the text."""
    slot["invoke"] = lambda **kw: _file_and_stdout(lambda: slot["answer"])()
    run = slot["make"]()
    with monkeypatch.context() as m:
        _publication_fails(m, slot["answer"])
        failed = _reads_fail(m, slot["answer"], once=True)
        result = _ask(run)
    assert len(failed) == 1
    assert result["success"] is False and "checks_passed" not in result
    assert result["error"].startswith("Recovered answer but failed to persist it:")
    assert slot["partial"].read_text(encoding="utf-8") == WHOLE
    # The file stays where it is, withheld as an unfinished answer.
    assert slot["answer"].read_text(encoding="utf-8") == "Earlier file content"
    assert run.answers_left_in_place == {SPEC}
    assert result_files.unchecked_models(slot["dir"]) == {
        SPEC: result_files.UNFINISHED_ANSWER_REASON}


def test_with_no_partial_written_the_file_left_takes_its_name(slot, monkeypatch):
    """The partial cannot be written either: the file is then the only text
    the run left, and moves to the partial's name as step 3 moves any."""
    slot["invoke"] = lambda **kw: _file_and_stdout(lambda: slot["answer"])()
    run = slot["make"]()
    real = ask_orchestrator._write_atomically

    def no_partial(path, text):
        if os.path.basename(path).startswith("partial_"):
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        return real(path, text)

    with monkeypatch.context() as m:
        _publication_fails(m, slot["answer"])
        _reads_fail(m, slot["answer"], once=True)
        m.setattr(ask_orchestrator, "_write_atomically", no_partial)
        result = _ask(run)
    assert result["success"] is False and result["checks_passed"] is True
    assert not slot["answer"].exists()
    assert slot["partial"].read_text(encoding="utf-8") == "Earlier file content"
    assert not run.answers_left_in_place


def test_the_text_a_failed_publication_kept_is_counted_once_the_file_is_moved(
    plan, monkeypatch, capsys,
):
    d = qdir(plan, "Q")
    with monkeypatch.context() as m:
        _publication_fails(m, answer_of(d, SPEC))
        _reads_fail(m, answer_of(d, SPEC), once=True)
        assert run_ask(ask_argv(plan, SPEC),
                       _file_and_stdout(lambda: answer_of(d, SPEC))) == 3
    assert partial_of(d, SPEC).read_text(encoding="utf-8") == WHOLE
    status = read_status(d)
    assert status["models_completed"] == []
    assert status[UNCHECKED] == {SPEC: result_files.UNFINISHED_ANSWER_REASON}
    assert "left in place" in failed_line(d, SPEC)
    assert "Earlier file content" not in read_answers(d)

    # --accept-partial refuses while the file is in place, then counts the
    # whole text, saying why it was withheld.
    assert run_ask(ask_argv(plan, SPEC, extra=("--accept-partial", SPEC)),
                   FakeInvoker()) == 3
    answer_of(d, SPEC).rename(d / "moved-aside.md")
    capsys.readouterr()
    assert run_ask(ask_argv(plan, SPEC, extra=("--accept-partial", SPEC)),
                   FakeInvoker()) == 0
    assert "left an unfinished answer" in _printed(capsys)
    assert answer_of(d, SPEC).read_text(encoding="utf-8") == WHOLE
    assert read_status(d)["models_completed"] == [SPEC]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_a_published_answer_keeps_the_mode_its_file_had(tmp_path):
    kept = tmp_path / "answer_kept.md"
    kept.write_text("", encoding="utf-8")
    os.chmod(kept, 0o640)
    ask_orchestrator.publish_atomically(str(kept), WHOLE)
    assert stat.S_IMODE(os.stat(kept).st_mode) == 0o640
    assert kept.read_text(encoding="utf-8") == WHOLE

    # A new file gets the mode a plain open() gives one.
    plain = tmp_path / "plain.md"
    with open(plain, "w", encoding="utf-8"):
        pass
    fresh = tmp_path / "answer_fresh.md"
    ask_orchestrator.publish_atomically(str(fresh), WHOLE)
    assert stat.S_IMODE(os.stat(fresh).st_mode) == stat.S_IMODE(os.stat(plain).st_mode)


def test_a_failed_answers_md_write_leaves_the_last_whole_one(tmp_path, monkeypatch):
    question_dir = tmp_path / "q"
    question_dir.mkdir()
    answer_of(question_dir, SPEC).write_text("An answer.", encoding="utf-8")
    path, completed = ask_orchestrator.write_answers_md(
        str(question_dir), "Q", "plan.md", [SPEC], {}, "2026-11-09 10:00:00")
    assert completed == [SPEC]
    before = Path(path).read_bytes()

    with monkeypatch.context() as m:
        _FullDisk().install(m)
        with pytest.raises(OSError) as raised:
            ask_orchestrator.write_answers_md(
                str(question_dir), f"Q {MARKER}", "plan.md", [SPEC], {},
                "2026-11-09 10:01:00")
    assert raised.value.errno == errno.ENOSPC
    assert Path(path).read_bytes() == before
    assert _leftover_temps(question_dir) == []


# ---------------------------------------------------------------------------
# What an earlier run left: removed before the dispatch, or no dispatch
# ---------------------------------------------------------------------------

def test_a_rerun_whose_earlier_answer_cannot_be_removed_is_never_dispatched(
    plan, monkeypatch, capsys,
):
    """Codex's repro (U90b review 2, finding 2): --rerun-all ignored a failed
    removal, and a slot that then wrote nothing counted the earlier answer
    as its own."""
    d = qdir(plan, "Q")
    first = FakeInvoker()
    first.configure(SPEC, write_file=True, content="Old answer")
    assert run_ask(ask_argv(plan, SPEC), first) == 0

    capsys.readouterr()
    with monkeypatch.context() as m:
        _removal_fails(m, answer_of(d, SPEC))
        inv = _answers_nothing()
        assert run_ask(ask_argv(plan, SPEC, extra=("--rerun-all",)), inv) == 3
    out = capsys.readouterr().out
    assert inv.calls == []
    assert f"--rerun-all could not remove answer_{STEM}.md" in out
    _assert_refused_rerun(d, f"--rerun-all could not remove answer_{STEM}.md")
    assert origin_of(d, SPEC).read_text(encoding="utf-8") == "file"

    # A resume keeps it, as an answer from an earlier run.
    resumed = FakeInvoker()
    assert run_ask(ask_argv(plan, SPEC), resumed) == 0
    assert resumed.calls == []
    assert provenance_of(d, SPEC) == FILE_LINE
    assert read_status(d)["answer_origins"]["kept"] == 1

    rerun = FakeInvoker()
    rerun.configure(SPEC, write_file=True, content="New answer")
    assert run_ask(ask_argv(plan, SPEC, extra=("--rerun-all",)), rerun) == 0
    assert rerun.calls == [SPEC]
    assert "New answer" in read_answers(d) and "Old answer" not in read_answers(d)


def test_a_partial_an_earlier_run_left_that_cannot_be_removed_is_never_asked_over(
    plan, monkeypatch,
):
    """The re-ask used to go ahead with a warning; failing with no text of
    its own, it left the earlier run's partial named as this run's."""
    d = qdir(plan, "Q")
    earlier = FakeInvoker()
    earlier.configure(SPEC, write_file=False, result=timeout_result(text="Earlier partial."))
    assert run_ask(ask_argv(plan, SPEC), earlier) == 3

    with monkeypatch.context() as m:
        _removal_fails(m, partial_of(d, SPEC))
        inv = FakeInvoker()
        inv.configure(SPEC, write_file=False, result=timeout_result())
        assert run_ask(ask_argv(plan, SPEC), inv) == 3
    assert inv.calls == []
    line = failed_line(d, SPEC)
    assert f"partial_{STEM}.md, the text an earlier run left, could not be removed" in line
    assert line.endswith(partial_note(SPEC, earlier=True))
    assert partial_of(d, SPEC).read_text(encoding="utf-8") == "Earlier partial."
    assert _records(d) == ({}, {})

    retry = FakeInvoker()
    retry.configure(SPEC, write_file=False, result=timeout_result())
    assert run_ask(ask_argv(plan, SPEC), retry) == 3
    assert retry.calls == [SPEC]
    assert not partial_of(d, SPEC).exists()


def test_a_refused_rerun_removes_nothing_of_the_earlier_answer(plan, monkeypatch):
    """The partial goes first: a slot refused over it leaves the earlier
    answer and its origin as they were."""
    d = qdir(plan, "Q")
    first = FakeInvoker()
    first.configure(SPEC, write_file=True, content="Old answer")
    assert run_ask(ask_argv(plan, SPEC), first) == 0
    partial_of(d, SPEC).write_text("A stray partial.", encoding="utf-8")

    with monkeypatch.context() as m:
        _removal_fails(m, partial_of(d, SPEC))
        inv = _answers_nothing()
        assert run_ask(ask_argv(plan, SPEC, extra=("--rerun-all",)), inv) == 3
    assert inv.calls == []
    _assert_refused_rerun(d, f"partial_{STEM}.md, the text an earlier run left, "
                             "could not be removed")
    assert origin_of(d, SPEC).read_text(encoding="utf-8") == "file"
    assert partial_of(d, SPEC).read_text(encoding="utf-8") == "A stray partial."


def test_a_rerun_counts_no_model_the_deadline_left_undispatched(plan, monkeypatch):
    """The fan-out's deadline cuts the first slot while the second still
    waits for its turn: the second's earlier answer is not this run's."""
    d = qdir(plan, "Q")
    first = FakeInvoker()
    first.configure(SPEC, write_file=True, content="Old answer")
    first.configure(OTHER, write_file=True, content="Other's old answer")
    assert run_ask(ask_argv(plan, SPEC, OTHER), first) == 0

    real = slot_guard.SlotGuard.finish

    async def finish(self, *args, **kwargs):
        await asyncio.sleep(10)
        return await real(self, *args, **kwargs)

    monkeypatch.setattr(slot_guard.SlotGuard, "finish", finish)
    monkeypatch.setattr("utils.concurrency.PROVIDER_STAGGER_DELAY", 0.0)
    rerun = FakeInvoker()
    rerun.configure(SPEC, write_file=True, content="Never checked")
    rerun.configure(OTHER, write_file=True, content="Never asked")
    extra = ("--rerun-all", "--timeout", "1", "--max-parallel", "1")
    assert run_ask(ask_argv(plan, SPEC, OTHER, extra=extra), rerun) == 3
    assert rerun.calls == [SPEC]
    status = read_status(d)
    assert status["models_completed"] == []
    assert status["models_failed"][OTHER] == f"{DEADLINE_ERROR} — {kept_note(OTHER)}"
    assert "Other's old answer" not in read_answers(d)
    assert answer_of(d, OTHER).read_text(encoding="utf-8") == "Other's old answer"
    assert origin_of(d, OTHER).read_text(encoding="utf-8") == "file"
    assert OTHER not in status.get(UNCHECKED, {})


@pytest.mark.parametrize("rerun_all", [True, False])
def test_an_origin_record_that_cannot_be_removed_stops_the_dispatch(
    plan, monkeypatch, rerun_all,
):
    """Codex's repro (U90d review, finding 4): the earlier origin record's
    removal fails, then so does the new answer's; the new answer used to be
    shown with the earlier origin."""
    d = qdir(plan, "Q")
    assert run_ask(ask_argv(plan, SPEC), _on_stdout("Old answer")) == 0
    if not rerun_all:
        # Moved out by hand: a resume asks again, over the record left.
        answer_of(d, SPEC).unlink()
    origin = origin_of(d, SPEC)
    real = ask_orchestrator._write_atomically

    def no_origin(path, text):
        if os.path.realpath(path) == os.path.realpath(origin):
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        return real(path, text)

    extra = ("--rerun-all",) if rerun_all else ()
    with monkeypatch.context() as m:
        _removal_fails(m, origin)
        m.setattr(ask_orchestrator, "_write_atomically", no_origin)
        inv = FakeInvoker()
        inv.configure(SPEC, write_file=True, content="Fresh file answer")
        assert run_ask(ask_argv(plan, SPEC, extra=extra), inv) == 3
    assert inv.calls == []
    assert f"origin_{STEM}.txt, the record of how an earlier answer was obtained, " \
           "could not be removed" in failed_line(d, SPEC)
    assert origin.read_text(encoding="utf-8") == "fallback"
    assert answer_of(d, SPEC).exists() is rerun_all
    assert read_status(d)["models_completed"] == []
    assert FALLBACK_LINE not in read_answers(d)
    assert _records(d) == ({}, {})

    retry = FakeInvoker()
    retry.configure(SPEC, write_file=True, content="Fresh file answer")
    assert run_ask(ask_argv(plan, SPEC, extra=extra), retry) == 0
    assert retry.calls == [SPEC]
    assert provenance_of(d, SPEC) == FILE_LINE


@pytest.mark.parametrize("rerun_all", [True, False], ids=["rerun-all", "plain"])
@pytest.mark.parametrize("refused", ["origin", "answer"])
def test_a_refused_preparation_keeps_the_earlier_answer(
    plan, monkeypatch, refused, rerun_all,
):
    """Codex's repro (U90d review 2, finding 1): --rerun-all removed the
    answer first, and a refusal over its origin record then left neither,
    with no provider call. Either refusal keeps the earlier answer and its
    origin, and a later resume finds it; a plain run's skip keeps it before
    any removal."""
    d = qdir(plan, "Q")
    assert run_ask(ask_argv(plan, SPEC), _on_stdout("Old answer")) == 0
    extra = ("--rerun-all",) if rerun_all else ()
    with monkeypatch.context() as m:
        _removal_fails(m, origin_of(d, SPEC) if refused == "origin" else answer_of(d, SPEC))
        inv = FakeInvoker()
        inv.configure(SPEC, write_file=True, content="New answer")
        code = run_ask(ask_argv(plan, SPEC, extra=extra), inv)
    assert inv.calls == []
    assert answer_of(d, SPEC).read_text(encoding="utf-8") == "Old answer"
    assert origin_of(d, SPEC).read_text(encoding="utf-8") == "fallback"
    if rerun_all:
        assert code == 3
        _assert_refused_rerun(d, {
            "origin": f"origin_{STEM}.txt, the record of how an earlier answer was "
                      "obtained, could not be removed",
            "answer": f"--rerun-all could not remove answer_{STEM}.md",
        }[refused])
    else:
        assert code == 0
        assert read_status(d)["progress"]["models"][SPEC]["source"] == "resume"

    resumed = FakeInvoker()
    assert run_ask(ask_argv(plan, SPEC), resumed) == 0
    assert resumed.calls == []
    assert "Old answer" in read_answers(d)
    assert provenance_of(d, SPEC) == FALLBACK_LINE
    assert read_status(d)["answer_origins"]["kept"] == 1


def test_an_origin_that_cannot_be_put_back_leaves_the_answer_unrecorded(
    plan, monkeypatch, capsys,
):
    """The answer's removal is refused after its origin record went, and
    writing that record again fails too: the answer is kept, and reads as
    not recorded rather than with another origin."""
    d = qdir(plan, "Q")
    assert run_ask(ask_argv(plan, SPEC), _on_stdout("Old answer")) == 0
    origin = origin_of(d, SPEC)
    real = ask_orchestrator._write_atomically

    def no_origin(path, text):
        if os.path.realpath(path) == os.path.realpath(origin):
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        return real(path, text)

    capsys.readouterr()
    with monkeypatch.context() as m:
        _removal_fails(m, answer_of(d, SPEC))
        m.setattr(ask_orchestrator, "_write_atomically", no_origin)
        inv = _answers_nothing()
        assert run_ask(ask_argv(plan, SPEC, extra=("--rerun-all",)), inv) == 3
    assert inv.calls == []
    assert f"could not record the origin of {SPEC}'s answer" in _printed(capsys)
    assert answer_of(d, SPEC).read_text(encoding="utf-8") == "Old answer"
    assert not origin.exists()

    resumed = FakeInvoker()
    assert run_ask(ask_argv(plan, SPEC), resumed) == 0
    assert resumed.calls == []
    assert provenance_of(d, SPEC) == ask_orchestrator.PROVENANCE_NOT_RECORDED


@pytest.mark.parametrize("which", ["answer", "partial"])
def test_only_a_missing_file_counts_as_removed(tmp_path, monkeypatch, which):
    if which == "answer":
        path = tmp_path / f"answer_{STEM}.md"

        def remove():
            return ask_orchestrator.remove_earlier_answer(SPEC, str(path))
    else:
        path = tmp_path / f"partial_{STEM}.md"

        def remove():
            return ask_orchestrator.remove_earlier_partial(str(tmp_path), SPEC)

    assert remove() is True
    path.write_text("Earlier.", encoding="utf-8")
    assert remove() is True and not path.exists()

    # A real error from the operating system: a directory is not removed.
    _block(path)
    assert remove() is False and path.is_dir()
    _unblock(path)

    path.write_text("Earlier.", encoding="utf-8")
    with monkeypatch.context() as m:
        _removal_fails(m, path)
        assert remove() is False
    assert path.read_text(encoding="utf-8") == "Earlier."


# ---------------------------------------------------------------------------
# A resume that cannot read the earlier answer
# ---------------------------------------------------------------------------

def test_a_resume_that_cannot_read_the_earlier_answer_asks_nothing_over_it(
    plan, monkeypatch,
):
    """The skip used to read an unreadable answer as no answer and ask the
    model again, removing its origin record on the way."""
    d = qdir(plan, "Q")
    assert run_ask(ask_argv(plan, SPEC), _on_stdout("Old answer")) == 0
    with monkeypatch.context() as m:
        _reads_fail(m, answer_of(d, SPEC))
        inv = _answers_nothing()
        assert run_ask(ask_argv(plan, SPEC), inv) == 3
    assert inv.calls == []
    assert f"answer_{STEM}.md, the answer an earlier run left, could not be read" \
        in failed_line(d, SPEC)
    status = read_status(d)
    assert status["progress"]["models"][SPEC]["source"] == "not-dispatched"
    assert answer_of(d, SPEC).read_text(encoding="utf-8") == "Old answer"
    assert origin_of(d, SPEC).read_text(encoding="utf-8") == "fallback"
    assert _records(d) == ({}, {})

    resumed = FakeInvoker()
    assert run_ask(ask_argv(plan, SPEC), resumed) == 0
    assert resumed.calls == []
    assert provenance_of(d, SPEC) == FALLBACK_LINE
    assert read_status(d)["answer_origins"]["kept"] == 1


@pytest.mark.parametrize("call", ["open", "stat"])
def test_a_transient_error_at_the_skip_never_makes_an_earlier_answer_new(
    plan, monkeypatch, call,
):
    """Codex's repro (U90d review, finding 3): one error at the skip, gone
    by step 1, used to dispatch the model, and a slot that wrote nothing
    counted the earlier answer as its own file, with a new origin."""
    d = qdir(plan, "Q")
    assert run_ask(ask_argv(plan, SPEC), _on_stdout("Old answer")) == 0
    target = str(answer_of(d, SPEC))
    real_skip = ask_orchestrator.AskRun.skip_existing_answer
    real_stat = os.stat
    failed = []

    def failing_stat(path, *args, **kwargs):
        if not isinstance(path, int) and os.fspath(path) == target and not failed:
            failed.append(path)
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES))
        return real_stat(path, *args, **kwargs)

    def skip(self, *args, **kwargs):
        with monkeypatch.context() as m:
            if call == "open":
                failed.extend(_reads_fail(m, target, once=True))
            else:
                m.setattr(os, "stat", failing_stat)
            return real_skip(self, *args, **kwargs)

    monkeypatch.setattr(ask_orchestrator.AskRun, "skip_existing_answer", skip)
    inv = _answers_nothing()
    assert run_ask(ask_argv(plan, SPEC), inv) == 0
    assert inv.calls == []
    status = read_status(d)
    assert status["progress"]["models"][SPEC]["source"] in ("not-dispatched", "resume")
    assert status["models_completed"] == [SPEC]
    assert status["answer_origins"]["kept"] == 1
    assert status["answer_origins"]["fallback"] == 1
    assert origin_of(d, SPEC).read_text(encoding="utf-8") == "fallback"
    assert provenance_of(d, SPEC) == FALLBACK_LINE


#: An answer whose bytes are not UTF-8 text (Latin-1's é).
NOT_UTF8 = b"Caf\xe9: an answer in Latin-1."


@pytest.mark.parametrize(
    "found", ["missing", "empty", "readable", "unreadable", "not-utf8"])
def test_the_resume_probe_tells_three_outcomes_apart(tmp_path, monkeypatch, found):
    path = tmp_path / "answer.md"
    if found == "empty":
        path.write_text("  \n", encoding="utf-8")
    elif found == "not-utf8":
        path.write_bytes(NOT_UTF8)
    elif found != "missing":
        path.write_text("An answer.", encoding="utf-8")
    with monkeypatch.context() as m:
        if found == "unreadable":
            _reads_fail(m, path)
        read = ask_orchestrator.probe_answer(str(path))
        text = ask_orchestrator.read_nonempty_answer(str(path))
    expected = {
        "missing": ask_orchestrator.ANSWER_MISSING,
        "empty": ask_orchestrator.ANSWER_MISSING,
        "readable": ask_orchestrator.ANSWER_READABLE,
        "unreadable": ask_orchestrator.ANSWER_UNREADABLE,
        "not-utf8": ask_orchestrator.ANSWER_UNREADABLE,
    }[found]
    assert read.state == expected
    assert read.text == text == ("An answer." if found == "readable" else None)
    assert (read.error is not None) == (found in ("unreadable", "not-utf8"))
    if found == "not-utf8":
        assert isinstance(read.error, UnicodeDecodeError)
        assert path.read_bytes() == NOT_UTF8


def test_a_resume_keeps_an_answer_that_is_not_utf8_and_collects_the_rest(plan):
    """Codex's finding (U90d review 2, finding 2): the reader raised
    ``UnicodeDecodeError`` on an earlier answer that is not UTF-8 text, and
    the resume stopped before any answer was collected."""
    d = qdir(plan, "Q")
    first = FakeInvoker()
    first.configure(SPEC, write_file=True, content="Old answer")
    first.configure(OTHER, write_file=True, content="Other's answer")
    assert run_ask(ask_argv(plan, SPEC, OTHER), first) == 0
    answer_of(d, SPEC).write_bytes(NOT_UTF8)

    resumed = FakeInvoker()
    assert run_ask(ask_argv(plan, SPEC, OTHER), resumed) == 0
    assert resumed.calls == []
    assert answer_of(d, SPEC).read_bytes() == NOT_UTF8
    assert origin_of(d, SPEC).read_text(encoding="utf-8") == "file"
    status = read_status(d)
    assert status["state"] == "partial"
    assert status["models_completed"] == [OTHER]
    assert status["progress"]["models"][SPEC]["source"] == "not-dispatched"
    assert (f"answer_{STEM}.md, the answer an earlier run left, could not be read "
            "('utf-8' codec can't decode") in failed_line(d, SPEC)
    assert "Other's answer" in read_answers(d)


def test_a_slot_that_leaves_an_answer_that_is_not_utf8_fails_alone(plan):
    """A model writes its answer file in another encoding: its slot fails
    with the file kept as it was, and the other model's answer is
    collected."""
    d = qdir(plan, "Q")
    inv = FakeInvoker()
    inv.configure(SPEC, write_file=False, result={"success": True, "details": {}})
    inv.configure(OTHER, write_file=True, content="Other's answer")

    def call(prompt, model_spec, *args, **kwargs):
        result = inv(prompt, model_spec, *args, **kwargs)
        if model_spec == SPEC:
            answer_of(d, SPEC).write_bytes(NOT_UTF8)
        return result

    assert run_ask(ask_argv(plan, SPEC, OTHER), call) == 0
    assert inv.calls == [SPEC, OTHER]
    assert answer_of(d, SPEC).read_bytes() == NOT_UTF8
    status = read_status(d)
    assert status["state"] == "partial"
    assert status["models_completed"] == [OTHER]
    assert SPEC in status["models_failed"]
    assert "Other's answer" in read_answers(d)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFOs")
def test_the_resume_probe_never_opens_a_fifo(tmp_path):
    """A path of size 0 is never opened: a read of a FIFO with no writer
    would block the run."""
    import threading

    path = tmp_path / "answer.md"
    os.mkfifo(path)
    seen = []
    probe = threading.Thread(
        target=lambda: seen.append(ask_orchestrator.probe_answer(str(path))), daemon=True)
    probe.start()
    probe.join(5)
    assert seen == [ask_orchestrator.AnswerRead(ask_orchestrator.ANSWER_MISSING)]


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

def test_a_healthy_stdout_answer_is_published_whole(plan):
    d = qdir(plan, "Q")
    assert run_ask(ask_argv(plan, SPEC), _on_stdout(WHOLE)) == 0
    assert answer_of(d, SPEC).read_text(encoding="utf-8") == WHOLE
    assert origin_of(d, SPEC).read_text(encoding="utf-8") == "fallback"
    assert read_status(d)["models_completed"] == [SPEC]
    assert WHOLE in read_answers(d)
    assert _leftover_temps(d) == []

    resumed = FakeInvoker()
    assert run_ask(ask_argv(plan, SPEC), resumed) == 0
    assert resumed.calls == []
    assert WHOLE in read_answers(d)


def test_a_healthy_rerun_replaces_the_earlier_answer(plan):
    d = qdir(plan, "Q")
    assert run_ask(ask_argv(plan, SPEC), _on_stdout("Old answer")) == 0
    partial_of(d, SPEC).write_text("A stray partial.", encoding="utf-8")
    rerun = FakeInvoker()
    rerun.configure(SPEC, write_file=True, content="New answer")
    assert run_ask(ask_argv(plan, SPEC, extra=("--rerun-all",)), rerun) == 0
    assert rerun.calls == [SPEC]
    assert answer_of(d, SPEC).read_text(encoding="utf-8") == "New answer"
    assert origin_of(d, SPEC).read_text(encoding="utf-8") == "file"
    assert not partial_of(d, SPEC).exists()
    assert "Old answer" not in read_answers(d)
    assert _records(d) == ({}, {})
