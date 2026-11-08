"""``utils.process_capture``: the bounded stream capture both spawn lanes share.

Real processes throughout: what a pipe holds, when it reaches end of file and
how much memory a capture keeps are questions only a real child answers. The
lane-level rows (a held pipe on each lane, a capped stream refused on each lane)
live beside each lane's own spawn tests in ``test_llm_client.py`` and
``test_subagent_executor.py``.
"""

import errno
import gc
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
import weakref
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import llm_client, process_capture  # noqa: E402

SKILL_DIR = Path(__file__).resolve().parents[1]

POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="POSIX shells and signals")

#: Both drain shapes run on POSIX: the selector thread (POSIX's own) and the
#: per-pipe blocking readers (the only shape Windows supports).
DRAINS = pytest.mark.parametrize(
    "selectable", [True, False], ids=["selector-drain", "reader-threads"],
)

#: Bytes that exercise every decode rule ``Popen.communicate`` applies in text
#: mode: CRLF and lone CR translation, an invalid byte, a multi-byte character,
#: and a CRLF split across two writes.
_AWKWARD = (
    b"line one\r\nline two\rline three\n"
    b"bad \xff byte, caf\xc3\xa9, emoji \xf0\x9f\x98\x80\r"
)


def _spawn(code, **kwargs):
    return subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=kwargs.pop("stdin", subprocess.DEVNULL),
        text=True, encoding="utf-8", errors="replace", **kwargs,
    )


def _capture(proc, selectable, **kwargs):
    return process_capture.PipeCapture(proc, selectable=selectable, **kwargs)


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


@DRAINS
def test_below_the_bound_the_text_is_what_communicate_returns(selectable):
    code = textwrap.dedent(f"""\
        import sys, time
        data = {_AWKWARD!r}
        for stream in (sys.stdout.buffer, sys.stderr.buffer):
            stream.write(data[:9]); stream.flush()   # ends inside the first CRLF
            time.sleep(0.05)
            stream.write(data[9:]); stream.flush()
        """)
    expected = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, timeout=30,
    )
    proc = _spawn(code)
    capture = _capture(proc, selectable)
    try:
        out, err = capture.communicate(timeout=30)
    finally:
        capture.close()
    assert (out, err) == (expected.stdout, expected.stderr)
    assert proc.returncode == 0
    assert capture.capped == frozenset() and not capture.pipe_held


@DRAINS
def test_stdin_input_is_written_whole_and_closed(selectable):
    text = "y" * 300_000 + "\nEND"
    proc = _spawn(
        "import sys; data = sys.stdin.read(); print(len(data), data[-3:])",
        stdin=subprocess.PIPE,
    )
    capture = _capture(proc, selectable)
    try:
        out, _err = capture.communicate(input=text, timeout=30)
    finally:
        capture.close()
    assert out.split() == [str(len(text)), "END"]


@DRAINS
def test_a_stream_past_the_bound_keeps_its_head_and_is_named_capped(selectable, capsys):
    proc = _spawn(
        "import sys; sys.stdout.write('a' * 1000 + 'b' * 9000); sys.stderr.write('tail')"
    )
    capture = _capture(proc, selectable, max_bytes=1000)
    try:
        out, err = capture.communicate(timeout=30)
    finally:
        capture.close()
    assert out == "a" * 1000
    assert err == "tail"
    assert capture.capped == frozenset({"stdout"})
    line = capsys.readouterr().err
    assert "[STREAM_CAP]" in line and "10000 bytes on stdout" in line


@POSIX_ONLY
@DRAINS
def test_an_exit_with_a_held_pipe_returns_after_the_grace_with_the_answer(
    selectable, tmp_path, capsys
):
    """PROV-4: the CLI exits with its answer while a process it started keeps
    the inherited stdout open. The wait ends with the CLI, not with that
    process, and the process is left running."""
    marker = tmp_path / "holder.pid"
    proc = subprocess.Popen(
        ["sh", "-c", f'echo "[{{\\"answer\\":1}}]"; sleep 30 & echo $! > "{marker}"; exit 0'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace",
    )
    capture = _capture(proc, selectable, grace_s=0.5)
    holder = None
    try:
        started = time.monotonic()
        out, err = capture.communicate(timeout=60)
        elapsed = time.monotonic() - started
        holder = int(marker.read_text(encoding="utf-8").strip())
        assert elapsed < 5.0, f"waited {elapsed:.1f}s on the held pipe"
        assert out == '[{"answer":1}]\n' and err == ""
        assert proc.returncode == 0
        assert capture.pipe_held
        assert _pid_alive(holder), "a normal exit must not stop what the CLI started"
        assert "[PIPE_HELD]" in capsys.readouterr().err
    finally:
        capture.close()
        if holder is not None:
            os.kill(holder, signal.SIGKILL)


@POSIX_ONLY
@DRAINS
def test_a_timeout_carries_the_output_so_far_and_a_second_call_collects_it(selectable):
    proc = _spawn(
        "import sys, time; print('partial', flush=True); time.sleep(30)",
        start_new_session=True,
    )
    capture = _capture(proc, selectable)
    try:
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            capture.communicate(timeout=1.0)
        assert caught.value.stdout == "partial\n"
        os.killpg(proc.pid, signal.SIGKILL)
        out, _err = capture.communicate(timeout=5.0)
        assert out == "partial\n"
        assert proc.returncode == -signal.SIGKILL
    finally:
        capture.close()


def _wait_for(predicate, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _capture_threads():
    return [t for t in threading.enumerate() if t.name.startswith("multi-llm-")]


def _kill(pid):
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


#: A CLI that starts a background writer on its own stdout and stderr and exits
#: 0 at once. The writer waits past the capture's grace, writes more than a
#: pipe buffer holds, and records whether the write went through.
_DELAYED_WRITER = textwrap.dedent("""\
    import subprocess, sys
    writer = (
        "import os, sys, time\\n"
        "time.sleep(1.0)\\n"
        "try:\\n"
        "    os.write(1, b'late\\\\n' * 50000)\\n"
        "    os.write(2, b'late\\\\n' * 50000)\\n"
        "    result = 'ok'\\n"
        "except BrokenPipeError:\\n"
        "    result = 'epipe'\\n"
        "open(sys.argv[1], 'w').write(result)\\n"
    )
    child = subprocess.Popen([sys.executable, "-c", writer, sys.argv[1]])
    open(sys.argv[2], "w").write(str(child.pid))
    print("answer", flush=True)
    """)


@POSIX_ONLY
@DRAINS
def test_a_survivor_that_writes_after_the_grace_is_not_cut_off(selectable, tmp_path):
    """A held pipe is never closed under its writer: a process the CLI left
    running that writes after the capture returned (more than a pipe buffer)
    neither blocks nor dies of SIGPIPE/EPIPE. The drain reads and drops it."""
    flag, marker = tmp_path / "write.result", tmp_path / "writer.pid"
    proc = subprocess.Popen(
        [sys.executable, "-c", _DELAYED_WRITER, str(flag), str(marker)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace",
    )
    capture = _capture(proc, selectable, grace_s=0.3)
    writer = None
    try:
        out, _err = capture.communicate(timeout=30)
        writer = int(marker.read_text(encoding="utf-8"))
        assert out == "answer\n" and proc.returncode == 0
        assert capture.pipe_held
        capture.close()
        assert _wait_for(flag.exists), "the survivor's write never finished"
        assert flag.read_text(encoding="utf-8") == "ok"
        # What it wrote after the capture let go was dropped, not kept.
        assert capture._sinks["stdout"].snapshot() == b"answer\n"
    finally:
        capture.close()
        if writer is not None:
            _kill(writer)


#: A CLI that starts a background child on its own stdin (which never reads
#: it), prints its answer and exits 0 without reading stdin either.
_STDIN_HOLDER = textwrap.dedent("""\
    import subprocess, sys
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    open(sys.argv[1], "w").write(str(child.pid))
    print("answer", flush=True)
    """)


@POSIX_ONLY
@DRAINS
def test_an_exit_while_a_survivor_holds_stdin_unread_returns_at_once(selectable, tmp_path):
    """The wait follows the CLI, not stdin delivery: a 1 MB prompt that a
    background child holds unread (the pipe stays full) does not keep a CLI
    that exited 0 waiting to the deadline."""
    marker = tmp_path / "holder.pid"
    proc = _spawn_with_stdin(_STDIN_HOLDER, str(marker))
    capture = _capture(proc, selectable, grace_s=1.0)
    holder = None
    try:
        started = time.monotonic()
        out, _err = capture.communicate(input="p" * 1_000_000, timeout=8)
        elapsed = time.monotonic() - started
        holder = int(marker.read_text(encoding="utf-8"))
        assert out == "answer\n" and proc.returncode == 0
        assert not capture.pipe_held
        assert elapsed < 4, f"{elapsed:.1f}s: the wait followed stdin, not the CLI"
        if selectable:
            # POSIX stops the feed once the CLI is gone, and closes its end.
            assert _wait_for(lambda: not any(
                t.name == "multi-llm-stdin-feed" for t in threading.enumerate()
            ), timeout=5)
    finally:
        capture.close()
        if holder is not None:
            _kill(holder)


def _spawn_with_stdin(code, *args, encoding="utf-8", errors="replace"):
    return subprocess.Popen(
        [sys.executable, "-c", code, *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE,
        text=True, encoding=encoding, errors=errors,
    )


#: Runs a few calls whose CLI exits while a background child holds its stdin
#: unread, each with a fresh multi-MiB prompt, and reports what the abandoned
#: feeds still hold in Python memory with every capture and ``Popen`` kept.
_STDIN_RETENTION_PROBE = textwrap.dedent("""\
    import gc, json, os, signal, subprocess, sys, threading, time, tracemalloc
    sys.path.insert(0, sys.argv[1])
    from utils import process_capture
    selectable, calls, prompt_bytes = sys.argv[2] == "1", int(sys.argv[3]), int(sys.argv[4])
    holder_cli, marker_dir = sys.argv[5], sys.argv[6]
    failures = []
    threading.excepthook = lambda args: failures.append(repr(args.exc_value))
    feeds = lambda: [t for t in threading.enumerate() if t.name == "multi-llm-stdin-feed"]
    kept, holders = [], []
    tracemalloc.start()
    gc.collect()
    baseline = tracemalloc.get_traced_memory()[0]
    for n in range(calls):
        marker = os.path.join(marker_dir, f"holder-{n}.pid")
        proc = subprocess.Popen(
            [sys.executable, "-c", holder_cli, marker],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        capture = process_capture.PipeCapture(proc, selectable=selectable, grace_s=1.0)
        prompt = chr(ord("a") + n) * prompt_bytes
        out, _err = capture.communicate(input=prompt, timeout=30)
        del prompt
        assert out == "answer\\n", out
        holders.append(int(open(marker).read()))
        kept.append((capture, proc))
    gc.collect()
    retained = tracemalloc.get_traced_memory()[0] - baseline
    def settle(seconds):
        deadline = time.monotonic() + seconds
        while feeds() and time.monotonic() < deadline:
            time.sleep(0.05)
        return len(feeds())
    stuck = settle(2)  # the holders still alive
    for pid in holders:
        os.kill(pid, signal.SIGKILL)
    left = settle(15)
    for capture, _proc in kept:
        capture.close()
    print(json.dumps({"retained": retained, "stuck": stuck, "left": left,
                      "failures": failures, "holders": holders}))
    """)


@POSIX_ONLY
@DRAINS
def test_abandoned_stdin_feeds_hold_one_chunk_not_the_prompt(selectable, tmp_path):
    """Once the CLI has exited, its capture lets go of the prompt and its
    encoded bytes even while a feed is still stuck writing to a pipe a survivor
    never reads: the stuck write holds one chunk. Repeated calls do not add up
    to their prompts (the reader-threads shape is the Windows one)."""
    calls, prompt_bytes = 6, 2 << 20
    probe = subprocess.run(
        [sys.executable, "-c", _STDIN_RETENTION_PROBE, str(SKILL_DIR),
         "1" if selectable else "0", str(calls), str(prompt_bytes),
         _STDIN_HOLDER, str(tmp_path)],
        capture_output=True, text=True, timeout=120,
    )
    try:
        assert probe.returncode == 0, probe.stderr
        report = json.loads(probe.stdout.strip().splitlines()[-1])
    finally:
        for marker in tmp_path.glob("holder-*.pid"):
            _kill(int(marker.read_text(encoding="utf-8")))
    budget = calls * (process_capture._CHUNK + (64 << 10))
    assert report["retained"] < budget, (
        f"{report['retained']} B retained after {calls} calls with "
        f"{prompt_bytes} B prompts (budget {budget} B)")
    # POSIX's feed stops on its own; a blocking write stays stuck, with its
    # one chunk and its descriptor, until the holder goes.
    assert report["stuck"] == (0 if selectable else calls)
    # The holders gone, every feed ends on the broken pipe, quietly.
    assert report["left"] == 0
    assert report["failures"] == []


@DRAINS
def test_an_input_the_stream_cannot_encode_raises_in_the_caller(selectable):
    """Stdin is encoded before any feed starts, so an encoding error reaches
    the caller, as ``Popen.communicate`` raises it, instead of the child
    getting an empty prompt and the call succeeding."""
    proc = _spawn_with_stdin(
        "import sys; print(len(sys.stdin.buffer.read()))",
        encoding="ascii", errors="strict",
    )
    capture = _capture(proc, selectable)
    try:
        with pytest.raises(UnicodeEncodeError):
            capture.communicate(input="price: €", timeout=30)
        assert not any(t.name == "multi-llm-stdin-feed" for t in threading.enumerate())
    finally:
        capture.close()
        proc.kill()
        proc.wait(timeout=30)


class _WritesFail:
    """``process_capture``'s view of ``os`` with ``os.write`` raising *error*."""

    def __init__(self, error, linesep=os.linesep):
        self._error = error
        self.linesep = linesep

    def __getattr__(self, name):
        return getattr(os, name)

    def write(self, fd, data):
        if self._error is None:
            return os.write(fd, data)
        raise self._error


#: (error, raised by the selector feed, raised by the reader-threads feed):
#: the feed ignores what ``Popen.communicate`` ignores on each path (the
#: selector loop only ``BrokenPipeError``; ``Popen._stdin_write`` also EINVAL,
#: Windows' broken pipe) and hands the caller anything else.
_FEED_ERRORS = [
    pytest.param(OSError(errno.EIO, "I/O error"), True, True, id="EIO"),
    pytest.param(BrokenPipeError(errno.EPIPE, "Broken pipe"), False, False, id="EPIPE"),
    pytest.param(OSError(errno.EINVAL, "Invalid argument"), True, False, id="EINVAL"),
]


@DRAINS
@pytest.mark.parametrize("error, by_selector, by_threads", _FEED_ERRORS)
def test_a_stdin_write_error_reaches_the_caller_unless_communicate_ignores_it(
    selectable, error, by_selector, by_threads, monkeypatch
):
    monkeypatch.setattr(process_capture, "os", _WritesFail(error))
    proc = _spawn_with_stdin("import sys; print(len(sys.stdin.buffer.read()))")
    capture = _capture(proc, selectable)
    try:
        if by_selector if selectable else by_threads:
            with pytest.raises(type(error)) as caught:
                capture.communicate(input="prompt", timeout=30)
            assert caught.value is error
            # Raised once: the lane's second call (its drain) collects output.
            out, _err = capture.communicate(timeout=30)
        else:
            out, _err = capture.communicate(input="prompt", timeout=30)
        # The failed feed closed stdin, so the child read an empty prompt.
        assert out == "0\n" and proc.returncode == 0
    finally:
        capture.close()


@DRAINS
def test_stdin_text_is_written_as_communicate_writes_it(selectable, monkeypatch):
    """The selector feed sends ``str.encode`` of the input, as POSIX
    ``Popen.communicate`` does; the reader-threads feed sends what the text
    stdin's ``TextIOWrapper.write`` would, with ``\\n`` turned into
    ``os.linesep`` (``\\r\\n`` on Windows, simulated here)."""
    monkeypatch.setattr(process_capture, "os", _WritesFail(None, linesep="\r\n"))
    proc = _spawn_with_stdin(
        "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read().hex().encode('utf-16'))",
        encoding="utf-16",
    )
    capture = _capture(proc, selectable)
    try:
        out, _err = capture.communicate(input="a\nb€", timeout=30)
    finally:
        capture.close()
    sent = "a\nb€" if selectable else "a\r\nb€"
    assert bytes.fromhex(out.strip()) == sent.encode("utf-16")


@POSIX_ONLY
@DRAINS
def test_a_closed_capture_is_freed_while_its_drain_still_runs(selectable, tmp_path):
    """After ``close()`` a drain still reading a held pipe keeps nothing of the
    capture: not the capture, not its buffers, not the ``Popen``. The
    reader-threads shape is the Windows one, run here on POSIX."""
    marker = tmp_path / "holder.pid"
    proc = subprocess.Popen(
        ["sh", "-c", f'head -c 300000 /dev/zero; sleep 30 & echo $! > "{marker}"; exit 0'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
    )
    capture = _capture(proc, selectable, grace_s=0.3)
    holder = None
    try:
        out, _err = capture.communicate(timeout=30)
        holder = int(marker.read_text(encoding="utf-8").strip())
        assert len(out) == 300000 and capture.pipe_held
        capture.close()
        refs = [weakref.ref(capture), weakref.ref(proc)]
        refs += [weakref.ref(sink) for sink in capture._sinks.values()]
        del capture, proc, out
        gc.collect()
        assert _capture_threads(), "the held pipe's drain should still be reading"
        assert [ref() for ref in refs] == [None] * len(refs), (
            "a surviving drain kept the capture, its buffers or the Popen alive")
    finally:
        if holder is not None:
            _kill(holder)
    # The holder gone, the drain reaches end of file and ends.
    assert _wait_for(lambda: not _capture_threads(), timeout=10)


def test_a_stand_in_without_pipes_answers_through_its_own_communicate():
    """A ``Popen`` stand-in has nothing to take; its ``communicate`` is the
    answer, its ``TimeoutExpired`` too."""

    class StandIn:
        stdin = stdout = stderr = None
        pid = -1
        returncode = None
        calls = []

        def communicate(self, input=None, timeout=None):
            self.calls.append((input, timeout))
            if len(self.calls) == 1:
                raise subprocess.TimeoutExpired("cli", timeout, output=b"so far", stderr=b"")
            self.returncode = 0
            return "out", "err"

    child = StandIn()
    capture = process_capture.PipeCapture(child)
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        capture.communicate(input="prompt", timeout=3)
    assert caught.value.stdout == b"so far"
    assert capture.communicate(timeout=5) == ("out", "err")
    assert child.calls == [("prompt", 3), (None, 5)]
    assert capture.capped == frozenset()
    capture.close()


_MEMORY_PROBE = textwrap.dedent("""\
    import json, subprocess, sys, tracemalloc
    sys.path.insert(0, sys.argv[1])
    from utils import llm_client
    bound, total = int(sys.argv[2]), int(sys.argv[3])
    llm_client.LANE_A_MAX_STREAM_BYTES = bound
    writer = (
        "import sys\\n"
        "chunk = b'z' * (1 << 20)\\n"
        f"for _ in range({total} >> 20):\\n"
        "    sys.stdout.buffer.write(chunk)\\n"
    )
    tracemalloc.start()
    completed = llm_client._run_provider_cli(
        [sys.executable, "-c", writer], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=120, stdin=subprocess.DEVNULL,
    )
    _current, peak = tracemalloc.get_traced_memory()
    print(json.dumps({
        "peak": peak, "kept": len(completed.stdout),
        "capped": sorted(completed.capped_streams),
        "returncode": completed.returncode,
    }))
    """)


def test_memory_stays_bounded_when_a_child_writes_past_the_bound():
    """PROV-11: a child writing 128 MiB through Lane A's real spawn costs about
    the 4 MiB bound, measured in a fresh interpreter with ``tracemalloc``.
    ``communicate`` held all of it, about three times over."""
    bound, total = 4 << 20, 128 << 20
    probe = subprocess.run(
        [sys.executable, "-c", _MEMORY_PROBE, str(SKILL_DIR), str(bound), str(total)],
        capture_output=True, text=True, timeout=300,
    )
    assert probe.returncode == 0, probe.stderr
    report = json.loads(probe.stdout.strip().splitlines()[-1])
    assert report["returncode"] == 0
    assert report["kept"] == bound
    assert report["capped"] == ["stdout"]
    assert report["peak"] < 6 * bound, (
        f"peak {report['peak']} B for a {bound} B bound and a {total} B stream")


def test_a_capped_stdout_is_never_audited_clean():
    """The read-only audit judges a whole stream; a capped one is not clean,
    on the rungs the audit serves, whatever its head shows."""
    from utils import provider_registry as registry

    provider = registry.get_provider("codex")
    stream = "\n".join(json.dumps(event) for event in (
        {"type": "thread.started", "thread_id": "t"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {
            "id": "i0", "type": "agent_message", "text": "[]"}},
        {"type": "turn.completed", "usage": {}},
    ))
    whole = llm_client._readonly_audit(provider, stream, True)
    assert whole is not None and whole["clean"] is True, whole
    capped = llm_client._readonly_audit(provider, stream, True, complete=False)
    assert capped["clean"] is False and "capture bound" in capped["reason"]
    # A default run carries no verdict either way.
    assert llm_client._readonly_audit(provider, stream, None, complete=False) is None
