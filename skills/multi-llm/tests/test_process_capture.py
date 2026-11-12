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
from tests.support.own_session import SETSID  # noqa: E402

SKILL_DIR = Path(__file__).resolve().parents[1]

POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="POSIX shells and signals")

#: Both drain shapes run on POSIX: the selector thread (POSIX's own) and the
#: per-pipe blocking readers (the only shape Windows supports). The selector
#: drain does not run on Windows: ``select`` takes sockets only there, never a
#: pipe (``WinError 10038``), and ``PipeCapture`` never picks it there
#: (``selectable=None`` is ``os.name != "nt"``, and no lane passes one).
DRAINS = pytest.mark.parametrize("selectable", [
    pytest.param(True, id="selector-drain", marks=pytest.mark.skipif(
        os.name == "nt", reason="the POSIX drain: Windows selects sockets only, "
                                "and PipeCapture never uses it there")),
    pytest.param(False, id="reader-threads"),
])

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


def _holder_cli(marker, *, own_session=False):
    """A CLI that prints its answer and exits 0 while a ``sleep`` it started
    keeps its stdout and stderr open; the sleep's pid goes to *marker*."""
    start = f"{SETSID} sleep 30" if own_session else "sleep 30"
    return subprocess.Popen(
        ["sh", "-c", f'echo "[{{\\"answer\\":1}}]"; {start} & echo $! > "{marker}"; exit 0'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace", start_new_session=True,
    )


@POSIX_ONLY
@DRAINS
def test_a_held_pipe_is_handed_to_the_lanes_stop_while_the_cli_is_unreaped(
    selectable, tmp_path, capsys
):
    """SP-1: with ``on_held`` the capture calls the lane's stop once the grace
    passed with a pipe held, before the dispatch returns, and keeps the CLI
    unreaped until then (its zombie still holds its pid and its group's
    number for the stop). The answer and the real exit code are kept."""
    marker = tmp_path / "holder.pid"
    proc = _holder_cli(marker)
    seen = {}

    def stop(held):
        seen["capture"] = held
        seen["returncode"] = proc.returncode
        seen["zombie"] = os.waitid(
            os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
        seen["holder"] = int(marker.read_text(encoding="utf-8").strip())
        os.killpg(proc.pid, signal.SIGKILL)
        return "stopped"

    capture = _capture(proc, selectable, grace_s=0.5, on_held=stop)
    try:
        out, err = capture.communicate(timeout=60)
        assert out == '[{"answer":1}]\n' and err == ""
        assert proc.returncode == 0, "reaped after the stop, with its own exit"
        assert seen["capture"] is capture and seen["returncode"] is None
        assert seen["zombie"], "the stop ran on a reaped CLI"
        assert capture.pipe_held and capture.held_stopped
        assert capture.held_stop == "stopped"
        assert capture.still_held == frozenset()
        assert _wait_for(lambda: not _pid_alive(seen["holder"]))
        assert "stopped what it left running" in capsys.readouterr().err
    finally:
        capture.close()
        _stop_group(proc)


@POSIX_ONLY
@DRAINS
def test_a_holder_the_lanes_stop_does_not_reach_is_named(selectable, tmp_path, capsys):
    """The pipes are watched once the stop returned: one still held is held
    by a process the stop did not reach, and the capture names it."""
    marker = tmp_path / "holder.pid"
    proc = _holder_cli(marker)
    holder = None
    capture = _capture(proc, selectable, grace_s=0.3, on_held=lambda _held: "missed")
    try:
        out, _err = capture.communicate(timeout=60)
        holder = int(marker.read_text(encoding="utf-8").strip())
        assert out == '[{"answer":1}]\n' and proc.returncode == 0
        assert capture.held_stop == "missed"
        assert capture.still_held == frozenset({"stdout", "stderr"})
        assert _pid_alive(holder)
        assert "still held by a process that stop did not reach" in capsys.readouterr().err
    finally:
        capture.close()
        _stop_group(proc)


@POSIX_ONLY
def test_the_lanes_stop_runs_once_and_never_on_a_clean_exit(tmp_path):
    """No held pipe, no stop; and a call after a timeout (the lane's own stop
    is under way) never calls it either."""
    calls = []
    clean = _spawn("print('answer')")
    capture = _capture(clean, True, on_held=calls.append)
    try:
        assert capture.communicate(timeout=30) == ("answer\n", "")
        assert clean.returncode == 0 and not capture.pipe_held
    finally:
        capture.close()
    slow = _spawn("import time; print('x', flush=True); time.sleep(30)",
                  start_new_session=True)
    capture = _capture(slow, True, grace_s=0.3, on_held=calls.append)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            capture.communicate(timeout=0.5)
        os.killpg(slow.pid, signal.SIGKILL)
        capture.communicate(timeout=5)
    finally:
        capture.close()
    assert calls == []


#: A helper in a session of its own: holds the CLI's output for *hold* seconds,
#: closes it, then writes *late* a second later (the residual: beyond reach). It
#: leaves the CLI's group itself (``os.setsid``), on every POSIX: macOS has no
#: ``setsid(1)``, and a second interpreter in front (``SETSID``) would eat into
#: the window between the lane's stop and the helper closing the pipe.
_CLOSING_HELPER = (
    "import os, sys, time; os.setsid(); time.sleep(float(sys.argv[2])); os.close(1); "
    "os.close(2); time.sleep(1); open(sys.argv[1], 'w').write('late')"
)


@POSIX_ONLY
@pytest.mark.parametrize("lane", ["executor", "lane-a"])
def test_a_helper_that_leaves_the_group_and_closes_its_pipes_is_never_signalled(
    tmp_path, monkeypatch, lane
):
    """Codex re-check 3 (``closes_pipe.py``), under the coordinator's design:
    a helper outside the run's group keeps the output past the grace, then
    closes it. The run's own group is stopped, the pipe is let go, and the
    dispatch goes on with the answer; the helper, outside the run's group, is
    never signalled, and its late write lands: the residual the doc names
    (beyond the supervisor's reach, as at the base)."""
    from utils import dispatch_spawn
    from utils.providers.codex import CodexProvider

    late = tmp_path / "late"
    script = (f"{_shell_quote(sys.executable)} -c {_shell_quote(_CLOSING_HELPER)} "
              f"{_shell_quote(str(late))} 0.75 & echo answer; exit 0")
    if lane == "executor":
        provider = CodexProvider()
        monkeypatch.setattr(type(provider), "build_agent_command_for_transport",
                            lambda self, *a, **k: (["sh", "-c", script], None))
        monkeypatch.setattr(dispatch_spawn, "TIMEOUT_DRAIN_S", 0.5)
        monkeypatch.setattr(dispatch_spawn, "TIMEOUT_KILL_GRACE_S", 0.1)
        capture = dispatch_spawn._spawn_in_context(
            provider=provider, model="m", prompt="p", profile="agent-write",
            transport="argv", timeout=30, cwd=str(tmp_path))
        assert not capture.timed_out and capture.exit_code == 0
        assert capture.stdout == "answer\n" and capture.pipe_held
    else:
        monkeypatch.setattr(llm_client, "LANE_A_DRAIN_S", 0.5)
        monkeypatch.setattr(llm_client, "LANE_A_TERMINATE_GRACE_S", 0.1)
        completed = llm_client._run_provider_cli(
            ["sh", "-c", script], capture_output=True, text=True, timeout=30,
            stdin=subprocess.DEVNULL)
        assert completed.returncode == 0 and completed.stdout == "answer\n"
        assert "swept before the slot's checks" in completed.pipe_held
    assert _wait_for(late.exists, timeout=10), "the helper outside the group was stopped"


def _shell_quote(text):
    import shlex

    return shlex.quote(text)


@POSIX_ONLY
def test_an_unrelated_reader_of_the_pipe_is_never_signalled(tmp_path):
    """Codex re-check 2 (``fd_reuse.py``), under the coordinator's design: a
    process outside the run (here this one's own fork) holds the pipe's read
    end, then reuses its number for a writable file. Nothing reads other
    processes' descriptors any more: the held pipe's stop is the run's group
    only, so the reader is never signalled, and the survivor in the group is."""
    from utils import dispatch_spawn

    marker = tmp_path / "holder.pid"
    proc = _holder_cli(marker)
    read_fd = proc.stdout.fileno()
    reader = os.fork()
    if reader == 0:  # pragma: no cover - the forked reader
        try:
            null = os.open(os.devnull, os.O_WRONLY)
            os.dup2(null, read_fd)
            time.sleep(15)
        finally:
            os._exit(0)
    capture = _capture(proc, True, grace_s=0.3,
                       on_held=lambda _held: dispatch_spawn._stop_held_group(proc, proc.pid))
    try:
        out, _err = capture.communicate(timeout=30)
        holder = int(marker.read_text(encoding="utf-8").strip())
        assert out == '[{"answer":1}]\n' and proc.returncode == 0
        assert capture.held_stopped and capture.held_stop == "killed"
        assert capture.still_held == frozenset()
        assert _wait_for(lambda: not _pid_alive(holder)), "the survivor in the group ran on"
        assert os.waitpid(reader, os.WNOHANG) == (0, 0), "the unrelated reader was signalled"
    finally:
        capture.close()
        _kill(reader)
        os.waitpid(reader, 0)
        _stop_group(proc)


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


#: The background writer: waits for a go file the test creates only after the
#: capture has returned and closed (an event, not a sleep, so machine load
#: cannot move the write before the grace), then writes more than a pipe buffer
#: holds to each stream, looping until every byte is in (one ``os.write`` may
#: take only part of it, so a single call proves nothing about the whole
#: payload), and records whether that went through (published whole by a
#: rename, so the flag never exists empty).
_LATE_WRITER = textwrap.dedent("""\
    import os, sys, time

    def put(fd, data):
        view = memoryview(data)
        while view:
            try:
                view = view[os.write(fd, view):]
            except BlockingIOError:
                time.sleep(0.01)

    deadline = time.monotonic() + 60
    while not os.path.exists(sys.argv[2]) and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        put(1, b"late\\n" * 50000)
        put(2, b"late\\n" * 50000)
        result = "ok"
    except BrokenPipeError:
        result = "epipe"
    open(sys.argv[1] + ".tmp", "w").write(result)
    os.replace(sys.argv[1] + ".tmp", sys.argv[1])
    """)

#: A CLI that starts that writer on its own stdout and stderr and exits 0 at
#: once.
_DELAYED_WRITER = (
    "import subprocess, sys\n"
    f"writer = {_LATE_WRITER!r}\n"
    + textwrap.dedent("""\
        child = subprocess.Popen([sys.executable, "-c", writer, sys.argv[1], sys.argv[3]])
        open(sys.argv[2], "w").write(str(child.pid))
        print("answer", flush=True)
        """)
)


def _stop_group(proc, *, timeout=15.0):
    """End a test subprocess and everything it started: it is its own process
    group leader (``start_new_session``), so the group takes the background
    writer with it even when the writer's pid was never read."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)


@POSIX_ONLY
@DRAINS
def test_a_survivor_that_writes_after_the_grace_is_not_cut_off(selectable, tmp_path):
    """A held pipe is never closed under its writer: a process the CLI left
    running that writes after the capture returned (more than a pipe buffer)
    neither blocks nor dies of SIGPIPE/EPIPE. The drain reads and drops it."""
    flag, marker = tmp_path / "write.result", tmp_path / "writer.pid"
    go = tmp_path / "go"
    threads_before = set(_capture_threads())
    proc = subprocess.Popen(
        [sys.executable, "-c", _DELAYED_WRITER, str(flag), str(marker), str(go)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace", start_new_session=True,
    )
    capture = _capture(proc, selectable, grace_s=0.3)
    try:
        out, _err = capture.communicate(timeout=30)
        assert out == "answer\n" and proc.returncode == 0
        assert capture.pipe_held
        capture.close()
        go.write_text("go", encoding="utf-8")
        assert _wait_for(flag.exists), "the survivor's write never finished"
        assert flag.read_text(encoding="utf-8") == "ok"
        # What it wrote after the capture let go was dropped, not kept.
        assert capture._sinks["stdout"].snapshot() == b"answer\n"
    finally:
        capture.close()
        _stop_group(proc)
        # With the writer gone the drains reach end of file and end.
        assert _wait_for(lambda: not (set(_capture_threads()) - threads_before)), (
            f"drain threads outlived the test: {_capture_threads()}")


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


# --------------------------------------------------------------------------- #
# The stdout watch (docs/development/providers.md#a-cli-that-loops)
# --------------------------------------------------------------------------- #

_PRINTS_FOREVER = (
    "import sys, time\n"
    "while True:\n"
    "    sys.stdout.write('tick\\n'); sys.stdout.flush(); time.sleep(0.01)\n"
)


class _CountTicks:
    """Asks for a stop at the fifth ``tick`` line, and records each chunk. A
    watch sees the raw bytes, and a Python child's text-mode stdout ends its
    lines with ``\\r\\n`` on Windows, so a line ends at ``\\n`` either way."""

    def __init__(self, at=5):
        self.at, self.seen, self.chunks = at, 0, 0

    def __call__(self, chunk):
        self.chunks += 1
        self.seen += chunk.replace(b"\r\n", b"\n").count(b"tick\n")
        return f"saw {self.seen} ticks" if self.seen >= self.at else None


def _reap(proc, capture):
    proc.kill()
    return capture.communicate(timeout=10.0)


@DRAINS
def test_a_watch_ends_the_wait_long_before_the_deadline(selectable, capsys):
    proc = _spawn(_PRINTS_FOREVER)
    watch = _CountTicks()
    capture = _capture(proc, selectable, watch=watch)
    started = time.monotonic()
    try:
        with pytest.raises(process_capture.StreamStopped) as caught:
            capture.communicate(timeout=120.0)
        waited = time.monotonic() - started
        assert waited < 30.0, waited
        # A TimeoutExpired, so a lane's timeout handler stops the CLI.
        assert isinstance(caught.value, subprocess.TimeoutExpired)
        assert caught.value.reason.startswith("saw ") and caught.value.timeout == 120.0
        assert caught.value.stdout.startswith("tick\n")
        assert capture.stream_stop == caught.value.reason
        assert "[STREAM_STOP]" in capsys.readouterr().err
        # The watch is dropped once it decided, and the stop's own second
        # call collects the rest without being stopped again.
        chunks = watch.chunks
        out, _err = _reap(proc, capture)
        assert out.startswith("tick\n") and len(out) >= len(caught.value.stdout)
        assert watch.chunks == chunks
    finally:
        capture.close()


@DRAINS
def test_a_watch_that_never_decides_leaves_the_exit_and_the_deadline_alone(selectable):
    proc = _spawn("print('done')")
    capture = _capture(proc, selectable, watch=lambda chunk: None)
    try:
        assert capture.communicate(timeout=30.0) == ("done\n", "")
        assert proc.returncode == 0 and capture.stream_stop is None
    finally:
        capture.close()
    proc = _spawn(_PRINTS_FOREVER)
    capture = _capture(proc, selectable, watch=lambda chunk: None)
    try:
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            capture.communicate(timeout=0.5)
        assert not isinstance(caught.value, process_capture.StreamStopped)
        assert caught.value.timeout == 0.5 and caught.value.stdout.startswith("tick\n")
        assert capture.stream_stop is None
        _reap(proc, capture)
    finally:
        capture.close()


@DRAINS
def test_a_watch_that_raises_is_dropped_and_the_cli_runs_on(selectable):
    calls = []

    def broken(chunk):
        calls.append(chunk)
        raise ValueError("a watch bug")

    proc = _spawn(
        "import sys, time\n"
        "for _ in range(3):\n"
        "    print('line', flush=True); time.sleep(0.05)\n"
    )
    capture = _capture(proc, selectable, watch=broken)
    try:
        assert capture.communicate(timeout=30.0) == ("line\n" * 3, "")
        assert proc.returncode == 0 and capture.stream_stop is None
        assert len(calls) == 1
    finally:
        capture.close()


def test_provider_watch_never_fails_a_dispatch():
    class _None:
        def stream_watch(self):
            return None

    class _Raises:
        def stream_watch(self):
            raise RuntimeError("no")

    class _NotCallable:
        def stream_watch(self):
            return "watch"

    watch = _CountTicks()

    class _Declares:
        def stream_watch(self):
            return watch

    for provider in (object(), _None(), _Raises(), _NotCallable()):
        assert process_capture.provider_watch(provider) is None
    assert process_capture.provider_watch(_Declares()) is watch
