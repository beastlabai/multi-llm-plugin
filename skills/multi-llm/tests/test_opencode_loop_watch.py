"""opencode's stdout watch: a ``run`` that loops on a failed provider response is
stopped early, and nothing else is.

opencode 1.19.5 re-sends the request after a step that finished ``unknown`` and
printed nothing, so a cut stream or an HTTP 200 error body loops until the
deadline (``tests/fixtures/opencode/loop/README.md``).
:class:`utils.providers.opencode.LoopWatch` stops it after
:data:`~utils.providers.opencode.LOOP_EMPTY_STEPS` such steps in a row; both
lanes stop it through their own timeout path and record ``runner_error``
(docs/development/providers.md#a-cli-that-loops). No real opencode runs here:
the lane tests drive a Python stand-in that replays the committed loop capture.
"""

import contextlib
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import attempt_classifier, dispatch_spawn, llm_client  # noqa: E402
from utils.process_capture import PipeCapture, StreamStopped  # noqa: E402
from utils.provider_outcome import FAILURE_CATEGORY_KEY, STREAM_STOP_KEY  # noqa: E402
from utils.provider_registry import _PROVIDERS  # noqa: E402
from utils.providers.base import PROFILE_AGENT_READONLY  # noqa: E402
from utils.providers.opencode import LOOP_EMPTY_STEPS, LOOP_LINE_MAX, LoopWatch  # noqa: E402
from utils.retry_ladder import STATUS_RUNNER_ERROR, STATUS_TIMEOUT  # noqa: E402
from utils.review_orchestrator_base import attempt_ladder_status  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "opencode"
LOOP = FIXTURES / "loop"
LOOP_HEADS = sorted(LOOP.glob("*.head.stdout.txt"))


def _committed_streams():
    """Every committed opencode stdout stream but the loop heads: the rig's
    captures, the manifest rows and anything added beside them."""
    found = [p for p in FIXTURES.rglob("*.jsonl") if not p.name.endswith(".requests.jsonl")]
    found += list(FIXTURES.rglob("*.stdout.txt"))
    return sorted(p for p in found if LOOP not in p.parents)


STREAMS = _committed_streams()


def _feed(data: bytes, size: int, watch=None):
    """Feed *data* in *size*-byte chunks; ``(bytes fed, reason)`` at the first
    reason, else ``(None, None)``."""
    watch = watch or LoopWatch()
    for start in range(0, len(data), size):
        chunk = data[start:start + size]
        reason = watch(chunk)
        if reason:
            return start + len(chunk), reason
    return None, None


def _event(kind, ts, **part):
    body = {"type": kind, "timestamp": ts, "sessionID": "ses_t"}
    if kind != "error":
        body["part"] = part
    else:
        body["error"] = part
    return (json.dumps(body) + "\n").encode()


def _boundary(kind, ts, msg, **part):
    """A ``step_start`` or ``step_finish`` line of message *msg*, shaped as
    1.19.5 prints them: the part's type, session and message ids."""
    part_type = "step-start" if kind == "step_start" else "step-finish"
    return _event(kind, ts, type=part_type, messageID=msg, sessionID="ses_t", **part)


def _step(ts, reason="unknown", *between):
    """One step: ``step_start``, the given events, ``step_finish`` *reason*."""
    msg = f"msg_{ts}"
    out = _boundary("step_start", ts, msg)
    for kind in between:
        if kind == "text":
            out += _event("text", ts + 1, type="text", text="Reading the plan.",
                          time={"start": ts, "end": ts + 1})
        elif kind == "tool_use":
            out += _event("tool_use", ts + 1, type="tool", tool="read",
                          state={"status": "completed", "output": "x"})
        else:
            out += _event(kind, ts + 1, type=kind)
    tokens = {"input": 0, "output": 0, "reasoning": 0, "cache": {"write": 0, "read": 0}}
    return out + _boundary("step_finish", ts + 2, msg, reason=reason,
                           tokens=tokens, cost=0)


def _empty_steps(n, ts=1_000):
    return b"".join(_step(ts + 30 * i) for i in range(n))


#: A ``step_finish`` with nothing to close: no start, session or part type.
_BARE_FINISH = b'{"type":"step_finish","part":{"reason":"unknown"}}\n'


def _pairs(start, finish=None, n=2 * LOOP_EMPTY_STEPS):
    """*n* steps built from ``start(ts, msg)`` and ``finish(ts, msg)``, each
    finishing ``unknown`` with nothing printed."""
    if finish is None:
        def finish(ts, msg):
            return _boundary("step_finish", ts + 2, msg, reason="unknown")
    return b"".join(
        start(1_000 + 30 * i, f"msg_{i}") + finish(1_000 + 30 * i, f"msg_{i}")
        for i in range(n)
    )


def _start(ts, msg):
    return _boundary("step_start", ts, msg)


# --------------------------------------------------------------------------- #
# The committed streams
# --------------------------------------------------------------------------- #

def test_the_committed_streams_are_all_here():
    """The sweep below is not vacuous: the rig's 72 captures and the rows."""
    assert len(STREAMS) >= 80, len(STREAMS)
    assert len(LOOP_HEADS) == 2, LOOP_HEADS


@pytest.mark.parametrize("size", [1, 7, 4096, 1 << 20], ids=lambda s: f"chunk{s}")
def test_no_committed_stream_but_the_loops_trips_it(size):
    tripped = [
        (path.relative_to(FIXTURES).as_posix(), reason)
        for path in STREAMS
        for _fed, reason in [_feed(path.read_bytes(), size)]
        if reason
    ]
    assert tripped == []


@pytest.mark.parametrize("path", LOOP_HEADS, ids=lambda p: p.name.split(".")[0])
@pytest.mark.parametrize("size", [1, 7, 4096, 1 << 20], ids=lambda s: f"chunk{s}")
def test_each_loop_capture_trips_at_the_threshold_step(path, size):
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    # Each step is a step_start and a step_finish line, nothing else.
    assert all(json.loads(line)["type"] in ("step_start", "step_finish") for line in lines)
    trip_at = len(b"".join(lines[:2 * LOOP_EMPTY_STEPS]))
    fed, reason = _feed(data, size)
    assert fed is not None, "the loop was not stopped"
    # Within the chunk that completed the threshold step's step_finish line.
    assert trip_at <= fed < trip_at + size
    assert reason.startswith(f"opencode looped: {LOOP_EMPTY_STEPS} steps in a row within ")
    assert 'finish reason "unknown"' in reason and "HTTP 200" in reason


@pytest.mark.parametrize("path", LOOP_HEADS, ids=lambda p: p.name.split(".")[0])
def test_each_loop_capture_stops_within_two_seconds_of_opencodes_clock(path):
    """Measured: 0.7 s (cut stream) and 1.6 s (error body) by the stream's own
    timestamps, against the 240 s and ~3,900 requests the deadline took."""
    lines = path.read_bytes().splitlines()
    first = json.loads(lines[0])["timestamp"]
    trip = json.loads(lines[2 * LOOP_EMPTY_STEPS - 1])["timestamp"]
    assert 0 < (trip - first) / 1000 < 2.0


# --------------------------------------------------------------------------- #
# The threshold
# --------------------------------------------------------------------------- #

def test_one_step_short_of_the_threshold_does_not_trip_and_the_next_does():
    watch = LoopWatch()
    assert watch(_empty_steps(LOOP_EMPTY_STEPS - 1)) is None
    assert watch.streak == LOOP_EMPTY_STEPS - 1
    assert watch(_step(99_000)) is not None


@pytest.mark.parametrize("breaker", [
    pytest.param(lambda ts: _step(ts, "unknown", "text"), id="text-in-an-unknown-step"),
    pytest.param(lambda ts: _step(ts, "unknown", "tool_use"), id="tool-in-an-unknown-step"),
    pytest.param(lambda ts: _step(ts, "unknown", "reasoning"), id="reasoning-in-a-step"),
    pytest.param(lambda ts: _step(ts, "tool-calls"), id="a-tool-calls-finish"),
    pytest.param(lambda ts: _step(ts, "stop"), id="a-stop-finish"),
    pytest.param(lambda ts: _step(ts, "length"), id="a-length-finish"),
    pytest.param(lambda ts: _event("error", ts, name="UnknownError"), id="an-error-event"),
    pytest.param(lambda ts: _event("step_progress", ts, type="x"), id="an-unemitted-type"),
    pytest.param(lambda ts: b"opencode: not json\n", id="a-plain-line"),
    pytest.param(lambda ts: b"[1, 2]\n", id="json-not-an-event"),
    # Boundaries out of order: each ends the streak, and the next well-formed
    # start counts again.
    pytest.param(lambda ts: _BARE_FINISH, id="a-bare-finish"),
    pytest.param(lambda ts: _boundary("step_finish", 1_542, "msg_1540", reason="unknown"),
                 id="a-duplicate-finish"),
    pytest.param(lambda ts: _event("step_start", ts, type="step-start")
                 + _boundary("step_finish", ts + 2, "msg_x", reason="unknown"),
                 id="a-malformed-start"),
    pytest.param(lambda ts: _boundary("step_start", ts, "msg_a")
                 + _boundary("step_finish", ts + 2, "msg_b", reason="unknown"),
                 id="a-finish-of-another-step"),
])
def test_anything_printed_or_any_other_finish_restarts_the_count(breaker):
    stream = _empty_steps(LOOP_EMPTY_STEPS - 1) + breaker(50_000)
    stream += _empty_steps(LOOP_EMPTY_STEPS - 1, ts=60_000)
    watch = LoopWatch()
    assert _feed(stream, 1, watch) == (None, None)
    assert _feed(_step(90_000), 1, watch)[1] is not None


@pytest.mark.parametrize("stream", [
    pytest.param(_BARE_FINISH * (2 * LOOP_EMPTY_STEPS), id="bare-finishes"),
    pytest.param(b"".join(_boundary("step_finish", 1_000 + i, f"msg_{i}", reason="unknown")
                          for i in range(2 * LOOP_EMPTY_STEPS)), id="finishes-with-no-start"),
    pytest.param(_pairs(lambda ts, msg: b'{"type":"step_start"}\n'), id="starts-without-a-part"),
    pytest.param(_pairs(lambda ts, msg: _event("step_start", ts, type="text", messageID=msg,
                                               sessionID="ses_t")),
                 id="starts-of-another-part-type"),
    pytest.param(_pairs(lambda ts, msg: _event("step_start", ts, type="step-start",
                                               sessionID="ses_t")),
                 id="starts-without-a-message-id"),
    pytest.param(_pairs(lambda ts, msg: (json.dumps({
        "type": "step_start", "part": {"type": "step-start", "messageID": msg},
    }) + "\n").encode()), id="starts-without-a-session"),
    pytest.param(_pairs(_start, lambda ts, msg: 2 * _boundary(
        "step_finish", ts + 2, msg, reason="unknown")), id="duplicate-finishes"),
    pytest.param(_pairs(lambda ts, msg: 2 * _start(ts, msg)), id="duplicate-starts"),
    pytest.param(_pairs(_start, lambda ts, msg: _boundary(
        "step_finish", ts + 2, "msg_other", reason="unknown")), id="finishes-of-another-step"),
    pytest.param(_pairs(_start, lambda ts, msg: _event(
        "step_finish", ts + 2, reason="unknown", messageID=msg, sessionID="ses_t")),
        id="finishes-without-a-part-type"),
])
def test_only_a_finish_closing_a_well_formed_step_counts(stream):
    """Twice the threshold of empty ``unknown`` finishes, none of them the
    close of one well-formed step: no stop, whatever the chunking."""
    for size in (1, 4096, len(stream)):
        assert _feed(stream, size) == (None, None)


def test_well_formed_pairs_still_trip_at_the_threshold():
    """The positive half of the row above: the same pairs with their part
    types, session and message ids stop the CLI at step 20, no sooner."""
    stream = _pairs(_start)
    lines = stream.splitlines(keepends=True)
    trip_at = len(b"".join(lines[:2 * LOOP_EMPTY_STEPS]))
    fed, reason = _feed(stream, 1)
    assert fed == trip_at and reason is not None


def test_an_over_long_line_counts_as_printed_and_is_not_kept():
    huge = _event("text", 2, type="text", text="x" * (2 * LOOP_LINE_MAX),
                  time={"start": 1, "end": 2})
    stream = _empty_steps(LOOP_EMPTY_STEPS - 1)
    stream += _boundary("step_start", 3, "msg_long") + huge
    stream += _boundary("step_finish", 4, "msg_long", reason="unknown")
    stream += _empty_steps(LOOP_EMPTY_STEPS - 1, ts=10_000)
    for size in (1, 4096, len(stream)):
        watch = LoopWatch()
        assert _feed(stream, size, watch) == (None, None)
        assert len(watch._pending) <= LOOP_LINE_MAX


def test_a_long_legitimate_run_never_trips_it():
    """60 steps, more than the 47 a recorded session reached, shaped like the
    real sessions on record: tool steps with narration, and a few short
    streaks of empty ``unknown`` steps (the longest real one was 2) that the
    next ordinary step recovers from, then the closing answer."""
    stream = b""
    ts = 1_000
    for i in range(60):
        ts += 5_000
        if i in (7, 21, 22, 40, 41, 42):
            stream += _step(ts)
        elif i % 3 == 0:
            stream += _step(ts, "tool-calls", "text", "tool_use", "tool_use")
        else:
            stream += _step(ts, "tool-calls", "tool_use")
    stream += _step(ts + 5_000, "stop", "text")
    for size in (1, 7, 4096, len(stream)):
        assert _feed(stream, size) == (None, None)


def test_the_threshold_keeps_its_margin_over_real_streaks():
    """Real sessions: the longest empty-``unknown`` streak was 2 in 11,187
    steps; the threshold is ten times that."""
    assert LOOP_EMPTY_STEPS >= 10 * 2


#: A run that loops for its first 32 steps, then answers in a step that
#: finishes ``stop`` and exits 0: the files named on its command line, in order.
_RECOVERS = textwrap.dedent("""\
    import sys
    out = sys.stdout.buffer
    for name in sys.argv[1:]:
        with open(name, "rb") as f:
            out.write(f.read())
    out.flush()
""")


@pytest.mark.parametrize("selectable", [
    pytest.param(True, id="selector-drain",
                 marks=pytest.mark.skipif(os.name == "nt", reason="POSIX drain")),
    pytest.param(False, id="reader-threads"),
])
def test_a_run_that_exited_before_the_stop_keeps_its_own_ending(selectable, tmp_path, capsys):
    """The drain fed the watch step 20 of a run that then answered and exited 0
    before the wait looked: the exit decides. Nothing is stopped, the capture
    is whole and ``stream_stop`` stays ``None``."""
    head = LOOP_HEADS[0]
    answer = tmp_path / "answer.jsonl"
    answer.write_bytes(_step(9_000_000, "stop", "text"))
    script = tmp_path / "recovers.py"
    script.write_text(_RECOVERS, encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(script), str(head), str(answer)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace",
    )
    loop, tripped = LoopWatch(), threading.Event()

    def watch(chunk):
        reason = loop(chunk)
        if reason:
            tripped.set()
        return reason

    capture = PipeCapture(proc, selectable=selectable, watch=watch)
    try:
        assert proc.wait(timeout=30) == 0
        assert tripped.wait(30), "the drain never fed the watch step 20"
        out, err = capture.communicate(timeout=30)
    finally:
        capture.close()
    assert capture.stream_stop is None and proc.returncode == 0
    assert out == (head.read_bytes() + answer.read_bytes()).decode("utf-8")
    assert err == ""
    assert "[STREAM_STOP]" not in capsys.readouterr().err


def test_each_dispatch_gets_a_fresh_watch():
    adapter = _PROVIDERS["opencode"]
    first, second = adapter.stream_watch(), adapter.stream_watch()
    assert isinstance(first, LoopWatch) and first is not second
    first(_empty_steps(5))
    assert first.streak == 5 and second.streak == 0


def test_no_other_adapter_declares_a_watch():
    """A watch is declared only for a CLI measured to loop."""
    declaring = sorted(name for name, adapter in _PROVIDERS.items()
                       if adapter.stream_watch() is not None)
    assert declaring == ["opencode"]


# --------------------------------------------------------------------------- #
# Both lanes, through their real stop paths
# --------------------------------------------------------------------------- #

#: Replays a loop capture forever, one line at a time, like opencode against a
#: failing provider; or, with ``progress``, prints a text event in every step.
_STAND_IN = textwrap.dedent("""\
    import os, sys, time
    head, pidfile, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(pidfile, "w") as f:
        f.write(str(os.getpid()))
    lines = open(head, "rb").read().splitlines(keepends=True)
    text = b'{"type":"text","part":{"type":"text","text":"working"}}\\n'
    out = sys.stdout.buffer
    while True:
        for line in lines:
            if mode == "progress" and b'"step_finish"' in line:
                out.write(text)
            out.write(line)
            out.flush()
            time.sleep(0.002)
""")


@pytest.fixture
def stand_in(tmp_path):
    script = tmp_path / "opencode_stand_in.py"
    script.write_text(_STAND_IN, encoding="utf-8")
    pidfile = tmp_path / "pid"

    def argv(mode="loop"):
        return [sys.executable, str(script), str(LOOP_HEADS[0]), str(pidfile), mode]

    argv.pidfile = pidfile
    return argv


def _gone(pidfile):
    """The stand-in is no longer running (POSIX; Windows has no signal 0)."""
    if os.name == "nt":
        return True
    pid = int(pidfile.read_text())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


@contextlib.contextmanager
def _lane_a_opencode(monkeypatch, argv):
    """``invoke_with_provider`` for opencode with its argv swapped for the
    stand-in; the real ``_run_provider_cli`` (and its sweep) runs."""
    adapter = _PROVIDERS["opencode"]

    @contextlib.contextmanager
    def no_dispatch_context(*_args, **_kwargs):
        class _Dispatch:
            refusal = None
        yield _Dispatch()

    monkeypatch.setattr(llm_client, "_resolve_executable", lambda name: ([name], False))
    monkeypatch.setattr(llm_client, "provider_dispatch_context", no_dispatch_context)
    # The registry's adapter is process-wide, so shadow its methods through the
    # instance ``__dict__``, whose undo deletes the key. Undoing an instance
    # ``setattr`` would leave the original bound method on the instance, where
    # it shadows a later test's class-level patch of the same method
    # (``test_provider_registry``'s session rung patches ``is_available`` on the
    # class), as ``test_lane_a_ladder._patch_adapter`` explains.
    monkeypatch.setitem(vars(adapter), "is_available", lambda: True)
    monkeypatch.setitem(vars(adapter), "build_command_for_transport",
                        lambda *a, **k: (list(argv), None))
    monkeypatch.setitem(vars(adapter), "resolve_env", lambda *a, **k: dict(os.environ))
    yield


def test_lane_a_stops_the_loop_early_and_records_runner_error(
    monkeypatch, stand_in, tmp_path, capsys,
):
    with _lane_a_opencode(monkeypatch, stand_in()):
        started = time.monotonic()
        result = llm_client.invoke_with_provider(
            "hi", "opencode:m", timeout=600, cwd=str(tmp_path),
        )
        waited = time.monotonic() - started
    assert waited < 60, waited
    details = result["details"]
    assert result["success"] is False
    assert result["error"].startswith("opencode looped: 20 steps in a row")
    assert "rather than at its 600s timeout" in result["error"]
    assert "timed out" not in result["error"]
    assert details[STREAM_STOP_KEY].startswith("opencode looped:")
    assert details[FAILURE_CATEGORY_KEY] == STATUS_RUNNER_ERROR
    assert attempt_ladder_status(result, details) == STATUS_RUNNER_ERROR
    # The plugin cut the run short: the salvage gate's deadline-kill rule holds.
    assert result["error_code"] == llm_client.ERROR_TIMEOUT
    assert details["salvage_text"] == ""
    assert "[STREAM_STOP]" in capsys.readouterr().err
    assert _gone(stand_in.pidfile)


def test_lane_a_still_times_out_a_run_that_prints(monkeypatch, stand_in, tmp_path):
    """The same stand-in printing a text event in every step is progress: the
    watch never trips and the deadline decides, as before."""
    with _lane_a_opencode(monkeypatch, stand_in("progress")):
        result = llm_client.invoke_with_provider(
            "hi", "opencode:m", timeout=2, cwd=str(tmp_path),
        )
    details = result["details"]
    assert result["error"].startswith("opencode timed out after 2s")
    assert STREAM_STOP_KEY not in details
    assert details[FAILURE_CATEGORY_KEY] == STATUS_TIMEOUT
    assert _gone(stand_in.pidfile)


def test_lane_a_spawn_raises_a_timeout_carrying_the_reason(stand_in):
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        llm_client._run_provider_cli(
            stand_in(), capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=600, stdin=subprocess.DEVNULL,
            stream_watch=LoopWatch(),
        )
    assert caught.value.stream_stop.startswith("opencode looped:")
    assert caught.value.timeout == 600
    assert caught.value.tree_sweep is not None
    assert llm_client._sweep_confirmed(caught.value.tree_sweep)
    assert _gone(stand_in.pidfile)


def test_lane_b_stops_the_loop_early_and_records_runner_error(
    monkeypatch, stand_in, tmp_path,
):
    adapter = _PROVIDERS["opencode"]
    monkeypatch.setattr(adapter, "build_agent_command_for_transport",
                        lambda *a, **k: (stand_in(), None))
    monkeypatch.setattr(dispatch_spawn, "dispatch_child_env", lambda *a, **k: None)
    started = time.monotonic()
    capture = dispatch_spawn._spawn_in_context(
        provider=adapter, model="m", prompt="hi", profile=PROFILE_AGENT_READONLY,
        transport="argv", timeout=600, cwd=str(tmp_path),
    )
    assert time.monotonic() - started < 60
    assert capture.timed_out is True
    assert capture.stream_stop.startswith("opencode looped: 20 steps in a row")
    assert capture.kill_verdict in ("killed", "gone"), capture.kill_verdict
    status, line = attempt_classifier._classify_pre_parse(capture)
    assert status == STATUS_RUNNER_ERROR
    assert line.startswith("opencode looped:")
    assert "rather than at its per-task timeout" in line
    assert _gone(stand_in.pidfile)


def test_the_executor_reads_a_deadline_kill_as_before():
    """No watch reason: rung 2, the per-task timeout line, unchanged."""
    capture = dispatch_spawn._Capture(timed_out=True)
    status, line = attempt_classifier._classify_pre_parse(capture)
    assert status == STATUS_TIMEOUT and "exceeded its per-task timeout" in line


def test_stream_stopped_is_a_timeout_for_every_lane_handler():
    assert issubclass(StreamStopped, subprocess.TimeoutExpired)
