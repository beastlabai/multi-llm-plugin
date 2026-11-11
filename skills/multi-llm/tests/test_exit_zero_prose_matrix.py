"""An exit 0's answer never decides a retry, a backoff or a stop, on either lane.

On an exit 0 that left no usable answer, stdout is usually the model's own
answer: prose instead of the JSON asked for, quoting whatever the plan says
about rate limits, credentials and context windows. Both lanes classify such a
run through one gate, ``base.exit_zero_failure_category``: only an adapter that
declares ``EXIT0_CLASSIFIABLE`` gives a verdict, read from what the CLI itself
wrote. This file is the review behind each declaration.

For every adapter in the registry, the answer is replaced by one prose kind at
a time, in three envelope shapes: the CLI's complete success shape, an
unfinished one (the stream stops after the answer's record, or a single
envelope flags the run as failed), and one cut right after the answer's text.
Each lane must reach the same status for every kind as for a neutral answer in
the same shape, and that status must be no backoff, no fatal verdict and no
terminal category. An adapter that is not ``EXIT0_CLASSIFIABLE`` keeps the
lanes' unclassified behaviour: the fan-out stops (``None``), the executor keeps
its parse failure's one retry.

The same holds for an answer that quotes a CLI's own diagnostic line verbatim
(aider's ``litellm.RateLimitError: …``, in a fenced example), and for goose's
own ``Ran into this error`` wording in a model turn goose did not stamp: a
line's shape cannot say who wrote it.

A parsed answer the file-output layer rejects (a bare ``{"raw": …}``) takes one
verdict on both lanes: the gate's, when stderr names a real diagnosis, else
``unparseable_result``.

A non-zero exit whose answer quotes a launch failure is not ``spawn_failed``
either: that verdict fails the executor's batch, so it is read only off exit
127, stderr and an adapter's own refusal shape.
See docs/development/delegation.md#read-and-analyze-executor.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_provider_fixture_rows import (  # noqa: E402
    MANIFESTS,
    _lane_a_status,
    _lane_b_status,
    _retried,
)
from utils.provider_outcome import Capture  # noqa: E402
from utils.provider_registry import _PROVIDERS  # noqa: E402
from utils.json_extractor import extract_json_from_text  # noqa: E402
from utils.providers.base import (  # noqa: E402
    LAUNCH_DIAGNOSTICS_NONE,
    LAUNCH_DIAGNOSTICS_STDERR,
    STDOUT_ENVELOPE,
    classify_streams,
    read_stdout,
)
from utils.retry_ladder import (  # noqa: E402
    BACKOFF_STATUSES,
    RetryLadder,
    FATAL_STATUSES,
    STATUS_BUDGET_EXHAUSTED,
    STATUS_CONTEXT_LENGTH_EXCEEDED,
    STATUS_OK,
    STATUS_SANDBOX_UNAVAILABLE,
    STATUS_SPAWN_FAILED,
    STATUS_UNPARSEABLE,
)

FIXTURES = Path(__file__).parent / "fixtures"

#: One answer per kind. ``neutral`` is the baseline every other kind must
#: match; each other one would decide a verdict if the tables read it.
#: Sandbox words an answer may use. ``sandbox_unavailable`` fails the
#: executor's whole batch, so it is read off stderr alone, as whole measured
#: lines (``base.SANDBOX_MARKERS``): none of these is one, on either stream.
SANDBOX_PROSE = {
    "bubblewrap": "The plan should explain bubblewrap.",
    "bwrap-line": "bwrap: loopback: Failed RTM_NEWADDR",
    "landlock": "The jail applies a Landlock ruleset before the exec.",
    "seccomp": "The container's seccomp profile blocks unshare.",
    "sandbox-unavailable": "doctor reports sandbox unavailable on Ubuntu 24.04.",
}
PROSE = {
    "neutral": "The implementation needs a focused regression test.",
    "number": "I reviewed 429 files.",
    "rate": "The plan says to retry HTTP 429 Too Many Requests and rate limit errors.",
    "auth": "The plan says 401 Unauthorized means an invalid api key.",
    "context": "The plan documents the maximum context length of 128000 tokens.",
    "launch": "The plan documents command not found failures.",
    **SANDBOX_PROSE,
}
KINDS = [kind for kind in PROSE if kind != "neutral"]
SHAPES = ("complete", "unfinished", "cut")

#: What no answer may ever produce: a backoff, a stop that fails the slot or
#: the batch, or the terminal context verdict.
FORBIDDEN = (
    BACKOFF_STATUSES | FATAL_STATUSES
    | {STATUS_CONTEXT_LENGTH_EXCEEDED, STATUS_BUDGET_EXHAUSTED, "permission_denied"}
)


def _read(provider: str, name: str) -> str:
    return (FIXTURES / provider / name).read_text(encoding="utf-8")


def _success(provider: str) -> str:
    row = MANIFESTS[provider]["rows"]["success"]
    return _read(provider, row["stdout"])


def _escaped(text: str) -> str:
    return json.dumps(text)[1:-1]


def _swap_answer(provider: str, stdout: str, prose: str) -> str:
    """``stdout`` with the answer the adapter salvages from it replaced."""
    answer = _PROVIDERS[provider].salvage_text(stdout)
    assert answer, f"{provider}: no answer in its success capture"
    swapped = stdout.replace(_escaped(answer), _escaped(prose)).replace(answer, prose)
    assert prose in swapped or _escaped(prose) in swapped, provider
    return swapped


def _envelope(provider: str, name: str, fields, prose: str, **extra) -> str:
    envelope = json.loads(_read(provider, name))
    for field in fields:
        envelope[field] = prose
    envelope.update(extra)
    return json.dumps(envelope)


def _jsonl(stdout: str, rewrite) -> str:
    out = []
    for line in stdout.split("\n"):
        if not line.strip():
            out.append(line)
            continue
        record = json.loads(line)
        rewrite(record)
        out.append(json.dumps(record))
    return "\n".join(out)


def _cline(prose: str) -> str:
    def rewrite(event):
        inner = event.get("event") or {}
        for key in ("text", "accumulated"):
            if isinstance(inner.get(key), str):
                inner[key] = prose
        if event.get("type") == "run_result":
            event["text"] = prose

    return _jsonl(_success("cline"), rewrite)


def _kimi(prose: str) -> str:
    def rewrite(record):
        if record.get("role") in ("assistant", "tool"):
            record["content"] = prose

    return _jsonl(_read("kimi", "multi-assistant.stdout.txt"), rewrite)


def _codex(prose: str) -> str:
    return "\n".join(json.dumps(event) for event in (
        {"type": "thread.started", "thread_id": "t"},
        {"type": "turn.started"},
        {"type": "item.completed",
         "item": {"id": "i", "type": "agent_message", "text": prose}},
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ))


def _opencode(prose: str) -> str:
    return "\n".join(json.dumps(event) for event in (
        {"type": "step_start", "part": {"type": "step-start"}},
        {"type": "text", "part": {"type": "text", "text": prose}},
        {"type": "step_finish", "part": {"type": "step-finish", "reason": "stop"}},
    ))


#: The complete success shape, per adapter, with the answer as ``prose``.
COMPLETE = {
    # agy's envelope opens on `conversation_id` (tests/fixtures/agy/auth-error).
    "agy": lambda p: json.dumps({"conversation_id": "conv-1", "status": "SUCCESS", "response": p}),
    "aider": lambda p: _swap_answer("aider", _success("aider"), p),
    "claude-code": lambda p: _envelope("claude-code", "success.stdout.txt", ["result"], p),
    "cline": _cline,
    "codex": _codex,
    "copilot": lambda p: _swap_answer("copilot", _success("copilot"), p),
    "cursor-agent": lambda p: _envelope("cursor-agent", "success.stdout.txt", ["result"], p),
    "fx": lambda p: _envelope("fx", "success.stdout.txt", ["output", "final_output"], p),
    "gemini": lambda p: _envelope("gemini", "success.stdout.txt", ["response"], p),
    "goose": lambda p: _swap_answer("goose", _success("goose"), p),
    "grok": lambda p: _envelope("grok", "success.stdout.txt", ["text"], p),
    "kilocode": lambda p: p,
    "kimi": _kimi,
    "muse": lambda p: _swap_answer("muse", _success("muse"), p),
    "omp": lambda p: _swap_answer("omp", _success("omp"), p),
    "opencode": _opencode,
    "pi": lambda p: _swap_answer("pi", _success("pi"), p),
    "qwen-code": lambda p: _swap_answer("qwen-code", _read("qwen-code", "success.stdout.json"), p),
    "vibe": lambda p: _swap_answer("vibe", _success("vibe"), p),
}

#: A single envelope has no record to stop after, so its unfinished shape is
#: the envelope flagging the run as failed with the answer still in it.
FLAGGED = {
    "agy": lambda p: json.dumps(
        {"conversation_id": "", "status": "ERROR", "error": "Request failed.", "response": p}
    ),
    "claude-code": lambda p: _envelope(
        "claude-code", "success.stdout.txt", ["result"], p,
        is_error=True, subtype="error_during_execution", terminal_reason="aborted",
    ),
    "cursor-agent": lambda p: _envelope(
        "cursor-agent", "success.stdout.txt", ["result"], p, is_error=True,
    ),
    "fx": lambda p: _envelope(
        "fx", "success.stdout.txt", ["output"], p, final_output="", exit_code=1,
    ),
    "gemini": lambda p: _envelope(
        "gemini", "success.stdout.txt", ["response"], p,
        error={"type": "INVALID_STREAM", "message": "The stream ended early."},
    ),
    "grok": lambda p: _envelope("grok", "success.stdout.txt", ["text"], p, stopReason="max_tokens"),
}


def _first_index(stdout: str, prose: str) -> int:
    hits = [i for i in (stdout.find(prose), stdout.find(_escaped(prose))) if i >= 0]
    assert hits, "the prose is in the capture"
    return min(hits)


def _unfinished(provider: str, prose: str) -> str:
    if provider in FLAGGED:
        return FLAGGED[provider](prose)
    stdout = COMPLETE[provider](prose)
    lines = stdout.split("\n")
    index = _first_index(stdout, prose)
    count = stdout[:index].count("\n")
    if provider == "kilocode" or len(lines) == 1:
        return stdout[: index + len(prose) // 2]
    return "\n".join(lines[: count + 1])


def _cut(provider: str, prose: str) -> str:
    stdout = COMPLETE[provider](prose)
    index = _first_index(stdout, prose)
    end = index + (len(prose) if stdout.startswith(prose, index) else len(_escaped(prose)))
    return stdout[:end]


BUILD = {
    "complete": lambda provider, prose: COMPLETE[provider](prose),
    "unfinished": _unfinished,
    "cut": _cut,
}


def _lanes(provider: str, stdout: str, exit_code: int = 0, stderr: str = ""):
    capture = Capture(stdout=stdout, stderr=stderr, exit_code=exit_code)
    return _lane_a_status(provider, capture), _lane_b_status(provider, capture)


def test_every_registry_adapter_is_in_the_matrix():
    assert set(COMPLETE) == set(_PROVIDERS)


def test_every_adapter_declares_exit0_classifiable_itself():
    """The property is a reviewed claim, so each adapter states it."""
    for provider, adapter in _PROVIDERS.items():
        assert "EXIT0_CLASSIFIABLE" in vars(type(adapter)), provider
        assert isinstance(type(adapter).EXIT0_CLASSIFIABLE, bool), provider


@pytest.mark.parametrize("kind", KINDS)
def test_each_kind_would_decide_a_verdict_unreduced(kind):
    if kind == "number":
        # The boundary-aware matcher already refuses it; the adapters that
        # widen the bare 429 (agy) are what this kind is for.
        return
    if kind == "launch":
        # Read on stderr only (`base.STDERR_LAUNCH_FAILURE_MARKERS`).
        assert classify_streams("", PROSE[kind], 1) == STATUS_SPAWN_FAILED
        assert classify_streams(PROSE[kind], "", 1) is None
        return
    if kind in SANDBOX_PROSE:
        # A sandbox verdict is a whole measured line on stderr; these words
        # are none, on either stream.
        assert classify_streams(PROSE[kind], "", 1) is None
        assert classify_streams("", PROSE[kind], 1) is None
        return
    assert classify_streams(PROSE[kind], "", 1) is not None


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("provider", sorted(COMPLETE))
def test_no_answer_decides_either_lane(provider, shape):
    adapter = _PROVIDERS[provider]
    baseline = BUILD[shape](provider, PROSE["neutral"])
    lane_a, lane_b = _lanes(provider, baseline)
    assert lane_a not in FORBIDDEN and lane_b not in FORBIDDEN, (provider, shape, lane_a, lane_b)
    # A prose answer is no answer in any shape. A cut envelope's nested JSON
    # (goose's echoed prompt, claude-code's bookkeeping) is never served as
    # one: the envelope is incomplete, so the parse fails.
    parsed = adapter.parse_output(baseline, "")
    assert parsed.get("success") is not True, (provider, shape, parsed.get("data"))
    if not adapter.EXIT0_CLASSIFIABLE:
        assert (lane_a, lane_b) == (None, STATUS_UNPARSEABLE), (
            f"{provider}/{shape}: an adapter that cannot isolate its diagnostics "
            f"is never classified on an exit 0"
        )
    for kind in KINDS:
        stdout = BUILD[shape](provider, PROSE[kind])
        found = _lanes(provider, stdout)
        assert found == (lane_a, lane_b), (
            f"{provider}/{shape}/{kind}: the answer moved the lanes from "
            f"{(lane_a, lane_b)} to {found}"
        )


#: Diagnostic lines exactly as a CLI prints its own (aider's litellm renderings,
#: its token-limit line, its gloss), quoted by a review in a fenced example.
QUOTED = {
    "auth": "litellm.AuthenticationError: HTTP 401 Unauthorized invalid api key",
    "rate": "litellm.RateLimitError: HTTP 429 Too Many Requests",
    "context": "Model openai/stub-model has hit a token limit!",
    "sandbox": "The API provider uses bubblewrap for isolation.",
}


def _quoting(line: str) -> str:
    return (
        "Here is an example diagnostic from the documentation:\n```text\n"
        + line + "\n```"
    )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("provider", sorted(COMPLETE))
def test_a_quoted_diagnostic_line_decides_neither_lane(provider, shape):
    baseline = _lanes(provider, BUILD[shape](provider, _quoting("All checks passed.")))
    assert not set(baseline) & FORBIDDEN, (provider, shape, baseline)
    for kind, line in QUOTED.items():
        found = _lanes(provider, BUILD[shape](provider, _quoting(line)))
        assert found == baseline, (
            f"{provider}/{shape}/quoted {kind}: the answer moved the lanes from "
            f"{baseline} to {found}"
        )


@pytest.mark.parametrize("accounted", [True, False], ids=["accounted", "cut-before-accounting"])
@pytest.mark.parametrize("kind", sorted(QUOTED))
def test_aider_never_reads_a_quoted_diagnostic(kind, accounted):
    """aider's answer cut before its trailing accounting line can carry its own
    diagnostic shapes verbatim; aider is not ``EXIT0_CLASSIFIABLE``, so neither
    lane reads them: the fan-out stops, the executor keeps its one retry, never
    three backoffs, a terminal context verdict or a batch-fatal one."""
    stdout = COMPLETE["aider"](_quoting(QUOTED[kind]))
    if not accounted:
        stdout = "\n".join(
            line for line in stdout.split("\n") if not line.startswith("Tokens:")
        )
        assert QUOTED[kind] in stdout and "Tokens:" not in stdout
    lane_a, lane_b = _lanes("aider", stdout)
    assert (lane_a, lane_b) == (None, STATUS_UNPARSEABLE)
    assert not _retried("aider", lane_a) and _retried("aider", lane_b)


def _goose_turn(text: str, *, stamped: bool) -> str:
    envelope = json.loads(COMPLETE["goose"](text))
    if not stamped:
        for message in envelope["messages"]:
            if message.get("role") == "assistant":
                message.get("metadata", {}).pop("inference", None)
    return json.dumps(envelope)


@pytest.mark.parametrize("stamped", [True, False], ids=["stamped", "unstamped"])
def test_goose_model_turn_in_its_own_error_wording_decides_nothing(stamped):
    """goose 1.41.0 leaves ``metadata.inference`` off every model turn when it
    cannot resolve the model's info, so a model answer opening with goose's
    own ``Ran into this error:`` wording is indistinguishable from goose's
    diagnosis: goose is not ``EXIT0_CLASSIFIABLE``, and the answer neither
    stops the run nor fails the batch."""
    answer = (
        "Ran into this error: HTTP 401 Unauthorized; the plan should document "
        "the remedy."
    )
    neutral = _lanes("goose", _goose_turn(PROSE["neutral"], stamped=stamped))
    found = _lanes("goose", _goose_turn(answer, stamped=stamped))
    assert found == neutral == (None, STATUS_UNPARSEABLE)
    assert _retried("goose", found[1])


#: Real diagnoses a CLI writes on stderr next to an answer it exited 0 on.
REJECTED_STDERR = {
    "auth": "HTTP 401 Unauthorized",
    "rate": "HTTP 429 Too Many Requests",
    "context": "This model's maximum context length is 128000 tokens",
}


@pytest.mark.parametrize("stderr", ["", *REJECTED_STDERR.values()],
                         ids=["no-stderr", *REJECTED_STDERR])
@pytest.mark.parametrize("provider", sorted(COMPLETE))
def test_a_rejected_answer_takes_one_verdict_on_both_lanes(provider, stderr):
    """A clean success shape whose answer parses to a bare ``{"raw": …}``
    wrapper: both lanes reject it, then both take the one exit-0 gate's
    verdict when stderr names one, else ``unparseable_result``."""
    stdout = COMPLETE[provider]('{"raw":"hello"}')
    lane_a, lane_b = _lanes(provider, stdout, 0, stderr)
    assert lane_a == lane_b, (provider, stderr, lane_a, lane_b)
    if not stderr:
        assert lane_a == STATUS_UNPARSEABLE, (provider, lane_a)


@pytest.mark.parametrize("stderr,expected", [
    (REJECTED_STDERR["auth"], "auth_failed"),
    (REJECTED_STDERR["rate"], "rate_limited"),
    (REJECTED_STDERR["context"], STATUS_CONTEXT_LENGTH_EXCEEDED),
])
def test_a_rejected_answers_real_diagnosis_wins_on_both_lanes(stderr, expected):
    """Codex's case: cursor-agent's clean envelope, ``{"raw":"hello"}``, no
    result file, a real diagnosis on stderr."""
    stdout = COMPLETE["cursor-agent"]('{"raw":"hello"}')
    assert _lanes("cursor-agent", stdout, 0, stderr) == (expected, expected)


@pytest.mark.parametrize("provider", sorted(COMPLETE))
def test_a_quoted_launch_failure_on_a_failed_exit_keeps_its_retry(provider):
    """A non-zero exit whose answer quotes "command not found", over a stderr
    that names a dropped connection: one transport retry on each lane, never
    the batch-fatal ``spawn_failed``."""
    stdout = COMPLETE[provider](PROSE["launch"])
    lane_a, lane_b = _lanes(provider, stdout, 1, "Connection reset by peer")
    assert STATUS_SPAWN_FAILED not in (lane_a, lane_b), (provider, lane_a, lane_b)
    assert _retried(provider, lane_a) and _retried(provider, lane_b), (
        provider, lane_a, lane_b,
    )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("provider", sorted(COMPLETE))
def test_sandbox_words_on_a_failed_exit_keep_the_transport_verdict(provider, shape):
    """Codex's case on every adapter and shape: an answer that explains
    bubblewrap (or Landlock, seccomp, an unavailable sandbox), a non-zero
    exit, stderr naming a dropped connection. The answer's words move
    neither lane, and nothing is batch-fatal: ``sandbox_unavailable`` is
    read off stderr alone."""
    stderr = "Connection reset by peer"
    baseline = _lanes(provider, BUILD[shape](provider, PROSE["neutral"]), 1, stderr)
    assert not set(baseline) & FATAL_STATUSES, (provider, shape, baseline)
    for kind in ("launch", *SANDBOX_PROSE):
        found = _lanes(provider, BUILD[shape](provider, PROSE[kind]), 1, stderr)
        assert found == baseline, (
            f"{provider}/{shape}/{kind} at exit 1: the answer moved the lanes "
            f"from {baseline} to {found}"
        )


#: bubblewrap's measured refusals, as it prints them (``base.SANDBOX_MARKERS``
#: names the capture of each).
MEASURED_BWRAP_LINES = (
    "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted\n",
    "bwrap: setting up uid map: Permission denied\n",
    "bwrap: No permissions to create a new namespace\n",
)

#: The adapters whose stderr is the CLI's own, and the ones whose stderr also
#: relays a model's or a tool's words (``LLMProvider.LAUNCH_DIAGNOSTICS``).
CLI_STDERR = sorted(
    p for p in COMPLETE if _PROVIDERS[p].LAUNCH_DIAGNOSTICS == LAUNCH_DIAGNOSTICS_STDERR
)
RELAYED_STDERR = sorted(
    p for p in COMPLETE if _PROVIDERS[p].LAUNCH_DIAGNOSTICS == LAUNCH_DIAGNOSTICS_NONE
)

#: stderr a relaying CLI can print: kilocode's banner and tool trace naming
#: what the model read or ran, then the tool's own output.
RELAYED_TRACES = (
    "> code · stub-model\n| Read docs/bubblewrap.md\nConnection reset by peer",
    "> code · stub-model\n| Bash cat missing\nbash: command not found\nConnection reset by peer",
    "| Bash ./probe.sh\nbwrap: loopback: Failed RTM_NEWADDR: Operation not permitted\n",
)


def _extra_dispatches(provider: str, status: object) -> int:
    """How many more times one lane dispatches a run that keeps ending on
    ``status``: the ladder both lanes use, with this adapter's envelope."""
    ladder = RetryLadder(
        timeout=100000, retries=1,
        envelope=getattr(_PROVIDERS[provider], "internal_retries", None),
        now=lambda: 0, jitter=lambda: 0,
    )
    for extra in range(30):
        if ladder.next_attempt() is None or not ladder.record(status).retry:
            return extra
    raise AssertionError(f"{provider}: {status} retries without bound")


@pytest.mark.parametrize("stderr", MEASURED_BWRAP_LINES)
@pytest.mark.parametrize("provider", CLI_STDERR)
def test_bubblewraps_own_refusal_on_stderr_still_fails_the_batch(provider, stderr):
    """The measured lines, on the stream bubblewrap writes them to, under any
    answer of an adapter whose stderr is the CLI's own:
    ``sandbox_unavailable`` on both lanes."""
    stdout = COMPLETE[provider](PROSE["neutral"])
    assert _lanes(provider, stdout, 1, stderr) == (
        STATUS_SANDBOX_UNAVAILABLE, STATUS_SANDBOX_UNAVAILABLE,
    )


@pytest.mark.parametrize("stderr", MEASURED_BWRAP_LINES + RELAYED_TRACES)
@pytest.mark.parametrize("provider", RELAYED_STDERR)
def test_a_relaying_adapters_stderr_never_fails_the_batch(provider, stderr):
    """On every adapter whose stderr also carries a
    model's or a tool's words: a tool's ``bubblewrap``, its ``command not
    found``, even a whole bwrap line, are no batch-fatal verdict. The run
    keeps its bounded retry, at most one extra dispatch per lane."""
    stdout = BUILD["cut"](provider, PROSE["neutral"])
    lanes = _lanes(provider, stdout, 1, stderr)
    assert not set(lanes) & FATAL_STATUSES, (provider, stderr, lanes)
    assert all(_extra_dispatches(provider, status) <= 1 for status in lanes), lanes


@pytest.mark.parametrize("line", MEASURED_BWRAP_LINES)
@pytest.mark.parametrize("provider", sorted(COMPLETE))
def test_a_refusal_merged_into_stdout_is_a_bounded_retry(provider, line):
    """No provider run merges stderr into stdout (providers.md#launch-
    diagnostics), and this pins what such a capture would be: a refusal on
    stdout is model-bearing text, never a batch-fatal verdict, so it costs at
    most one extra dispatch per lane."""
    for stdout in (line, line + BUILD["cut"](provider, PROSE["neutral"])):
        lanes = _lanes(provider, stdout, 1, "")
        assert not set(lanes) & FATAL_STATUSES, (provider, stdout[:60], lanes)
        assert all(_extra_dispatches(provider, status) <= 1 for status in lanes), lanes


# ---------------------------------------------------------------------------
# A cut envelope is no answer
# ---------------------------------------------------------------------------

#: A review's answer as JSON: its nested objects are what a text extraction
#: over a cut capture would otherwise return.
JSON_ANSWER = json.dumps([
    {"severity": "HIGH", "title": "first", "detail": {"file": "a.py", "line": 3}},
    {"severity": "LOW", "title": "second", "detail": {"file": "b.py", "line": 9}},
])

#: The adapters whose JSON mode prints one envelope (or one array of records)
#: around the answer. kilocode's stdout is the answer itself, not an envelope.
SINGLE_DOCUMENT = (
    "agy", "claude-code", "cursor-agent", "fx", "gemini", "goose", "grok",
    "qwen-code", "vibe",
)


@pytest.mark.parametrize("provider", SINGLE_DOCUMENT)
def test_every_cut_of_an_envelope_is_a_parse_failure(provider):
    """Wherever the capture stops, inside the answer or after it, an envelope
    that does not close is refused: never its bookkeeping, goose's echoed
    user turn or a piece of the answer served as the answer."""
    adapter = _PROVIDERS[provider]
    stdout = COMPLETE[provider](JSON_ANSWER)
    assert json.loads(stdout) is not None
    assert adapter.parse_output(stdout, "")["data"] == json.loads(JSON_ANSWER)
    served = {
        end: adapter.parse_output(stdout[:end], "").get("data")
        for end in range(1, len(stdout))
        if stdout[:end].strip() != stdout.strip()
        and adapter.parse_output(stdout[:end], "").get("success") is True
    }
    assert not served, (provider, {end: str(data)[:80] for end, data in list(served.items())[:3]})


#: The adapters that read their stdout by its envelope's signature
#: (``ENVELOPE_SIGNATURE``, ``base.read_stdout``).
PARTIAL_ENVELOPE_READERS = ("agy", "claude-code", "cursor-agent", "gemini", "goose", "grok")

#: What can come ahead of the envelope. ANSI colour and a JSON log line leave
#: it where the CLI writes it; a ``[``-led log line, an inline prefix and a
#: banner make the capture ambiguous (``base.STDOUT_AMBIGUOUS``).
PREFIXES = {
    "bare": "",
    "ansi": "\x1b[32m",
    "json-log": '{"level":"info"}\n',
    "log": "[INFO] starting\n",
    "inline": "notice: ",
    "banner": "Loaded 3 extensions\n",
}
#: The prefixes that keep the envelope where the CLI writes it.
IN_PLACE = ("bare", "ansi", "json-log")

#: Every shape of these adapters that stops inside the envelope (the others'
#: unfinished shape is a whole envelope flagged as failed).
_CUT_ENVELOPES = [(provider, "cut") for provider in PARTIAL_ENVELOPE_READERS] + [
    ("goose", "unfinished"),
]


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("provider,shape", _CUT_ENVELOPES,
                         ids=[f"{p}-{s}" for p, s in _CUT_ENVELOPES])
def test_a_cut_envelope_is_an_incomplete_parse_failure_on_both_lanes(provider, shape, prefix):
    """A success capture cut inside its envelope, after any
    banner, log line, inline prefix or colour. The parse fails, as
    ``incomplete`` where the envelope sits where the CLI writes it (the ask
    lane keeps any text as partial), and neither lane reads it as ``ok``:
    the executor keeps its parse failure's one retry."""
    stdout = PREFIXES[prefix] + BUILD[shape](provider, JSON_ANSWER)
    with pytest.raises(ValueError):
        json.loads(stdout)
    parsed = _PROVIDERS[provider].parse_output(stdout, "")
    assert parsed["success"] is False and parsed["data"] is None
    assert bool(parsed.get("incomplete")) is (prefix in IN_PLACE)
    lane_a, lane_b = _lanes(provider, stdout)
    assert STATUS_OK not in (lane_a, lane_b)
    assert (lane_a, lane_b) == _lanes(provider, BUILD[shape](provider, PROSE["neutral"]))
    assert lane_b == STATUS_UNPARSEABLE and _retried(provider, lane_b)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("provider", PARTIAL_ENVELOPE_READERS)
def test_no_answer_decides_either_lane_behind_any_prefix(provider, prefix, shape):
    """The classifier reads stdout by the parser's rule
    (``base.diagnostic_streams``), so a prefix or colour ahead of the
    envelope never hands the model's answer to it: every kind of answer
    leaves both lanes where the neutral one does."""
    baseline = _lanes(provider, PREFIXES[prefix] + BUILD[shape](provider, PROSE["neutral"]))
    for kind in KINDS:
        found = _lanes(provider, PREFIXES[prefix] + BUILD[shape](provider, PROSE[kind]))
        assert found == baseline, (provider, prefix, shape, kind, found, baseline)


#: Text around a whole envelope that leaves it where the CLI writes it:
#: trailing text, colour, JSON log lines.
SURROUNDINGS = {
    "trailing-line": ("", "\nnotice"),
    "trailing-log": ("", "\n[INFO] done in 3.2s"),
    "ansi": ("\x1b[32m", "\x1b[0m"),
    "json-logs": ('{"level":"info"}\n', '\n{"level":"info","msg":"done"}'),
}

#: Text ahead of a whole envelope that does not.
LEADING_TEXT = {
    "leading-line": "notice\n",
    "log-line": "[INFO] starting\n",
    "inline": "notice: ",
}


@pytest.mark.parametrize("surround", SURROUNDINGS)
@pytest.mark.parametrize("provider", PARTIAL_ENVELOPE_READERS)
def test_a_whole_envelope_amid_other_text_is_unwrapped(provider, surround):
    """A complete envelope where the CLI writes it, with text around it, is
    read as the whole stdout would be: its answer field, never the envelope
    or its bookkeeping."""
    adapter = _PROVIDERS[provider]
    envelope = COMPLETE[provider](JSON_ANSWER)
    before, after = SURROUNDINGS[surround]
    parsed = adapter.parse_output(before + envelope + after, "")
    assert parsed["success"] is True, (provider, surround, parsed.get("error"))
    assert parsed["data"] == adapter.parse_output(envelope, "")["data"] == json.loads(JSON_ANSWER)


@pytest.mark.parametrize("lead", LEADING_TEXT)
@pytest.mark.parametrize("provider", PARTIAL_ENVELOPE_READERS)
def test_a_whole_envelope_behind_plain_text_is_refused(provider, lead):
    """Plain text ahead of the envelope could be a model's answer quoting
    one, so the capture is ambiguous: no answer, a bounded retry."""
    stdout = LEADING_TEXT[lead] + COMPLETE[provider](JSON_ANSWER)
    parsed = _PROVIDERS[provider].parse_output(stdout, "")
    assert parsed["success"] is False and parsed["data"] is None, (provider, lead)
    lane_a, lane_b = _lanes(provider, stdout)
    assert STATUS_OK not in (lane_a, lane_b) and _retried(provider, lane_b)


#: Stdout with no envelope in it: a fenced or prose-wrapped answer.
ORDINARY_ANSWERS = {
    "fenced-object": '```json\n{"findings": []}\n```',
    "prose-then-object": 'Here is the result:\n{"findings": []}',
    "object-then-prose": '{"findings": []}\nDone.',
    "fenced-array": "```json\n" + JSON_ANSWER + "\n```",
    "prose-then-array": "Review:\n" + JSON_ANSWER,
}


@pytest.mark.parametrize("answer", ORDINARY_ANSWERS)
@pytest.mark.parametrize("provider", PARTIAL_ENVELOPE_READERS)
def test_an_answer_with_no_envelope_signature_goes_to_text_extraction(provider, answer):
    """On all six fallback readers: without the adapter's
    envelope signature, stdout gets the ordinary text extraction, unchanged."""
    stdout = ORDINARY_ANSWERS[answer]
    parsed = _PROVIDERS[provider].parse_output(stdout, "")
    assert parsed == extract_json_from_text(stdout.strip(), prefer_arrays=True)
    assert parsed["success"] is True and parsed["data"] in (
        {"findings": []}, json.loads(JSON_ANSWER),
    ), (provider, answer, parsed)


@pytest.mark.parametrize("provider", PARTIAL_ENVELOPE_READERS)
def test_the_signature_recognises_every_captured_envelope(provider):
    """Each signature is the opening of the adapter's captured envelopes, so
    every committed JSON envelope is read whole, behind a JSON log line too."""
    adapter = _PROVIDERS[provider]
    assert "ENVELOPE_SIGNATURE" in vars(type(adapter)), provider
    envelopes = []
    for path in sorted((FIXTURES / provider).glob("*.stdout*")):
        try:
            envelope = json.loads(path.read_text())
        except ValueError:
            continue
        if isinstance(envelope, dict):
            envelopes.append((path.name, envelope, path.read_text()))
    assert envelopes, provider
    for name, envelope, text in envelopes:
        for before in ("", '{"level":"info"}\n'):
            shape = read_stdout(before + text, adapter.ENVELOPE_SIGNATURE)
            assert shape.kind == STDOUT_ENVELOPE and shape.value == envelope, name


@pytest.mark.parametrize("stderr,exit_code", [
    ("bash: copilot: command not found", 127),
    ("bash: copilot: command not found", 1),
    ("exec: \"copilot\": executable file not found in $PATH", 1),
])
def test_the_launchers_own_refusal_is_still_spawn_failed(stderr, exit_code):
    assert _lanes("copilot", "", exit_code, stderr) == (
        STATUS_SPAWN_FAILED, STATUS_SPAWN_FAILED,
    )
