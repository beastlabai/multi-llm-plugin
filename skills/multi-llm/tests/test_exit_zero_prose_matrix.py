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
from utils.providers.base import classify_streams  # noqa: E402
from utils.retry_ladder import (  # noqa: E402
    BACKOFF_STATUSES,
    FATAL_STATUSES,
    STATUS_BUDGET_EXHAUSTED,
    STATUS_CONTEXT_LENGTH_EXCEEDED,
    STATUS_SPAWN_FAILED,
    STATUS_UNPARSEABLE,
)

FIXTURES = Path(__file__).parent / "fixtures"

#: One answer per kind. ``neutral`` is the baseline every other kind must
#: match; each other one would decide a verdict if the tables read it.
PROSE = {
    "neutral": "The implementation needs a focused regression test.",
    "number": "I reviewed 429 files.",
    "rate": "The plan says to retry HTTP 429 Too Many Requests and rate limit errors.",
    "auth": "The plan says 401 Unauthorized means an invalid api key.",
    "context": "The plan documents the maximum context length of 128000 tokens.",
    "launch": "The plan documents command not found failures.",
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
    "agy": lambda p: json.dumps({"status": "SUCCESS", "response": p}),
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
    "agy": lambda p: json.dumps({"status": "ERROR", "error": "Request failed.", "response": p}),
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
    assert classify_streams(PROSE[kind], "", 1) is not None


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("provider", sorted(COMPLETE))
def test_no_answer_decides_either_lane(provider, shape):
    adapter = _PROVIDERS[provider]
    baseline = BUILD[shape](provider, PROSE["neutral"])
    lane_a, lane_b = _lanes(provider, baseline)
    assert lane_a not in FORBIDDEN and lane_b not in FORBIDDEN, (provider, shape, lane_a, lane_b)
    # A cut envelope can parse (goose's echoed prompt, claude-code's
    # bookkeeping hold JSON of their own): then nothing is classified at all,
    # and the equality below still holds.
    failed = adapter.parse_output(baseline, "").get("success") is not True
    if failed and not adapter.EXIT0_CLASSIFIABLE:
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


@pytest.mark.parametrize("stderr,exit_code", [
    ("bash: copilot: command not found", 127),
    ("bash: copilot: command not found", 1),
    ("exec: \"copilot\": executable file not found in $PATH", 1),
])
def test_the_launchers_own_refusal_is_still_spawn_failed(stderr, exit_code):
    assert _lanes("copilot", "", exit_code, stderr) == (
        STATUS_SPAWN_FAILED, STATUS_SPAWN_FAILED,
    )


@pytest.mark.parametrize("stdout,expected", [
    # The CLI's own plain text, with no envelope: kept as written.
    ("Error: Authentication required. Please run 'agent login' first.",
     "Error: Authentication required. Please run 'agent login' first."),
    # An envelope: the answer field goes, everything else stays.
    ('{"is_error": true, "result": "HTTP 429"}', '{"is_error": true}'),
    # Plain lines before and after an envelope are the CLI's and stay.
    ('Loaded cached credentials.\n{"result": "x", "stats": 1}\ntrailer',
     'Loaded cached credentials.\n{"stats": 1}\ntrailer'),
    # An envelope that never closes holds the answer somewhere: dropped.
    ('note\n{"is_error": true, "result": "The plan says HTTP 429', "note\n"),
])
def test_envelope_without_answer(stdout, expected):
    from utils.providers.base import envelope_without_answer

    assert envelope_without_answer(stdout, ("result",)) == expected
