"""``vendor_text``: an exit 0's answer is never the vendor's verdict.

Both lanes classify a run that exited 0 with no usable answer: the executor in
``_classify_failed_run``, the review fan-out from ``details["failure_category"]``
(``provider_outcome.outcome_from_capture``). On such a run stdout is usually the
model's own answer, prose instead of the JSON asked for, and a review routinely
quotes "429", "401 Unauthorized" or a context limit out of the plan. Each adapter
here prints that answer on stdout, so its ``vendor_text`` drops it, measured from
the shape of its committed ``success`` capture (the failure rows are pinned by
``test_provider_fixture_rows.py``). See
docs/development/delegation.md#read-and-analyze-executor.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.provider_outcome import Capture, outcome_from_capture  # noqa: E402
from utils.provider_registry import _PROVIDERS  # noqa: E402
from utils.providers.base import classify_streams  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"

#: A review that discusses the failure modes it read about. Every table in
#: ``classify_streams`` that matters to the ladder would fire on it.
PROSE = (
    "Reviewed 429 files. The client retries on HTTP 429 Too Many Requests, "
    "stops on 401 Unauthorized: invalid api key, and the plan notes that the "
    "maximum context length is 128000 tokens."
)


def _read(provider: str, name: str) -> str:
    return (FIXTURES / provider / name).read_text(encoding="utf-8")


def _aider() -> str:
    lines = _read("aider", "success.stdout.txt").split("\n")
    return "\n".join(PROSE if line.startswith("[{") else line for line in lines)


def _cursor_agent() -> str:
    envelope = json.loads(_read("cursor-agent", "success.stdout.txt"))
    envelope["result"] = PROSE
    return json.dumps(envelope)


def _gemini() -> str:
    envelope = json.loads(_read("gemini", "success.stdout.txt"))
    envelope["response"] = PROSE
    return json.dumps(envelope, indent=2)


def _grok() -> str:
    envelope = json.loads(_read("grok", "success.stdout.txt"))
    envelope["text"] = PROSE
    return json.dumps(envelope, indent=2)


def _jsonl(provider: str, name: str, rewrite) -> str:
    out = []
    for line in _read(provider, name).split("\n"):
        if not line.strip():
            out.append(line)
            continue
        record = json.loads(line)
        rewrite(record)
        out.append(json.dumps(record))
    return "\n".join(out)


def _cline() -> str:
    def rewrite(event):
        inner = event.get("event") or {}
        for key in ("text", "accumulated"):
            if isinstance(inner.get(key), str):
                inner[key] = PROSE
        if event.get("type") == "run_result":
            event["text"] = PROSE

    return _jsonl("cline", "success.stdout.txt", rewrite)


def _kimi() -> str:
    def rewrite(record):
        if record.get("role") in ("assistant", "tool"):
            record["content"] = PROSE

    return _jsonl("kimi", "multi-assistant.stdout.txt", rewrite)


ANSWERS = {
    "aider": _aider,
    "cline": _cline,
    "cursor-agent": _cursor_agent,
    "gemini": _gemini,
    "grok": _grok,
    "kimi": _kimi,
}


def test_the_prose_is_one_every_table_would_read():
    assert classify_streams(PROSE, "") is not None


@pytest.mark.parametrize("provider", sorted(ANSWERS))
def test_the_answer_is_dropped_and_nothing_is_added(provider):
    adapter = _PROVIDERS[provider]
    stdout = ANSWERS[provider]()
    assert PROSE in stdout
    assert adapter.parse_output(stdout, "")["success"] is False, (
        "the prose is the exit-0 failure this file is about"
    )
    assert classify_streams(
        stdout, "", 0,
        auth_patterns=adapter.auth_error_patterns,
        rate_limit_patterns=adapter.rate_limit_patterns,
        exit_zero_markers=adapter.exit_zero_failure_markers,
    ) is not None, "without the reduction the answer would decide the verdict"

    reduced, stderr = adapter.vendor_text(stdout, "kept")
    assert stderr == "kept"
    assert PROSE not in reduced
    if adapter.ENVELOPE_SIGNATURE:
        # The envelope without its answer fields (`base.diagnostic_streams`).
        assert json.loads(reduced) == adapter.envelope_diagnostics(json.loads(stdout))
    else:
        source_lines = set(stdout.split("\n"))
        assert all(line in source_lines for line in reduced.split("\n") if line), (
            "vendor_text may only remove lines"
        )
    assert adapter.classify_failure(stdout, "", 0) is None


@pytest.mark.parametrize("provider", sorted(ANSWERS))
def test_neither_lane_reads_the_answer_as_a_verdict(provider):
    """The fan-out's category and the executor's reclassification, for the
    same capture: nothing recognised (one transport retry on each), or, for
    an adapter that is not ``EXIT0_CLASSIFIABLE`` (aider), never classified."""
    from utils import subagent_executor as executor

    adapter = _PROVIDERS[provider]
    stdout = ANSWERS[provider]()
    outcome = outcome_from_capture(
        adapter, Capture(stdout=stdout, stderr="", exit_code=0),
        provider_name=provider, model="m",
    )
    assert outcome["success"] is False
    if adapter.EXIT0_CLASSIFIABLE:
        assert outcome["details"]["failure_category"] is None
    else:
        assert "failure_category" not in outcome["details"]
    capture = executor._Capture(argv=["/bin/true"], stdout=stdout, stderr="", exit_code=0)
    assert executor._classify_failed_run(adapter, capture, "m") is None


@pytest.mark.parametrize("provider", sorted(ANSWERS))
def test_a_stream_the_reduction_cannot_read_is_returned_whole(provider):
    adapter = _PROVIDERS[provider]
    for stdout in ("", "not json at all\nHTTP 429 Too Many Requests"):
        assert adapter.vendor_text(stdout, "err") == (stdout, "err")
