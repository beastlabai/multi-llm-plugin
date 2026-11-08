"""One rule for an envelope adapter's stdout: where the envelope sits.

``base.read_stdout`` is read by the six envelope adapters' parsers and by
``base.diagnostic_streams``, their ``vendor_text``: so by the classifier on
every exit code and by the executor's failure quote
(``subagent_executor._classify_failed_run``). Provenance is position, never
keys: the envelope is a JSON value where the CLI writes it (the start of
stdout, or after complete JSON log lines), trailing text after it is the
CLI's, its opening there with no complete value is a cut, and its opening
anywhere else makes the capture ambiguous: refused, a bounded retry.

Each case below broke under a rule that looked for the envelope anywhere: a
JSON log line ahead of it let the model's answer decide ``auth_failed``; a
signature nested in an answer array was served instead of the array; a
broken banner quote hid a cut envelope, whose bookkeeping was then served;
an example envelope in a raw answer replaced the findings after it; and on
a non-zero exit, a prefixed envelope's answer decided ``auth_failed`` and
supplied its quote.
See docs/development/providers.md#reading-a-capture.
"""

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_exit_zero_prose_matrix import (  # noqa: E402
    BUILD,
    COMPLETE,
    FIXTURES,
    JSON_ANSWER,
    PARTIAL_ENVELOPE_READERS,
    PROSE,
    _lanes,
)
from tests.test_provider_fixture_rows import MANIFESTS, _retried  # noqa: E402
from utils.json_extractor import extract_json_from_text  # noqa: E402
from utils.provider_outcome import Capture  # noqa: E402
from utils.provider_registry import _PROVIDERS  # noqa: E402
from utils.providers.base import (  # noqa: E402
    STDOUT_AMBIGUOUS,
    STDOUT_CUT,
    STDOUT_ENVELOPE,
    STDOUT_PLAIN,
    LLMProvider,
    diagnostic_streams,
    read_stdout,
)
from utils.providers.result_envelope import RESULT_ENVELOPE_SIGNATURE  # noqa: E402
from utils.retry_ladder import STATUS_AUTH_FAILED, STATUS_OK  # noqa: E402
from utils.subagent_executor import _classify_failed_run  # noqa: E402

#: The four whose exit 0 is classified (``EXIT0_CLASSIFIABLE``), so a model's
#: answer reaching the classifier could decide a batch-fatal verdict.
EXIT0_READERS = ("claude-code", "cursor-agent", "gemini", "grok")

#: An envelope of each adapter, the shapes Codex's probes used
#: (``u83-r2c/attack.py``), with ``answer`` in its answer field.
_ANSWER_FIELD = {
    "claude-code": "result", "cursor-agent": "result", "gemini": "response",
    "grok": "text", "agy": "response",
}


def _envelope(provider: str, answer: str) -> dict:
    if provider == "goose":
        return {"messages": [{"role": "assistant",
                              "content": [{"type": "text", "text": answer}]}]}
    opening = {
        "claude-code": {"type": "result"}, "cursor-agent": {"type": "result"},
        "gemini": {"session_id": "example"}, "grok": {},
        "agy": {"conversation_id": "example", "status": "SUCCESS"},
    }[provider]
    return {**opening, _ANSWER_FIELD[provider]: answer}


def _parse(provider: str, stdout: str) -> dict:
    return _PROVIDERS[provider].parse_output(stdout, "")


def _refused(provider: str, stdout: str) -> bool:
    parsed = _parse(provider, stdout)
    return parsed["success"] is False and parsed["data"] is None


# ── the rule ────────────────────────────────────────────────────────────────

_ENV = '{"type":"result","result":"x"}'


@pytest.mark.parametrize("stdout,kind,envelope", [
    # The envelope where the CLI writes it.
    (_ENV, STDOUT_ENVELOPE, 0),
    ("  \n\x1b[32m" + _ENV + "\x1b[0m\n", STDOUT_ENVELOPE, 0),
    (_ENV + " trailing log on its line", STDOUT_ENVELOPE, 0),
    (_ENV + "\n[INFO] done\nnotice: " + _ENV, STDOUT_ENVELOPE, 0),
    ('{"level":"info"}\n["startup", 1]\n' + _ENV + '\n{"level":"done"}', STDOUT_ENVELOPE, 2),
    (_ENV + "\n" + _ENV.replace('"x"', '"y"'), STDOUT_ENVELOPE, 1),
    ('{\n  "duration_api_ms": 1,\n  "result": "x"\n}', STDOUT_ENVELOPE, 0),
    # Its opening there, with no complete value: cut.
    ('{"type":"result","result":"x', STDOUT_CUT, None),
    ('{"level":"info"}\n{"type":"result",', STDOUT_CUT, None),
    (_ENV + '\n{"type":"result","res', STDOUT_CUT, None),
    ('{"type":"result",\nActual findings: [1]', STDOUT_CUT, None),
    # Its opening anywhere else: ambiguous.
    ("notice: " + _ENV, STDOUT_AMBIGUOUS, None),
    ("[INFO] starting\n" + _ENV, STDOUT_AMBIGUOUS, None),
    ('{"banner": "' + _ENV, STDOUT_AMBIGUOUS, None),
    ('banner "unterminated\n' + _ENV, STDOUT_AMBIGUOUS, None),
    ('{"level":"info"} notice\n' + _ENV, STDOUT_AMBIGUOUS, None),
    ("```json\n[" + _ENV + ', {"id": 2}]\n```', STDOUT_AMBIGUOUS, None),
    ("[" + _ENV + "]", STDOUT_AMBIGUOUS, None),
    ("Example:\n" + _ENV + '\nActual: [{"id": 1}]', STDOUT_AMBIGUOUS, None),
    ("\x1b[" + _ENV, STDOUT_AMBIGUOUS, None),
    # No opening: the ordinary extraction's.
    ('{"findings": []}', STDOUT_PLAIN, None),
    ('Here is the result:\n{"findings": []}', STDOUT_PLAIN, None),
    ('{"note": ' + json.dumps(_ENV) + "}", STDOUT_PLAIN, None),
    ('{"type": "results"}', STDOUT_PLAIN, None),
    ("", STDOUT_PLAIN, None),
])
def test_the_shape_of_stdout_decides_where_the_envelope_is(stdout, kind, envelope):
    shape = read_stdout(stdout, RESULT_ENVELOPE_SIGNATURE)
    assert shape.kind == kind, shape.kind
    if envelope is not None:
        assert shape.envelope == envelope
        assert shape.value == json.loads(shape.text[slice(*shape.records[envelope][:2])])


def test_with_no_signature_every_stdout_is_plain():
    assert read_stdout("notice: " + _ENV, ()).kind == STDOUT_PLAIN


def _fixture_stdouts(provider: str):
    for path in sorted((FIXTURES / provider).glob("*.stdout*")):
        yield path.name, path.read_text(encoding="utf-8")


@pytest.mark.parametrize("provider", PARTIAL_ENVELOPE_READERS)
def test_no_committed_capture_is_cut_or_ambiguous(provider):
    """The measurement the rule rests on: every committed stdout of the six
    is one envelope with nothing after it, or no envelope at all (an empty
    stdout); no real capture is refused by the cut or ambiguity rules."""
    adapter = _PROVIDERS[provider]
    seen = {STDOUT_ENVELOPE: 0, STDOUT_PLAIN: 0}
    for name, text in _fixture_stdouts(provider):
        shape = read_stdout(text, adapter.ENVELOPE_SIGNATURE)
        assert shape.kind in seen, (name, shape.kind)
        seen[shape.kind] += 1
        if shape.kind == STDOUT_ENVELOPE:
            assert len(shape.records) == 1, name
            assert shape.value == json.loads(text), name
            assert shape.text[shape.records[0][1]:].strip() == "", name
        else:
            assert not text.strip(), name
    assert seen[STDOUT_ENVELOPE], provider
    rows = [row for row in MANIFESTS[provider]["rows"].values() if row.get("stdout")]
    assert rows, provider


# ── the one diagnostic reduction ────────────────────────────────────────────


@pytest.mark.parametrize("stdout,expected", [
    # The CLI's own plain text, with no envelope: kept as written.
    ("Error: Authentication required. Please run 'agent login' first.",
     "Error: Authentication required. Please run 'agent login' first."),
    # An envelope: the answer field goes, the vendor's refusal fields stay.
    ('{"type": "result", "is_error": true, "subtype": "x", "result": "HTTP 429"}',
     '{"type":"result","is_error":true,"subtype":"x"}'),
    # JSON log lines and trailing text are the CLI's, kept as written.
    ('{"level":"info"}\n{"type":"result","result":"x"}\ntrailer',
     '{"level":"info"}\n{"type":"result"}\ntrailer'),
    # Every record that opens on the signature loses its answer.
    ('{"type":"result","result":"a"}\n{"type":"result","result":"b"}',
     '{"type":"result"}\n{"type":"result"}'),
    # Nothing taken out: the record's own bytes.
    ('{"type": "result", "is_error": true}', '{"type": "result", "is_error": true}'),
    # Cut, or ambiguous: nothing of stdout.
    ('{"type":"result","is_error":true,"result":"The plan says HTTP 429', ""),
    ('note\n{"type":"result","is_error":true,"result":"HTTP 429"}', ""),
])
def test_the_reduction_reads_stdout_by_the_parsers_rule(stdout, expected):
    adapter = _PROVIDERS["cursor-agent"]
    assert diagnostic_streams(stdout, "stderr kept", adapter) == (expected, "stderr kept")
    assert adapter.vendor_text(stdout, "stderr kept") == (expected, "stderr kept")


@pytest.mark.parametrize("extra,keeps_result", [
    ({"terminal_reason": "api_error", "is_error": True}, True),
    ({"terminal_reason": "completed"}, False),
    ({"terminal_reason": "aborted", "is_error": True}, False),
    ({"is_error": True, "api_error_status": 500}, False),
])
def test_claude_codes_result_is_read_only_when_the_api_call_failed(extra, keeps_result):
    """``result`` is the CLI's ``API Error: …`` sentence beside
    ``terminal_reason: "api_error"`` (all eight refusing captures), and the
    model's prose otherwise; ``permission_denials`` always goes."""
    envelope = {"type": "result", **extra, "result": "API Error: 401",
                "permission_denials": [{"tool_input": {"command": "cat key"}}]}
    reduced, _ = _PROVIDERS["claude-code"].vendor_text(json.dumps(envelope), "")
    kept = json.loads(reduced)
    assert ("result" in kept) is keeps_result
    assert "permission_denials" not in kept
    assert {k: kept[k] for k in extra} == extra


def test_gooses_reduction_drops_the_echoed_user_turn_alone():
    envelope = {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "HTTP 401 in the plan"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Ran into this error"}]},
    ], "metadata": {}}
    reduced, _ = _PROVIDERS["goose"].vendor_text(json.dumps(envelope), "")
    assert json.loads(reduced) == {**envelope, "messages": envelope["messages"][1:]}


@pytest.mark.parametrize("provider", PARTIAL_ENVELOPE_READERS)
def test_the_reduction_only_removes_on_every_capture(provider):
    """Every committed stdout, alone and behind a JSON log line: the log
    line as written, the envelope as ``envelope_diagnostics`` keeps it
    (a subset of its fields), and stderr untouched."""
    adapter = _PROVIDERS[provider]
    log = '{"level":"info"}\n'
    for name, text in _fixture_stdouts(provider):
        if not text.strip():
            assert adapter.vendor_text(text, "err") == (text, "err")
            continue
        envelope = json.loads(text)
        kept = adapter.envelope_diagnostics(envelope)
        assert set(kept) <= set(envelope), (provider, name)
        for before in ("", log):
            reduced, stderr = adapter.vendor_text(before + text, "err")
            assert reduced.startswith(before) and stderr == "err", (provider, name)
            assert json.loads(reduced[len(before):]) == kept, (provider, name)


# ── point 1 (round 3): a non-zero exit reads the same text ─────────────────

_PREFIXES = {"plain": "", "inline": "notice: ", "json-log": '{"level":"info"}\n'}


@pytest.mark.parametrize("prefix", _PREFIXES)
@pytest.mark.parametrize("provider", EXIT0_READERS)
def test_a_failed_exit_never_reads_the_models_answer(provider, prefix):
    """``u83-r3``: exit 1, empty stderr, a clean envelope whose answer says
    ``HTTP 401 Unauthorized``: no lane reads it as the batch-wide
    ``auth_failed``, whatever is ahead of the envelope."""
    stdout = _PREFIXES[prefix] + COMPLETE[provider]("The plan documents HTTP 401 Unauthorized")
    assert STATUS_AUTH_FAILED not in _lanes(provider, stdout, exit_code=1)
    assert _classify_failed_run(
        _PROVIDERS[provider], Capture(stdout=stdout, stderr="", exit_code=1)
    ) is None


_LONG_PROSE = (
    "ordinary prose " * 400 + "MODEL_QUOTED: HTTP 401 Unauthorized " + "ordinary prose " * 400
)
_ACTUAL = "ACTUAL_DIAGNOSTIC: HTTP 401 Unauthorized; token revoked."


@pytest.mark.parametrize("exit_code", [0, 1])
@pytest.mark.parametrize("prefix", ["", "notice: ", '{"level":"info"}\n', "\x1b[32m"])
@pytest.mark.parametrize("provider", EXIT0_READERS)
def test_a_real_auth_failure_quotes_stderr_not_the_models_example(provider, prefix, exit_code):
    """``u83-r2c/quote_probe.py`` and ``u83-r3``: the answer quotes a 401
    among long prose, stderr holds the CLI's own. On either exit the verdict
    and its quote both come from stderr."""
    stdout = prefix + COMPLETE[provider](_LONG_PROSE)
    verdict = _classify_failed_run(
        _PROVIDERS[provider], Capture(stdout=stdout, stderr=_ACTUAL, exit_code=exit_code)
    )
    assert verdict is not None and verdict[0] == STATUS_AUTH_FAILED, verdict
    assert "ACTUAL_DIAGNOSTIC" in verdict[1] and "MODEL_QUOTED" not in verdict[1], verdict


@pytest.mark.parametrize("stdout", ["cut", "ambiguous"])
@pytest.mark.parametrize("provider", PARTIAL_ENVELOPE_READERS)
def test_a_dropped_stdout_leaves_stderr_read_as_before(provider, stdout):
    """Nothing of a cut or ambiguous stdout is read, and stderr is, under
    the adapter's ``LAUNCH_DIAGNOSTICS``: bubblewrap's measured refusal
    there still fails the batch, except on gemini, whose stderr relays a
    tool's words."""
    text = {
        "cut": BUILD["cut"](provider, PROSE["auth"]),
        "ambiguous": "notice: " + COMPLETE[provider](PROSE["auth"]),
    }[stdout]
    adapter = _PROVIDERS[provider]
    assert adapter.vendor_text(text, "err") == ("", "err")
    lanes = _lanes(provider, text, 1, "bwrap: setting up uid map: Permission denied\n")
    fatal = adapter.LAUNCH_DIAGNOSTICS == "stderr"
    assert (lanes == ("sandbox_unavailable",) * 2) is fatal, lanes


def test_no_adapter_overrides_the_exit_zero_verdict_itself():
    """Both exits read ``vendor_text``'s text, which is what the executor's
    quote reduces through, so an adapter changes what is read there alone."""
    for provider, adapter in _PROVIDERS.items():
        assert type(adapter).classify_exit_zero is LLMProvider.classify_exit_zero, provider


# ── points 2 and 3 (round 3): ambiguous captures are refused ───────────────

_BROKEN_BANNERS = ('{"banner": "', '{"banner": ', '["banner", ', 'banner "unterminated ')


@pytest.mark.parametrize("shape", ["cut", "complete"])
@pytest.mark.parametrize("banner", _BROKEN_BANNERS)
@pytest.mark.parametrize("provider", PARTIAL_ENVELOPE_READERS)
def test_a_broken_inline_banner_never_serves_the_envelopes_insides(provider, banner, shape):
    """``u83-r3``: a banner whose quote or bracket swallows the envelope's
    opening. The opening is still seen as plain text, so the capture is
    refused: never claude-code's ``usage`` or goose's echoed user turn
    served as the answer, never the wrapper as data."""
    stdout = banner + BUILD[shape](provider, JSON_ANSWER)
    assert _refused(provider, stdout), (provider, banner, shape, _parse(provider, stdout))
    lane_a, lane_b = _lanes(provider, stdout)
    assert STATUS_OK not in (lane_a, lane_b) and _retried(provider, lane_b)


@pytest.mark.parametrize("provider", PARTIAL_ENVELOPE_READERS)
def test_an_example_envelope_never_replaces_the_findings(provider):
    """``u83-r3``: raw text with an example envelope, then the real findings.
    The example is never served; the ambiguous capture is refused."""
    stdout = (
        "Example response:\n" + json.dumps(_envelope(provider, '[{"id":"example"}]'))
        + '\nActual findings:\n[{"id":"correct"}]'
    )
    assert read_stdout(stdout, _PROVIDERS[provider].ENVELOPE_SIGNATURE).kind == STDOUT_AMBIGUOUS
    assert _refused(provider, stdout)


# ── round 2: a signature in an answer, a JSON log line ─────────────────────


@pytest.mark.parametrize("provider", PARTIAL_ENVELOPE_READERS)
def test_a_signature_inside_an_answer_array_never_loses_the_array(provider):
    """A raw answer array holding an envelope-shaped element: never one
    element served for the array; the capture is refused."""
    answer = [_envelope(provider, '[{"id":"inner"}]'), {"id": "second"}]
    stdout = "```json\n" + json.dumps(answer) + "\n```"
    assert _refused(provider, stdout)
    assert STATUS_OK not in _lanes(provider, stdout)


@pytest.mark.parametrize("provider", PARTIAL_ENVELOPE_READERS)
def test_a_signature_inside_a_string_field_is_not_an_envelope(provider):
    answer = {"findings": [], "note": json.dumps(_envelope(provider, "x"))}
    stdout = "Review:\n" + json.dumps(answer) + "\nDone."
    assert read_stdout(stdout, _PROVIDERS[provider].ENVELOPE_SIGNATURE).kind == STDOUT_PLAIN
    assert _parse(provider, stdout) == extract_json_from_text(stdout.strip(), prefer_arrays=True)


#: How prose quotes an envelope's opening (``u83-r2c/attack.py`` cuts the dump
#: at its first colon, ``cut_quote_probe.py`` keeps the first field).
_QUOTED_OPENINGS = {
    "claude-code": ('{"type":"result"', '{"type":"result",'),
    "cursor-agent": ('{"type":"result"', '{"type":"result",'),
    "gemini": ('{"session_id":', '{"session_id":'),
    "grok": ('{"text":', '{"text":'),
    "agy": ('{"conversation_id":', '{"conversation_id":'),
    "goose": ('{"messages":', '{"messages":'),
}


@pytest.mark.parametrize("after", ["findings", "envelope", "cut"])
@pytest.mark.parametrize("variant", [0, 1])
@pytest.mark.parametrize("provider", PARTIAL_ENVELOPE_READERS)
def test_prose_quoting_an_envelopes_opening_is_refused(provider, variant, after):
    """A quoted opening, then findings, a real envelope or a cut one: the
    capture is ambiguous, so it is refused rather than guessed at."""
    tail = {
        "findings": '\nActual findings:\n[{"id":"correct"}]',
        "envelope": "\n" + COMPLETE[provider](JSON_ANSWER),
        "cut": "\n" + BUILD["cut"](provider, JSON_ANSWER),
    }[after]
    stdout = f"For example, a response begins: {_QUOTED_OPENINGS[provider][variant]}" + tail
    assert _refused(provider, stdout), (provider, variant, after)


#: Complete JSON before and after the envelope that is not the envelope.
JSON_LOGS = {
    "log-before": ('{"level":"info"}\n', ""),
    "two-logs-before": ('{"level":"info","msg":"start"}\n{"level":"debug"}\n', ""),
    "log-after": ("", '\n{"level":"info","msg":"done"}'),
    "logs-around": ('{"level":"info"}\n', '\n{"level":"info"}'),
    "log-then-notice": ('{"level":"info"}\nnotice: ', ""),
    "array-log-before": ('["startup", 1]\n', ""),
    "ansi-log-before": ('\x1b[2m{"level":"info"}\x1b[0m\n', ""),
}


@pytest.mark.parametrize("exit_code", [0, 1])
@pytest.mark.parametrize("surround", JSON_LOGS)
@pytest.mark.parametrize("provider", EXIT0_READERS)
def test_a_json_log_line_never_hands_the_answer_to_the_classifier(provider, surround, exit_code):
    """Empty stderr, the answer says ``HTTP 401 Unauthorized``: the
    reduction strips the envelope the parser reads, not the log line, so no
    answer moves either lane off the neutral answer's status, and the
    executor forwards no batch-wide verdict."""
    before, after = JSON_LOGS[surround]
    baseline = _lanes(provider, before + COMPLETE[provider](PROSE["neutral"]) + after, exit_code)
    for kind in ("auth", "rate", "context", "launch", "bwrap-line"):
        stdout = before + COMPLETE[provider](PROSE[kind]) + after
        assert _lanes(provider, stdout, exit_code) == baseline, (provider, surround, kind)
    attack = before + json.dumps(_envelope(provider, "This plan should document HTTP 401 Unauthorized")) + after
    assert STATUS_AUTH_FAILED not in _lanes(provider, attack, exit_code)
    assert _classify_failed_run(
        _PROVIDERS[provider], Capture(stdout=attack, stderr="", exit_code=exit_code)
    ) is None


@pytest.mark.parametrize("provider", EXIT0_READERS)
def test_the_reduction_strips_the_envelope_and_keeps_the_log(provider):
    field = _ANSWER_FIELD[provider]
    envelope = _envelope(provider, "HTTP 401 Unauthorized")
    stdout = '{"level":"info"}\n' + json.dumps(envelope)
    reduced, _ = _PROVIDERS[provider].vendor_text(stdout, "")
    kept = {k: v for k, v in envelope.items() if k != field}
    assert reduced == '{"level":"info"}\n' + json.dumps(kept, separators=(",", ":"))


# ── several records: the last is the envelope ──────────────────────────────


@pytest.mark.parametrize("provider", PARTIAL_ENVELOPE_READERS)
def test_of_two_records_the_last_is_read(provider):
    """Two capture-derived records, one per line: the parser reads the
    last, and the reduction strips the answer from both."""
    first = COMPLETE[provider](PROSE["auth"])
    last = COMPLETE[provider](JSON_ANSWER)
    stdout = first + "\n" + last
    shape = read_stdout(stdout, _PROVIDERS[provider].ENVELOPE_SIGNATURE)
    assert shape.kind == STDOUT_ENVELOPE and len(shape.records) == 2
    assert shape.value == json.loads(last)
    assert _parse(provider, stdout)["data"] == json.loads(JSON_ANSWER)
    if provider in EXIT0_READERS:
        reduced, _ = _PROVIDERS[provider].vendor_text(stdout, "")
        assert "401" not in reduced and "severity" not in reduced


@pytest.mark.parametrize("provider", PARTIAL_ENVELOPE_READERS)
def test_a_record_cut_after_a_complete_one_is_refused(provider):
    stdout = COMPLETE[provider](JSON_ANSWER) + "\n" + BUILD["cut"](provider, JSON_ANSWER)
    assert read_stdout(stdout, _PROVIDERS[provider].ENVELOPE_SIGNATURE).kind == STDOUT_CUT
    parsed = _parse(provider, stdout)
    assert parsed["success"] is False and parsed.get("incomplete") is True
    assert _PROVIDERS[provider].vendor_text(stdout, "err") == ("", "err")


# ── one linear pass ────────────────────────────────────────────────────────

_MB = 1 << 20

#: Inputs a retrying decoder would walk again from every bracket.
_HOSTILE = {
    "open-brackets": "[" * _MB + "x",
    "nested-objects": '{"a":' * (_MB // 5) + "x",
    "deep-balanced": "[" * (_MB // 2) + "]" * (_MB // 2),
    "quoted-openings": '{"type":"result",\nX ' * (_MB // 20),
    "many-records": '{"type":"result","result":"x"}\n' * (_MB // 31),
    "cut-envelope": '{"type":"result","result":"' + "y" * _MB,
    "numbers": "[" + "1," * (_MB // 2),
    "openings-in-prose": 'see {"type":"result" ' * (_MB // 20),
}

_HOSTILE_KIND = {
    "quoted-openings": STDOUT_CUT, "many-records": STDOUT_ENVELOPE,
    "cut-envelope": STDOUT_CUT, "openings-in-prose": STDOUT_AMBIGUOUS,
}


def test_the_rule_is_linear_on_a_megabyte():
    """Each hostile 1 MB stdout is read, and reduced, in one pass (well
    under a second each when measured)."""
    adapter = _PROVIDERS["claude-code"]
    started = time.perf_counter()
    for name, text in _HOSTILE.items():
        shape = read_stdout(text, RESULT_ENVELOPE_SIGNATURE)
        assert shape.kind == _HOSTILE_KIND.get(name, STDOUT_PLAIN), name
        diagnostic_streams(text, "", adapter)
    assert time.perf_counter() - started < 30


# A structural check (claude-code's and cursor-agent's ``api_error_status``
# rung, goose's error-part kinds) reads the same stdout decision as the
# verdict: an ambiguous capture gives it nothing either (Codex U83 round 4).
_STRUCTURAL_AMBIGUOUS = {
    "claude-code": {"example": {"type": "result", "result": "example"},
                    "api_error_status": 401},
    "cursor-agent": {"example": {"type": "result", "result": "example"},
                     "api_error_status": 401},
    "goose": {"metadata": {}, "messages": [{"role": "assistant", "content": [
        {"type": "error", "kind": "authentication", "message": "401"}]}],
              "example": {"messages": []}},
}


@pytest.mark.parametrize("exit_code", [0, 1])
@pytest.mark.parametrize("provider", sorted(_STRUCTURAL_AMBIGUOUS))
def test_a_structural_check_never_reads_an_ambiguous_stdout(provider, exit_code):
    adapter = _PROVIDERS[provider]
    stdout = json.dumps(_STRUCTURAL_AMBIGUOUS[provider])
    assert read_stdout(stdout, adapter.ENVELOPE_SIGNATURE).kind == STDOUT_AMBIGUOUS
    assert adapter.classify_failure(stdout, "", exit_code) != STATUS_AUTH_FAILED
    verdict = _classify_failed_run(adapter, Capture(stdout=stdout, stderr="",
                                                   exit_code=exit_code))
    assert "auth" not in str(verdict).lower()
    if hasattr(adapter, "describe_failure") and provider != "goose":
        assert adapter.describe_failure(stdout, "") in (None, "")
