"""Tests for failure classification and description across adapters.

One of the per-provider test files; helpers that several of them share live in
``provider_test_support.py``. See docs/development/providers.md.
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.base import LLMProvider
from utils.retry_ladder import STATUS_RATE_LIMITED
from tests.provider_test_support import (
    C1_FIXTURES,
    OC_GRADED_CAPTURES,
    _envelope_sweep_names,
    _oc_stream,
)


#: `TestClassifyFailure`'s case tables, lifted to module scope so
#: `TestMatchStreams` twins every row through `match_streams` too: a case added
#: here is graded by both tests without anyone copying it.
CLASSIFY_FAILURE_BARE_429_NOT_RATE_LIMIT = [
    '{"type":"step_finish","cost":0.0429,"tokens":{"output":14290}}',
    "total cost 1.0429 for 4290 tokens",
    "429 things were reviewed",
]


CLASSIFY_FAILURE_REAL_429 = [
    "HTTP 429 Too Many Requests",
    '{"error":{"status":429,"message":"slow down"}}',
    "Error: 429",
    "Requests are throttled: status code 429",
]


CLASSIFY_FAILURE_TEXTUAL_RATE = [
    "rate limit exceeded",
    "the provider reported rate limits",
    "API rate-limited, retry later",
    "quota exceeded for this model",
    '{"type":"overloaded_error"}',
]


CLASSIFY_FAILURE_PROSE_AUTH_WORD = [
    "the unauthorized branch in review.py returns early",
    "T004 adds a test for unauthorized callers",
]


CLASSIFY_FAILURE_REAL_LOGIN_NOTICE = [
    "Not logged in. Please log in with `codex login`.",
    "401 Unauthorized",
    "Error: unauthorized",
    "authentication required",
    "invalid api key",
    "invalid_api_key",
    "no credentials found for provider",
]


#: `(stdout, stderr)` pairs `TestClassifyFailure` pins outside a parametrize.
CLASSIFY_FAILURE_PAIRS = [
    ("", ""),
    ("   \n ", ""),
    ("429 too many requests\nnot logged in", ""),
]


CLASSIFY_FAILURE_STREAMS = (
    CLASSIFY_FAILURE_BARE_429_NOT_RATE_LIMIT
    + CLASSIFY_FAILURE_REAL_429
    + CLASSIFY_FAILURE_TEXTUAL_RATE
    + CLASSIFY_FAILURE_PROSE_AUTH_WORD
    + CLASSIFY_FAILURE_REAL_LOGIN_NOTICE
)


class TestClassifyFailure:
    """D3i/D3k: the classifier must read a stream, not find digits in one."""

    @pytest.fixture
    def provider(self):
        from utils.provider_registry import get_provider

        return get_provider("opencode")

    @pytest.mark.parametrize("stream", CLASSIFY_FAILURE_BARE_429_NOT_RATE_LIMIT)
    def test_a_bare_429_inside_a_number_is_not_a_rate_limit(self, provider, stream):
        """A misread here is retried WITHOUT consuming the retry budget."""
        assert provider.classify_failure(stream, "") is None

    @pytest.mark.parametrize("stream", CLASSIFY_FAILURE_REAL_429)
    def test_a_real_429_is_still_a_rate_limit(self, provider, stream):
        assert provider.classify_failure("", stream) == "rate_limited"

    @pytest.mark.parametrize("stream", CLASSIFY_FAILURE_TEXTUAL_RATE)
    def test_the_textual_rate_patterns_keep_matching(self, provider, stream):
        assert provider.classify_failure("", stream) == "rate_limited"

    @pytest.mark.parametrize("stream", CLASSIFY_FAILURE_PROSE_AUTH_WORD)
    def test_prose_quoting_an_auth_word_is_not_an_auth_failure(self, provider, stream):
        """auth_failed is batch-fatal, so a quoted word must never trigger it."""
        assert provider.classify_failure(stream, "") is None

    @pytest.mark.parametrize("stream", CLASSIFY_FAILURE_REAL_LOGIN_NOTICE)
    def test_a_real_login_notice_is_still_auth_failed(self, provider, stream):
        assert provider.classify_failure("", stream) == "auth_failed"

    def test_an_empty_stream_classifies_nothing(self, provider):
        for stdout, stderr in CLASSIFY_FAILURE_PAIRS[:2]:
            assert provider.classify_failure(stdout, stderr) is None

    def test_auth_wins_over_a_rate_limit_in_the_same_stream(self, provider):
        stdout, stderr = CLASSIFY_FAILURE_PAIRS[2]
        assert provider.classify_failure(stdout, stderr) == "auth_failed"


#: The adapters that override ``describe_failure``, each with **the channel its
#: CLI really writes its diagnosis on**. It is a two-column table rather than a
#: set because the channel is the thing being pinned: the value is what
#: ``test_each_override_reads_only_the_channel_it_declares`` measures against
#: that adapter's committed captures, so a row cannot be added without saying
#: which stream it reads and being held to it.
#:
#: ``opencode`` and ``muse`` report their errors on **stdout** and leave stderr
#: uninformative — the shape the hook was written for. ``pi`` and ``gemini``
#: joined on 2026-08-20 (C4) as the exact inverse: both took ``stderr`` in
#: ``parse_output`` and never mentioned it again, so 100% of the diagnosis was
#: discarded while the CLI exited non-zero with the vendor's full error object
#: sitting unused. The hook is therefore a *channel* rule, not a stream name.
#:
#: ``kilocode`` joined on 2026-09-10 on the stderr side, from a live failure
#: rather than a probe: a `review-code` slot on a dead account whose stderr read
#: ``Error: User not found.`` in ANSI colour and whose operator-visible line read
#: ``kilocode exited with code 1``. It ships **no ``.txt`` capture pairs**, so
#: ``test_each_override_reads_only_the_channel_it_declares`` skips it here and
#: ``tests/test_stderr_diagnosis.py`` makes the same assertion over the literals
#: that file commits — the declared channel is graded either way, and the row is
#: what stops the override landing without declaring one.
#: ``opencode`` is the third shape (D5): ``stdout`` is still the declared
#: channel — blanking it yields ``None`` — but since 1.18.x the stdout event can
#: be a wrapper (``UnknownError: Unexpected server error``) whose cause is a
#: ``level=ERROR … error="…"`` line on stderr, and ``describe_failure`` appends
#: that value in parentheses. So stderr is declared *enrichment* that may change
#: the text: blanking it must leave a non-``None`` answer that is a **prefix** of
#: the composed one. Encoded as a value rather than an exemption, so the row
#: stays graded against the rig's ``invalid_model`` capture.
#:
#: ``claude-code`` and ``cursor-agent`` joined on 2026-09-17, from one live
#: ``review-code`` fan-out whose two failed slots were both recorded as the bare
#: ``<provider> exited with code 1``. ``claude-code`` is plain **stdout**: the
#: envelope carries ``api_error_status`` and the vendor sentence, stderr is 0
#: bytes on all eleven committed captures, and the override renders
#: ``result_envelope.describe_refused_envelope``'s line (a 491 s slot had died
#: on a 429 usage cap with $25.05 spent, and nothing on screen said so).
#: ``cursor-agent`` is the **fourth** shape: a run that reached the backend
#: answers with the same envelope on stdout, so stdout is asked first — but a
#: run that never got one leaves its reason as stderr's **last** line
#: (``RetriableError: [resource_exhausted] Error`` under three reconnect
#: lines, stdout empty), and that channel answers *only when stdout has
#: nothing to say*. The graded rule is therefore precedence, not enrichment:
#: with stdout answering, blanking stderr changes nothing; with stdout blank,
#: the answer is stderr's own. Its committed captures are all stdout-shaped
#: (stderr 0 bytes on both), and ``tests/test_failed_slot_reporting.py``
#: grades the stderr leg over the measured kimi lines.
STDOUT_THEN_STDERR = "stdout-then-stderr"


STDOUT_ELSE_STDERR = "stdout-else-stderr"


_DESCRIBE_FAILURE_CHANNELS = {
    "fx": STDOUT_ELSE_STDERR,
    "opencode": STDOUT_THEN_STDERR,
    "muse": "stdout",
    # pi moved from "stderr" with `--mode json` (plan §7, T7.6), to omp's shape
    # and for omp's reason, measured at 0.87.1 (tests/fixtures/pi/manifest.yaml):
    # every in-stream failure is an assistant `message_end` on stdout carrying
    # the vendor's `errorMessage` with ZERO bytes on stderr and exit 0, while a
    # thrown pre-flight (`preflight-no-key`) is a one-line `session` header on
    # stdout — silent — with the diagnosis on stderr and exit 1. Precedence, not
    # enrichment: stderr may never edit an answer the stream already gave.
    "pi": STDOUT_ELSE_STDERR,
    "gemini": "stderr",
    "kilocode": "stderr",
    "claude-code": "stdout",
    "cursor-agent": STDOUT_ELSE_STDERR,
    # omp is cursor-agent's shape, and the asymmetry that makes it precedence
    # rather than enrichment is measured: in `--mode json` an API failure is a
    # JSONL stream on stdout with ZERO bytes on stderr, while a pre-flight fault
    # (bad model, missing key, rejected --tools name, missing overlay) is zero
    # bytes on stdout with a plain-text diagnosis on stderr. STDOUT_THEN_STDERR
    # would fail those pre-flight rows by construction — it requires the
    # declared stdout channel to answer on its own — and composing the two would
    # let stderr edit an answer the stream had already given.
    "omp": STDOUT_ELSE_STDERR,
    # codex names one failure only — the exit-101 panic a globally enabled
    # `use_legacy_landlock` causes on every workspace-write dispatch (landlock
    # rung, constraint 2) — and a Rust panic is printed on stderr.
    "codex": "stderr",
}


def _describe_failure_overrides():
    """The roster read off the CLASSES, never off the table above.

    Parametrizing over the registry rather than over a literal roster is what
    makes a new override fail by name on the day it lands instead of whenever
    somebody remembers to widen a set.
    """
    from utils.provider_registry import _PROVIDERS

    return {
        name
        for name, provider in _PROVIDERS.items()
        if type(provider).describe_failure is not LLMProvider.describe_failure
    }


def _provider_capture_rows(name):
    """``(row, stdout, stderr)`` for every committed ``.txt`` capture pair."""
    directory = Path(__file__).parent / "fixtures" / name
    if not directory.is_dir():
        return []
    rows = []
    for row in sorted({path.name.split(".")[0] for path in directory.glob("*.txt")}):
        streams = []
        for key in ("stdout", "stderr"):
            path = directory / f"{row}.{key}.txt"
            streams.append(path.read_text(encoding="utf-8") if path.is_file() else "")
        rows.append((row, streams[0], streams[1]))
    return rows


class TestDescribeFailureChannels:
    """The opt-in hook that recovers a failure reason from **either** channel.

    Renamed from ``TestDescribeFailure`` on 2026-08-20, with
    ``test_stdout_error_providers_are_the_only_overrides`` renamed with it: both
    names encoded the base docstring's old framing, that this is the hook for *"a
    provider that reports its errors on stdout rather than stderr"*. C4 made that
    false in the most direct way available — by adding the two adapters that are
    its mirror image — so what is pinned here is the channel each override
    declares, measured, rather than one stream's name.

    D5 added the third shape the table has to name: ``stdout-then-stderr``,
    where the declared channel is still one stream (blanking it is ``None``)
    and the other is enrichment that may *extend* the answer but never replace
    it — blanking the enrichment leaves a prefix of the composed text.
    """

    @pytest.mark.parametrize("name", _envelope_sweep_names())
    def test_an_adapter_without_an_override_inherits_none(self, name):
        """No editorial from an adapter that has none to add.

        The base returns ``None`` and most adapters inherit it, so the caller
        sees the exit code plus ``details["stderr"]`` and nothing invented.
        Parametrized over the whole registry, so a *new* adapter is covered on
        the day it is registered.
        """
        from utils.provider_registry import _PROVIDERS

        provider = _PROVIDERS[name]
        if name in _describe_failure_overrides():
            pytest.skip(f"{name} overrides describe_failure; see the channel test")

        assert provider.describe_failure("some stdout", "some stderr") is None

    def test_the_override_roster_and_the_channel_table_agree(self):
        """Neither half may move without the other.

        The roster is derived from the classes; the table is hand-written and
        carries the channel. Equality between them is what stops an override
        landing with no declared channel — and stops a stale row outliving the
        override it describes.
        """
        assert _describe_failure_overrides() == set(_DESCRIBE_FAILURE_CHANNELS)

    @pytest.mark.parametrize("name", sorted(_DESCRIBE_FAILURE_CHANNELS))
    def test_each_override_reads_only_the_channel_it_declares(self, name):
        """The rule, measured on committed bytes rather than asserted in prose.

        For every capture this adapter ships: blanking the **other** channel
        must not change the answer, and blanking the **declared** channel must
        make it ``None``. That is "stderr is the diagnosis, stdout is the
        answer" — or its mirror — expressed as an assertion, and it is what
        stops a future edit quietly reading the answer channel and reporting a
        model's own words as the failure reason. gemini's ``exit-zero-failure``
        is the row where it bites: 1,174 bytes of (false) answer on stdout and
        193 bytes of real reason on stderr, in one capture.
        """
        from utils.provider_registry import _PROVIDERS

        provider = _PROVIDERS[name]
        declared = _DESCRIBE_FAILURE_CHANNELS[name]
        rows = _provider_capture_rows(name)
        if not rows:
            pytest.skip(f"{name} ships no .stdout.txt/.stderr.txt capture pairs")

        if name == "fx":
            # fx carries terminal errors inside its JSON stdout envelope and
            # only uses anchored stderr when that envelope cannot supply a
            # diagnosis. The generic stream-channel table cannot model that
            # nested envelope; the fx-specific fixture assertion below does.
            pytest.skip("fx envelope channel behavior is tested separately")

        for row, stdout, stderr in rows:
            both = provider.describe_failure(stdout, stderr)
            if declared == STDOUT_THEN_STDERR:
                # The declared channel alone still answers, and the composed
                # answer only ever extends it; the enrichment alone is nothing.
                stdout_only = provider.describe_failure(stdout, "")
                assert both is not None and stdout_only is not None, (
                    f"{name}/{row}: the declared stdout channel recovered nothing"
                )
                assert both.startswith(stdout_only), (
                    f"{name}/{row}: stderr enrichment replaced the stdout answer "
                    f"instead of extending it"
                )
                assert provider.describe_failure("", stderr) is None, (
                    f"{name}/{row}: a reason was recovered from stderr alone, "
                    f"which is enrichment and not the declared channel"
                )
                continue
            if declared == STDOUT_ELSE_STDERR:
                # Precedence: an answer from stdout is the answer, and stderr
                # may not touch it; with stdout silent, stderr speaks alone.
                stdout_only = provider.describe_failure(stdout, "")
                if stdout_only is not None:
                    assert both == stdout_only, (
                        f"{name}/{row}: stderr changed an answer stdout had given"
                    )
                else:
                    assert both == provider.describe_failure("", stderr), (
                        f"{name}/{row}: with stdout silent the answer is not stderr's own"
                    )
                continue
            if declared == "stdout":
                declared_only, other_only = (stdout, ""), ("", stderr)
            else:
                declared_only, other_only = ("", stderr), (stdout, "")

            assert provider.describe_failure(*declared_only) == both, (
                f"{name}/{row}: the answer changed when the {declared} capture "
                f"was presented on its own — the other channel is being read"
            )
            assert provider.describe_failure(*other_only) is None, (
                f"{name}/{row}: a reason was recovered from the channel "
                f"{name} does NOT declare"
            )

    def test_fx_failure_description_uses_its_json_envelope(self):
        """A successful JSON envelope outranks incidental stderr; auth errors
        are read from the JSON result even when stderr repeats the diagnosis."""
        from utils.provider_registry import _PROVIDERS

        provider = _PROVIDERS["fx"]
        fixture_dir = Path(__file__).parent / "fixtures" / "fx"
        # UTF-8 as the adapter decodes the child's streams: the `·` its patterns
        # anchor on is mojibake under Windows' cp1252 default.
        success = provider.describe_failure(
            (fixture_dir / "multi-assistant.stdout.txt").read_text(encoding="utf-8"),
            (fixture_dir / "multi-assistant.stderr.txt").read_text(encoding="utf-8"),
        )
        assert success is None

        stdout = (fixture_dir / "auth-error.stdout.txt").read_text(encoding="utf-8")
        stderr = (fixture_dir / "auth-error.stderr.txt").read_text(encoding="utf-8")
        full = provider.describe_failure(stdout, stderr)
        assert full is not None
        assert full == provider.describe_failure(stdout, "")

    @pytest.mark.parametrize("name", sorted(_DESCRIBE_FAILURE_CHANNELS))
    def test_every_override_survives_garbage_on_both_channels(self, name):
        """Diagnostics must never raise.

        Both callers wrap this hook in ``try``/``except`` precisely because an
        override that threw would mask the failure it exists to explain — so the
        contract is that it returns a string or ``None`` for any bytes at all.
        """
        from utils.provider_registry import _PROVIDERS

        provider = _PROVIDERS[name]

        for stdout, stderr in (("", ""), ("{", "}"), ("\x00", "\x00"),
                               ("[]", "ApiError: {not json"),
                               ("x" * 40000, "9" * 40000)):
            reason = provider.describe_failure(stdout, stderr)

            assert reason is None or isinstance(reason, str)


class TestMatchStreams:
    """T2.5: the quote-only twin of `classify_streams`, and its offsets."""

    @staticmethod
    def _corpus():
        """Every committed manifest row, through its own adapter's reduction
        and tables, plus the opencode captures and the hand-written
        `classify_streams` cases the suite pins elsewhere."""
        from utils.provider_registry import _PROVIDERS

        rows = []
        for manifest_path in sorted(C1_FIXTURES.glob("*/manifest.yaml")):
            import yaml

            provider = manifest_path.parent.name
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            for name, row in (manifest.get("rows") or {}).items():
                if not isinstance(row, dict) or row.get("not_applicable"):
                    continue
                streams = []
                for key in ("stdout", "stderr"):
                    file_name = row.get(key)
                    streams.append(
                        (manifest_path.parent / file_name).read_bytes().decode("utf-8")
                        if file_name else ""
                    )
                rows.append((f"{provider}/{name}", provider, streams[0], streams[1],
                             int(row.get("exit_code") or 0)))
        for row_id, out, err, exit_code, _before, _after in OC_GRADED_CAPTURES:
            rows.append((f"opencode/{row_id}", "opencode", _oc_stream(out),
                         _oc_stream(err), exit_code))
        # `TestClassifyFailure` grades these through opencode's adapter, so they
        # are twinned through that adapter's reduction and tables here.
        for index, stream in enumerate(CLASSIFY_FAILURE_STREAMS):
            rows.append((f"classify-failure/{index}", "opencode", "", stream, 1))
        for index, (out, err) in enumerate(CLASSIFY_FAILURE_PAIRS):
            rows.append((f"classify-failure-pair/{index}", "opencode", out, err, 1))
        for index, (haystack, _expected) in enumerate(LADDER_ORDER_CASES):
            rows.append((f"ladder-order/{index}", None, haystack, "", 1))
        # Edges no other class pins as a `classify_streams` case.
        literal = [
            ("", "", 127),
            ("", "bwrap: setting up uid map: Permission denied\n", 1),
            ("", "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted\n", 1),
            ("event: started\n", "Error: 401 Unauthorized\n", 1),
            ("Ran into this error: 429 rate limit exceeded", "", 0),
            ("cost 0.0429 in 14290 ms", "", 0),
            ("cursor-agent exited with code 1", "", 1),
            ("", "", 0),
            ("   ", "\n", 0),
        ]
        for index, (out, err, exit_code) in enumerate(literal):
            rows.append((f"literal/{index}", None, out, err, exit_code))
        assert all(p in _PROVIDERS for _i, p, *_ in rows if p is not None)
        return rows

    def test_the_status_always_equals_classify_streams(self):
        from utils.provider_registry import _PROVIDERS
        from utils.providers.base import classify_streams, match_streams

        for row_id, provider, stdout, stderr, exit_code in self._corpus():
            kwargs = {}
            if provider is not None:
                adapter = _PROVIDERS[provider]
                stdout, stderr = adapter.vendor_text(stdout, stderr)
                kwargs = dict(
                    auth_patterns=adapter.auth_error_patterns,
                    rate_limit_patterns=adapter.rate_limit_patterns,
                    exit_zero_markers=adapter.exit_zero_failure_markers,
                )
            for out, err, code in ((stdout, stderr, exit_code),
                                   (stderr, stdout, exit_code),
                                   (stdout, stderr, 0), (stdout, stderr, 1)):
                expected = classify_streams(out, err, code, **kwargs)
                match = match_streams(out, err, code, **kwargs)
                got = None if match is None else match.status
                assert got == expected, (row_id, code)
                if match is not None and match.pattern is not None:
                    text = (out if match.channel == "stdout" else err).lower()
                    assert 0 <= match.offset < len(text), row_id

    def test_the_offset_points_at_the_matched_text(self):
        from utils.providers.base import match_streams

        match = match_streams("model said nothing", "quota\nError: 429 Too Many Requests", 1)
        assert (match.status, match.channel) == (STATUS_RATE_LIMITED, "stderr")
        text = "quota\nError: 429 Too Many Requests".lower()
        assert text[match.offset:].startswith(("429", "too many requests"))

        match = match_streams("x\nthe provider said: rate limit exceeded", "", 1)
        assert match.channel == "stdout"
        assert "x\nthe provider said: rate limit exceeded"[match.offset:].startswith(
            "rate limit"
        )

    def test_the_patternless_steps_carry_no_position(self):
        import re

        from utils.providers.base import match_streams

        spawn = match_streams("", "", 127)
        assert (spawn.status, spawn.pattern, spawn.channel, spawn.offset) == (
            "spawn_failed", None, None, None,
        )
        marker = match_streams('{"is_error":true}', "", 0,
                               exit_zero_markers=(re.compile(r'"is_error":\s*true'),))
        assert (marker.status, marker.pattern, marker.channel, marker.offset) == (
            "runner_error", None, None, None,
        )

    def test_classify_streams_does_not_delegate_to_it(self):
        import inspect

        from utils.providers import base

        assert "match_streams" not in inspect.getsource(base.classify_streams)


#: `TestClassifyLadderOrder`'s `(haystack, expected)` cases, at module scope so
#: `TestMatchStreams` twins them through `match_streams` as well.
LADDER_MODEL_NAMED_404 = (
    "API error (status 404 Not Found): The model `stub-model` does not "
    "exist or you do not have access to it."
)


LADDER_SERVER_ERRORS = [
    "API error (status 503)",
    "internal server error",
    # OpenAI's 500 sentence, and codex's gloss of it with either apostrophe
    # (tests/fixtures/*/manifest.yaml `server-error` rows, codex `stdout-error`).
    "Error: The server had an error while processing your request. Sorry about that!",
    "We’re currently experiencing high demand, which may cause temporary errors.",
    "We're currently experiencing high demand, which may cause temporary errors.",
]

#: The two whole-clause server rows, each beside a verdict the ladder consults
#: earlier: the earlier one must still win.
OPENAI_500 = "The server had an error while processing your request. Sorry about that!"
CODEX_500 = "We’re currently experiencing high demand, which may cause temporary errors."


LADDER_ORDER_CASES = [(LADDER_MODEL_NAMED_404, "invalid_model")] + [
    (haystack, "server_error") for haystack in LADDER_SERVER_ERRORS
]


class TestClassifyLadderOrder:
    """C2: invalid-model is consulted BEFORE server-error, and stays that way.

    ``\\bapi error\\b`` is a member of ``SERVER_ERROR_MARKERS``, and several
    vendors wrap a model-named 404 in exactly that phrase. While the server
    table was asked first, a *terminal* config error landed on ``server_error``
    — a **retryable** category — and burned the retry budget on a condition no
    retry can fix. A model-named 404 is strictly more specific than "api
    error", so the specific table is asked first.

    These tests pin the ORDER, not one vendor's wording, because the defect was
    an ordering fact rather than a missing pattern: no adapter gained a pattern
    in C2.
    """

    def test_a_model_named_404_wrapped_in_api_error_is_invalid_model(self):
        """The exact collision: both tables match, and the order decides."""
        from utils.providers.base import classify_streams

        assert classify_streams(LADDER_MODEL_NAMED_404, "", 1) == "invalid_model"

    def test_a_server_error_carrying_no_model_name_is_still_server_error(self):
        """Reordering must not launder a genuine 5xx into a config error."""
        from utils.providers.base import classify_streams

        for haystack in LADDER_SERVER_ERRORS:
            assert classify_streams(haystack, "", 1) == "server_error"

    def test_the_two_tables_are_consulted_in_the_declared_order(self):
        """Source-order guard, so a later reshuffle is red rather than silent.

        Asserted against the function's own body rather than against a sample,
        because a sample only catches the collisions somebody thought of.
        """
        import inspect

        from utils.providers import base

        body = inspect.getsource(base.classify_streams)
        invalid_at = body.index("INVALID_MODEL_MARKERS, haystack")
        server_at = body.index("SERVER_ERROR_MARKERS, haystack")
        assert invalid_at < server_at, (
            "C2: INVALID_MODEL_MARKERS must be consulted before "
            "SERVER_ERROR_MARKERS — a model-named 404 is more specific than "
            "'api error', and classifying it as a retryable server_error burns "
            "the retry budget on a terminal condition."
        )

    def test_the_ladder_docstring_still_documents_the_shipped_order(self):
        """The docstring moved with the code in C2; keep them together."""
        from utils.providers.base import classify_streams

        doc = classify_streams.__doc__ or ""
        assert "invalid model, then server" in doc


class TestTheWholeClauseServerRows:
    """OpenAI's 500 sentence and codex's gloss of it read ``server_error``
    only as whole clauses, and only when nothing earlier in the ladder matched.
    """

    @pytest.mark.parametrize("sentence", [OPENAI_500, CODEX_500])
    @pytest.mark.parametrize("earlier,expected", [
        ("This model's maximum context length is 8192 tokens.", "context_length_exceeded"),
        ("Incorrect API key provided. invalid api key", "auth_failed"),
        ("Rate limit reached for stub-model", "rate_limited"),
        ("Request timed out.", "timeout"),
        ("Connection reset by server", "connection_error"),
        (LADDER_MODEL_NAMED_404, "invalid_model"),
    ], ids=["context", "auth", "rate", "timeout", "connection", "invalid-model"])
    def test_an_earlier_verdict_in_the_same_capture_still_wins(
        self, sentence, earlier, expected
    ):
        from utils.providers.base import classify_streams

        assert classify_streams(sentence, earlier, 1) == expected
        assert classify_streams(f"{earlier}\n{sentence}", "", 1) == expected

    @pytest.mark.parametrize("fragment", [
        "The server had an error.",
        "while processing your request",
        "the server had an error while processing your requests is a typo",
        "We are currently experiencing high demand.",
        "which may cause temporary errors",
        "the review notes the endpoint is experiencing high demand",
    ])
    def test_a_fragment_of_either_clause_is_no_verdict(self, fragment):
        from utils.providers.base import classify_streams

        assert classify_streams(fragment, "", 1) is None
