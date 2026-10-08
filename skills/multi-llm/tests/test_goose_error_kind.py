"""goose's failure verdict read off its own error ``kind``, wording second.

goose 1.53.0 writes a provider failure as a ``{"type": "error", "kind": …,
"message": …}`` content part of the last assistant message. ``kind`` is an enum
in goose's own source (``MessageErrorKind`` in
``crates/goose-provider-types/src/conversation/message.rs`` at v1.53.0:
``authentication``, ``contextLengthExceeded``, ``creditsExhausted``, ``other``,
serialized camelCase, built only by ``Message::from_provider_error``), so it is a
structural verdict where the message is prose a release can reword — and the
1.53.0 re-capture did reword the context-overflow sentence, which then matched
no marker and was retried as transport (review 24, pins #4).

The derived envelopes below start from the committed ``auth-error`` capture and
swap only the error part, so the surrounding shape is goose's own bytes; the
neutral message is chosen to match no classifier table — not even goose's own
exit-0 marker, ``Ran into this error:``, which says only "this run failed" — so
a verdict can only come from ``kind``.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.base import classify_streams  # noqa: E402
from utils.providers.goose import GooseProvider  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "goose"

#: Matches no classifier table (asserted below), so only ``kind`` can decide.
NEUTRAL = (
    "The provider refused this request.\n\n"
    "Please retry if you think this is a transient or recoverable error."
)

#: What goose really writes in front of every non-network provider error
#: (``Message::from_provider_error``): the exit-0 marker, ``runner_error``.
PREFIXED = f"Ran into this error: {NEUTRAL}"


def _capture(row):
    return (
        (FIXTURES / f"{row}.stdout.txt").read_text(encoding="utf-8"),
        (FIXTURES / f"{row}.stderr.txt").read_text(encoding="utf-8"),
    )


def _with_error_part(kind, message=NEUTRAL):
    stdout, _stderr = _capture("auth-error")
    envelope = json.loads(stdout)
    last = [m for m in envelope["messages"] if m.get("role") == "assistant"][-1]
    last["content"] = [{"type": "error", "kind": kind, "message": message}]
    return json.dumps(envelope, indent=2)


def test_the_neutral_message_matches_no_table():
    """The precondition every kind row below leans on."""
    assert classify_streams(NEUTRAL, "", 1) is None
    assert GooseProvider().classify_failure(_with_error_part("other"), "", 1) is None


def test_the_1_53_context_overflow_capture_is_terminal():
    """The committed 1.53.0 row: goose's compaction report, at exit 0.

    goose swallowed the upstream 400 and wrote ``Ran into this error trying to
    compact: … exceeds the model's effective context window …`` as a TEXT part
    (no error part, so no ``kind``). Before the fix it classified ``None`` —
    transport-shaped — and Lane B spent paid retries on a condition no retry
    can fix.
    """
    stdout, stderr = _capture("context-overflow")
    assert GooseProvider().classify_failure(stdout, stderr, 0) == (
        "context_length_exceeded"
    )


@pytest.mark.parametrize(
    "kind,category",
    [
        ("contextLengthExceeded", "context_length_exceeded"),
        ("authentication", "auth_failed"),
        # Out of credits is the codebase's quota category, the one
        # `insufficient_quota` and muse's `api error 402` land on.
        ("creditsExhausted", "rate_limited"),
    ],
)
def test_the_error_kind_decides_the_category(kind, category):
    assert GooseProvider().classify_failure(_with_error_part(kind), "", 1) == category


@pytest.mark.parametrize(
    "kind,category",
    [
        ("contextLengthExceeded", "context_length_exceeded"),
        ("authentication", "auth_failed"),
        ("creditsExhausted", "rate_limited"),
    ],
)
def test_the_kind_is_more_specific_than_gooses_exit_zero_marker(kind, category):
    """goose's own prefix alone says ``runner_error``; the kind names why."""
    provider = GooseProvider()
    assert provider.classify_failure(_with_error_part("other", PREFIXED), "", 1) == (
        "runner_error"
    )
    assert provider.classify_failure(_with_error_part(kind, PREFIXED), "", 1) == (
        category
    )


def test_the_kind_wins_over_wording_that_says_something_else():
    """goose's enum is the verdict; a message quoting a 429 does not move it."""
    message = "Ran into this error: upstream said 429 Too Many Requests."
    stdout = _with_error_part("contextLengthExceeded", message)
    assert GooseProvider().classify_failure(stdout, "", 1) == (
        "context_length_exceeded"
    )


def test_kind_other_and_an_unknown_kind_fall_back_to_the_wording():
    provider = GooseProvider()
    worded = "Ran into this error: This model's maximum context length is 8192 tokens."
    assert provider.classify_failure(_with_error_part("other", worded), "", 1) == (
        "context_length_exceeded"
    )
    # A kind a later goose adds is not guessed at: the tables decide.
    assert provider.classify_failure(_with_error_part("somethingNew"), "", 1) is None


def test_a_kind_written_in_the_answer_text_is_not_read():
    """Only a ``type: "error"`` part carries a kind; a model's prose never does."""
    stdout, _stderr = _capture("success")
    envelope = json.loads(stdout)
    last = [m for m in envelope["messages"] if m.get("role") == "assistant"][-1]
    last["content"] = [{
        "type": "text",
        "text": 'goose writes {"type": "error", "kind": "contextLengthExceeded"} parts.',
    }]
    assert GooseProvider().classify_failure(json.dumps(envelope), "", 0) is None


def test_only_the_last_assistant_messages_kind_counts():
    """An error part in an earlier turn, answered after, decides nothing."""
    envelope = json.loads(_with_error_part("authentication"))
    envelope["messages"].append({
        "role": "assistant",
        "content": [{"type": "text", "text": "[]"}],
    })
    assert GooseProvider().classify_failure(json.dumps(envelope), "", 0) is None
