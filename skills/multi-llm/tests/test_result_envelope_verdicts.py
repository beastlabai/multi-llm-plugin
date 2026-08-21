"""The result envelope's three machine-readable verdicts, graded (C5).

``claude -p --output-format json`` and ``cursor-agent --output-format json``
print three fields that say what the run did, and all three were being thrown
away: ``is_error`` and ``terminal_reason`` (did it fail?) and
``api_error_status`` (why?). §17.3's own diagnosis had to be read out of
``permission_denials`` by hand for exactly that reason — the parser held the
verdict and dropped it.

**The fix splits across two functions, so this module grades two things.**

* ``parse_result_envelope`` owns ``is_error`` / ``terminal_reason``: an envelope
  the CLI flagged is refused rather than having its ``result`` served as the
  answer. Both value sets are *measured*, not chosen — see
  :data:`SUCCESS_TERMINAL_REASONS` and the table in that function's docstring —
  and ``TestTheMeasuredValueTable`` walks every cell of it.
* Each adapter owns ``api_error_status``, as a rung **ahead of** the text
  ladder, delegating to ``super()`` when the field is absent or ``null``.
  ``TestTheTwoRoutesAgree`` is the acceptance test the plan asks for: C2
  reordered the ladder and C3 widened claude-code's auth patterns, and this task
  fixes the same two cells structurally, so the guard against moving a cell
  twice is an assertion that the two routes never answer differently.

Everything here reads the committed captures under ``tests/fixtures/``. Nothing
is synthesized except the four envelopes in ``TestTheMeasuredValueTable`` and
``TestTheStatusRungIsConsultedFirst``, which exist to cover cells no CLI has yet
produced (a non-``completed`` ``terminal_reason`` other than ``api_error``, a
403) and are labelled as such.
"""
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.base import (  # noqa: E402
    CATEGORY_INVALID_MODEL,
    CATEGORY_SERVER_ERROR,
    STATUS_AUTH_FAILED,
    STATUS_RATE_LIMITED,
    LLMProvider,
)
from utils.providers.result_envelope import (  # noqa: E402
    SUCCESS_TERMINAL_REASONS,
    category_for_api_error_status,
    classify_api_error_status,
    envelope_api_error_status,
    envelope_verdict,
    parse_result_envelope,
)
from utils.provider_registry import _PROVIDERS  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def _rows(provider: str):
    """Every graded row of one manifest: ``(name, stdout, stderr, exit_code)``."""
    directory = FIXTURES / provider
    manifest = yaml.safe_load((directory / "manifest.yaml").read_text(encoding="utf-8"))
    out = []
    for name, row in (manifest.get("rows") or {}).items():
        if "not_applicable" in row:
            continue
        stdout = (directory / row["stdout"]).read_text(encoding="utf-8")
        stderr = (
            (directory / row["stderr"]).read_text(encoding="utf-8")
            if row.get("stderr")
            else ""
        )
        out.append((name, stdout, stderr, row["exit_code"]))
    return out


CLAUDE_ROWS = _rows("claude-code")
CURSOR_ROWS = _rows("cursor-agent")

#: The three claude-code rows the CLI itself reports as successful runs. Every
#: other row is a failure row, and the split is asserted below rather than
#: trusted, so a re-capture that changes it cannot quietly shrink the sets the
#: rest of this module iterates.
CLAUDE_ANSWERING_ROWS = frozenset({"success", "ephemeral-only", "multi-assistant"})


def _envelope(stdout: str) -> dict:
    return json.loads(stdout.strip())


def _text_ladder(adapter, stdout: str, stderr: str, exit_code: int):
    """What ``classify_failure`` would answer with the status rung removed.

    ``LLMProvider.classify_failure`` unbound *is* what the adapter's override
    calls through ``super()``, so this is the delegate path itself rather than a
    reimplementation of it.
    """
    return LLMProvider.classify_failure(adapter, stdout, stderr, exit_code)


# --------------------------------------------------------------------------- #
# is_error / terminal_reason — the parser's half
# --------------------------------------------------------------------------- #

class TestTheMeasuredValueTable:
    """Every cell of ``parse_result_envelope``'s value table, one test each."""

    ANSWER = json.dumps([{"id": "S001"}])

    def _envelope(self, **fields):
        body = {"type": "result", "subtype": "success", "result": self.ANSWER}
        body.update(fields)
        return json.dumps(body)

    def test_is_error_true_refuses_even_though_the_result_parses(self):
        """The defect itself: a parseable ``result`` under an error flag."""
        result = parse_result_envelope(self._envelope(is_error=True))

        assert result["success"] is False
        assert result["data"] is None
        assert "is_error: true" in result["error"]

    def test_is_error_false_defers_to_the_next_field(self):
        assert parse_result_envelope(self._envelope(is_error=False)) == {
            "success": True,
            "data": [{"id": "S001"}],
        }

    def test_is_error_absent_defers_to_the_next_field(self):
        assert parse_result_envelope(self._envelope()) == {
            "success": True,
            "data": [{"id": "S001"}],
        }

    def test_terminal_reason_completed_serves_the_result_exactly_as_before(self):
        assert parse_result_envelope(
            self._envelope(is_error=False, terminal_reason="completed")
        ) == {"success": True, "data": [{"id": "S001"}]}

    def test_terminal_reason_null_serves_the_result(self):
        assert parse_result_envelope(
            self._envelope(is_error=False, terminal_reason=None)
        ) == {"success": True, "data": [{"id": "S001"}]}

    def test_any_other_terminal_reason_refuses_and_quotes_the_value(self):
        """SYNTHETIC value: no capture carries a third reason, so the rule is
        stated over one rather than left to be inferred from ``api_error``."""
        result = parse_result_envelope(
            self._envelope(is_error=False, terminal_reason="cancelled_by_user")
        )

        assert result["success"] is False
        assert "terminal_reason: cancelled_by_user" in result["error"]

    def test_the_success_value_set_is_named_rather_than_implied(self):
        assert SUCCESS_TERMINAL_REASONS == frozenset({"completed"})

    def test_both_flags_are_named_when_both_object(self):
        """The ``or`` is belt-and-braces, so a divergence must be *reported*."""
        flags = envelope_verdict({"is_error": True, "terminal_reason": "api_error"})

        assert flags == ["is_error: true", "terminal_reason: api_error"]

    def test_a_refusal_carries_the_raw_envelope_and_the_vendors_words(self):
        stdout = json.dumps(
            {"type": "result", "is_error": True, "result": "API Error: 500 boom"}
        )

        result = parse_result_envelope(stdout)

        assert result["raw"] == stdout
        assert result["error"].endswith("API Error: 500 boom")

    def test_an_error_flagged_envelope_with_no_result_key_is_refused_too(self):
        """The same defect one branch down: it used to serve the bookkeeping."""
        result = parse_result_envelope(
            json.dumps({"type": "result", "is_error": True, "session_id": "s1"})
        )

        assert result["success"] is False
        assert result["data"] is None


class TestClaudeCodeCaptures:
    """The eleven committed claude-code envelopes, read as the CLI wrote them."""

    def test_the_answering_rows_are_the_three_named(self):
        answering = {
            name for name, stdout, _, _ in CLAUDE_ROWS
            if not envelope_verdict(_envelope(stdout))
        }

        assert answering == CLAUDE_ANSWERING_ROWS

    @pytest.mark.parametrize(
        "name,stdout", [(n, s) for n, s, _, _ in CLAUDE_ROWS], ids=[n for n, _, _, _ in CLAUDE_ROWS]
    )
    def test_is_error_and_terminal_reason_agree(self, name, stdout):
        """THE ASSERTION THAT MAKES THE ``or`` SAFE.

        The two fields never disagree on any committed capture — ``completed``
        on exactly the three ``is_error: false`` rows, ``api_error`` on all
        eight ``is_error: true`` rows — so today either field alone would give
        the same verdict. This test is what turns a future capture where they
        diverge into a red row rather than into "whichever field the code
        happened to consult first".
        """
        envelope = _envelope(stdout)
        flagged = envelope["is_error"] is True
        reason = envelope["terminal_reason"]

        assert (reason not in SUCCESS_TERMINAL_REASONS) is flagged, (
            f"claude-code/{name}: is_error={envelope['is_error']!r} but "
            f"terminal_reason={reason!r} — the two verdict fields disagree. "
            f"Decide which one wins IN parse_result_envelope, in writing, "
            f"rather than letting field order decide it."
        )

    @pytest.mark.parametrize(
        "name,stdout", [(n, s) for n, s, _, _ in CLAUDE_ROWS], ids=[n for n, _, _, _ in CLAUDE_ROWS]
    )
    def test_every_flagged_envelope_is_refused_and_no_other_is(self, name, stdout):
        verdict = envelope_verdict(_envelope(stdout))
        result = parse_result_envelope(stdout)

        if name in CLAUDE_ANSWERING_ROWS:
            # `ephemeral-only` is still a failure — its `result` is the empty
            # string — but not one this refusal produces. What matters is that
            # the verdict fields did not fire on a row the CLI called good.
            assert verdict == []
        else:
            assert verdict, f"claude-code/{name} carries no verdict to refuse on"
            assert result["success"] is False, (
                f"claude-code/{name} was served despite its own verdict"
            )
            for flag in verdict:
                assert flag in result["error"], (
                    f"claude-code/{name}: the refusal does not name {flag!r}"
                )

    def test_multi_assistant_is_untouched_by_the_refusal(self):
        """C5 and C6 share this row and do not collide on it.

        The capture reads ``is_error: false`` / ``terminal_reason: completed``,
        so the refusal never fires and the last-wins selection C6 landed stands.
        """
        stdout = next(s for n, s, _, _ in CLAUDE_ROWS if n == "multi-assistant")
        envelope = _envelope(stdout)

        assert envelope["is_error"] is False
        assert envelope["terminal_reason"] == "completed"
        assert parse_result_envelope(stdout) == {
            "success": True,
            "data": [
                {"id": "S001", "title": "Bound the retry ladder", "severity": "high"},
                {"id": "S002", "title": "Strip the vendor token", "severity": "medium"},
            ],
        }


# --------------------------------------------------------------------------- #
# api_error_status — the classifier's half
# --------------------------------------------------------------------------- #

class TestTheStatusMap:
    """The measured map, cell by cell, plus the two cells that defer."""

    @pytest.mark.parametrize("status,expected", [
        (401, STATUS_AUTH_FAILED),
        (403, STATUS_AUTH_FAILED),
        (429, STATUS_RATE_LIMITED),
        (404, CATEGORY_INVALID_MODEL),
        (500, CATEGORY_SERVER_ERROR),
        (502, CATEGORY_SERVER_ERROR),
        (503, CATEGORY_SERVER_ERROR),
    ])
    def test_a_status_that_names_a_category(self, status, expected):
        assert category_for_api_error_status(status) == expected

    @pytest.mark.parametrize("status", [400, None, "", "not-a-status", True, 200, 302])
    def test_a_status_that_defers(self, status):
        """``None`` is *defer to the text ladder*, never "this run was fine".

        400 is the load-bearing one: claude-code's ``context-overflow`` capture
        is a 400 that must stay ``context_length_exceeded``.
        """
        assert category_for_api_error_status(status) is None

    def test_the_field_is_read_off_the_envelope_the_cli_prints(self):
        stdout = json.dumps({"type": "result", "api_error_status": 429})

        assert envelope_api_error_status(stdout) == 429

    @pytest.mark.parametrize("stdout", ["", "  ", "not json", "{oops", "[]"])
    def test_a_capture_that_is_not_an_envelope_defers(self, stdout):
        assert envelope_api_error_status(stdout) is None
        assert classify_api_error_status(stdout) is None


class TestTheStatusRungIsConsultedFirst:
    """Ahead of the text ladder — and standing aside where the map defers."""

    ADAPTER = _PROVIDERS["claude-code"]

    def _envelope(self, status, result):
        return json.dumps({
            "type": "result",
            "is_error": True,
            "terminal_reason": "api_error",
            "api_error_status": status,
            "result": result,
        })

    def test_the_status_wins_over_a_sentence_that_says_otherwise(self):
        """SYNTHETIC: a 404 whose text would classify ``server_error``."""
        stdout = self._envelope(404, "API Error: 404 something went wrong")

        assert _text_ladder(self.ADAPTER, stdout, "", 1) == CATEGORY_SERVER_ERROR
        assert self.ADAPTER.classify_failure(stdout, "", 1) == CATEGORY_INVALID_MODEL

    def test_a_403_is_auth_even_though_no_capture_carries_one(self):
        """SYNTHETIC: 403 shares the 401 cell and nothing measured pins it."""
        stdout = self._envelope(403, "API Error: 403 Forbidden")

        assert self.ADAPTER.classify_failure(stdout, "", 1) == STATUS_AUTH_FAILED

    def test_a_400_defers_so_the_sentence_still_decides(self):
        stdout = self._envelope(
            400, "API Error: 400 This model's maximum context length is 8192 tokens."
        )

        assert self.ADAPTER.classify_failure(stdout, "", 1) == "context_length_exceeded"

    def test_a_run_that_printed_no_envelope_still_reaches_the_ladder(self):
        """The rung cannot mask a launch failure: there is no envelope to read."""
        assert self.ADAPTER.classify_failure("", "claude: command not found", 127) == (
            "spawn_failed"
        )


class TestTheTwoRoutesAgree:
    """C5's acceptance test, and the guard against moving one cell twice.

    C3 fixes claude-code's 401 by widening the auth patterns; C2 reordered the
    ladder; C5 answers both from the envelope's own status. Whichever landed
    second must not move a cell the first already moved — so where **both**
    routes name a category they must name the same one.

    One row of the eleven is deliberately not symmetric, and it is named rather
    than glossed: on ``invalid-model`` the text ladder answers ``None`` at any
    ladder order, because claude-code substitutes its own sentence for the
    vendor's and no ``INVALID_MODEL_MARKERS`` entry matches it. The status rung
    is the only route to ``invalid_model`` there, which is the whole reason C5
    is structural rather than another pattern.
    """

    ADAPTER = _PROVIDERS["claude-code"]

    #: Every claude-code failure row: (status-map answer, text-ladder answer).
    #: Both columns are asserted, so a change to either route is one visible
    #: cell rather than a silently absorbed one.
    EXPECTED = {
        "truncated": (None, CATEGORY_SERVER_ERROR),
        "error-envelope": (None, CATEGORY_SERVER_ERROR),
        "stdout-error": (CATEGORY_SERVER_ERROR, CATEGORY_SERVER_ERROR),
        "auth-error": (STATUS_AUTH_FAILED, STATUS_AUTH_FAILED),
        "rate-limit": (STATUS_RATE_LIMITED, STATUS_RATE_LIMITED),
        "invalid-model": (CATEGORY_INVALID_MODEL, None),
        "context-overflow": (None, "context_length_exceeded"),
        "zero-byte-response": (None, CATEGORY_SERVER_ERROR),
    }

    def test_the_table_covers_every_failure_row(self):
        failures = {n for n, _, _, _ in CLAUDE_ROWS if n not in CLAUDE_ANSWERING_ROWS}

        assert set(self.EXPECTED) == failures

    @pytest.mark.parametrize(
        "name,stdout,stderr,exit_code", CLAUDE_ROWS,
        ids=[n for n, _, _, _ in CLAUDE_ROWS],
    )
    def test_the_two_routes_never_contradict_each_other(
        self, name, stdout, stderr, exit_code
    ):
        status = classify_api_error_status(stdout)
        ladder = _text_ladder(self.ADAPTER, stdout, stderr, exit_code)

        if status is not None and ladder is not None:
            assert status == ladder, (
                f"claude-code/{name}: the api_error_status rung says {status!r} "
                f"and the text ladder says {ladder!r}. Two routes to one cell "
                f"must not answer differently — fix the one that is wrong "
                f"rather than relying on which is consulted first."
            )

    @pytest.mark.parametrize(
        "name", sorted(EXPECTED), ids=sorted(EXPECTED),
    )
    def test_each_failure_rows_two_routes_are_pinned(self, name):
        stdout, stderr, exit_code = next(
            (s, e, c) for n, s, e, c in CLAUDE_ROWS if n == name
        )

        assert (
            classify_api_error_status(stdout),
            _text_ladder(self.ADAPTER, stdout, stderr, exit_code),
        ) == self.EXPECTED[name]

    def test_only_invalid_model_needs_the_status_rung_today(self):
        """The status map is a no-op on ten of eleven rows, and that is its point.

        C2 and C3 already moved the cells the map would have moved, so its value
        is the future wordings it makes irrelevant — not a cell it moves today.
        """
        map_only = {
            name for name, (status, ladder) in self.EXPECTED.items()
            if status is not None and ladder is None
        }

        assert map_only == {"invalid-model"}

    @pytest.mark.parametrize(
        "name,stdout,stderr,exit_code", CLAUDE_ROWS,
        ids=[n for n, _, _, _ in CLAUDE_ROWS],
    )
    def test_the_adapter_answers_the_more_specific_of_the_two(
        self, name, stdout, stderr, exit_code
    ):
        status = classify_api_error_status(stdout)
        ladder = _text_ladder(self.ADAPTER, stdout, stderr, exit_code)

        assert self.ADAPTER.classify_failure(stdout, stderr, exit_code) == (
            status if status is not None else ladder
        )


class TestCursorAgentDelegatesOnEveryCommittedRow:
    """Two rows, both without the field — so both must reach ``super()``.

    cursor-agent's nine-key envelope carries ``is_error`` and neither of the
    other two verdicts. A status rung that swallowed the text ladder would take
    this CLI's only failure signal with it, since ``exit_zero_failure_markers``
    is consulted last and the exit code carries no verdict at all.
    """

    ADAPTER = _PROVIDERS["cursor-agent"]

    @pytest.mark.parametrize(
        "name,stdout,stderr,exit_code", CURSOR_ROWS,
        ids=[n for n, _, _, _ in CURSOR_ROWS],
    )
    def test_neither_capture_carries_a_status_or_a_terminal_reason(
        self, name, stdout, stderr, exit_code
    ):
        envelope = _envelope(stdout)

        assert "api_error_status" not in envelope
        assert "terminal_reason" not in envelope

    @pytest.mark.parametrize(
        "name,stdout,stderr,exit_code", CURSOR_ROWS,
        ids=[n for n, _, _, _ in CURSOR_ROWS],
    )
    def test_the_override_delegates(self, name, stdout, stderr, exit_code):
        assert self.ADAPTER.classify_failure(stdout, stderr, exit_code) == (
            _text_ladder(self.ADAPTER, stdout, stderr, exit_code)
        )

    def test_the_flagged_envelope_is_refused_by_the_parser_and_still_runner_error(self):
        """The division of labour this row exists to pin, on both sides."""
        stdout = next(s for n, s, _, _ in CURSOR_ROWS if n == "exit-zero-failure")

        parsed = parse_result_envelope(stdout)

        assert parsed["success"] is False
        assert parsed["data"] is None
        assert "is_error: true" in parsed["error"]
        assert self.ADAPTER.classify_failure(stdout, "", 0) == "runner_error"


class TestNeitherAdapterReadsItsOwnHappyPathAsAFailure:
    """The assertion the Done-when singles out, on both adapters."""

    @pytest.mark.parametrize("provider,row", [
        ("claude-code", "success"),
        ("claude-code", "ephemeral-only"),
        ("cursor-agent", "success"),
    ])
    def test_the_answering_rows_are_unchanged(self, provider, row):
        rows = CLAUDE_ROWS if provider == "claude-code" else CURSOR_ROWS
        stdout, stderr, exit_code = next(
            (s, e, c) for n, s, e, c in rows if n == row
        )
        adapter = _PROVIDERS[provider]

        assert envelope_verdict(_envelope(stdout)) == []
        assert classify_api_error_status(stdout) is None
        assert adapter.classify_failure(stdout, stderr, exit_code) is None

    def test_the_two_success_rows_still_return_their_document(self):
        for provider, rows, expected in (
            ("claude-code", CLAUDE_ROWS, [
                {"id": "S001", "title": "Bound the retry ladder", "severity": "high"},
                {"id": "S002", "title": "Strip the vendor token", "severity": "medium"},
            ]),
            ("cursor-agent", CURSOR_ROWS, ["ok"]),
        ):
            stdout = next(s for n, s, _, _ in rows if n == "success")

            assert _PROVIDERS[provider].parse_output(stdout, "") == {
                "success": True,
                "data": expected,
            }

    def test_ephemeral_only_still_fails_on_its_empty_result_not_on_a_verdict(self):
        """It is a failure, and the reason must stay the empty answer.

        ``is_error: false`` / ``terminal_reason: "completed"`` / **exit 0**: the
        only thing separating this capture from a good run is that ``result`` is
        the empty string, and that is still what the parser refuses it for.
        """
        stdout = next(s for n, s, _, _ in CLAUDE_ROWS if n == "ephemeral-only")

        result = _PROVIDERS["claude-code"].parse_output(stdout, "")

        assert result["success"] is False
        assert "No valid JSON found in text" in result["error"]
        assert "is_error" not in result["error"]
