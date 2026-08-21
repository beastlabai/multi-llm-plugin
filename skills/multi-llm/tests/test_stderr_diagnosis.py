"""C4 and C3's gemini row: the stderr that `pi` and `gemini` used to discard.

**What C4 fixed.** Both adapters have taken ``parse_output(self, stdout,
stderr)`` since the base class was written, and both accepted ``stderr`` and
then never mentioned it again — so 100% of the diagnosis was thrown away.
``pi`` writes **zero bytes to stdout on all ten** of its committed failure
captures and ``gemini`` on **eight of its nine**; every one of those failures
reported the same contentless string (``Empty output``, ``JSON parse error:
Expecting value: line 1 column 1 (char 0)``) while the vendor's full error
object sat unread on the other channel. The repair is two-part, because the two
lanes reach different hooks: ``describe_failure`` for the non-zero-exit path
(Lane A on every non-zero exit, Lane B when ``parse_output`` failed and
recovered no text) and ``parse_output``'s own error branch for the exit-0 case
and the §6.2 fixture rig.

**The five-rung stderr precedence, which this module grades rung by rung.**

1. **Channel** — stderr is the diagnosis, stdout is the answer. For seventeen of
   the nineteen failure captures the rung decides nothing (one channel is
   empty); gemini's ``exit-zero-failure`` is the nineteenth and is why it is
   written down at all.
2. **Within pi's stderr** — strip the bare ``NNN: `` status prefix, report the
   vendor JSON's ``message`` with its ``code`` and the stripped status; a
   non-JSON remainder is reported verbatim.
3. **Within gemini's stderr, three tiers** — the last ``ApiError`` blob, then
   the prose after the ``Full report`` path, then the first line verbatim.
4. **Two things are dropped first** — the Node ``    at …`` stack and the
   ``/tmp/mllm-inv/…`` report path, both per-invocation noise — and what is left
   is capped to one short line.
5. **Both channels empty keeps today's string, unembellished** — pi's three
   silent rows, with ``describe_failure`` returning ``None``. Nothing is
   invented from an exit code.

**The rule that binds all of it: ``describe_failure`` is diagnostics, never
classification.** Both callers wrap it in ``try``/``except`` and neither lets it
change a category. :class:`TestNoClassifyCellMoved` is that rule as an
assertion, and it is why gemini's ``exit-zero-failure`` markers (``Error
executing tool``, ``not found in registry``) are surfaced here as *prose* and
deliberately not declared as ``exit_zero_failure_markers``.

**C3's gemini row** is the one classification change in this file's scope, and
it is a pattern-table change rather than a prose one: gemini's measured 401
wording matched none of the ten base auth patterns, so a logged-out gemini read
as transport-shaped and burned the whole bounded-retry budget.
:class:`TestGeminiAuthPatterns` pins the fix and — the half that matters after
C1 — pins that **no** wording C3 adds matches a success capture anywhere in the
corpus, including the four adapters that echo our own prompt back into their
envelopes. That sweep is derived from every C3 adapter's own
``auth_error_patterns`` delta (:data:`C3_ADAPTERS`), not from gemini's two, so
muse's ``api key from`` is graded as directly as gemini's ``unauthenticated``.

C4's and C3's tests live here rather than in ``test_providers.py`` on purpose:
they are one subject read end to end, and the hand-written per-adapter classes
in that module grade parsing.
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.base import LLMProvider, matches_any  # noqa: E402
from utils.providers.claude_code import ClaudeCodeProvider  # noqa: E402
from utils.providers.cline import ClineProvider  # noqa: E402
from utils.providers.gemini import GeminiProvider  # noqa: E402
from utils.providers.muse import MuseProvider  # noqa: E402
from utils.providers.gemini import _MAX_DIAGNOSIS_CHARS as GEMINI_CAP  # noqa: E402
from utils.providers.pi import PiProvider  # noqa: E402
from utils.providers.pi import _MAX_DIAGNOSIS_CHARS as PI_CAP  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"

PI = PiProvider()
GEMINI = GeminiProvider()


def _capture_rows(provider):
    """Every committed capture row for ``provider``, by name."""
    directory = FIXTURES / provider
    return sorted({path.name.split(".")[0] for path in directory.glob("*.txt")})


def _streams(provider, row):
    """``(stdout, stderr)`` for one capture row; a missing file is 0 bytes."""
    out = []
    for key in ("stdout", "stderr"):
        path = FIXTURES / provider / f"{row}.{key}.txt"
        out.append(path.read_text(encoding="utf-8") if path.is_file() else "")
    return out[0], out[1]


def _manifest(provider):
    return yaml.safe_load(
        (FIXTURES / provider / "manifest.yaml").read_text(encoding="utf-8")
    )


# --------------------------------------------------------------------------- #
# The tier tables. Every committed capture is named, with the rung that must
# answer it — so the precedence is GRADED rather than the text alone, and a new
# capture cannot be added without deciding which rung owns it.
# --------------------------------------------------------------------------- #

#: pi's seven stderr-bearing rows split **4 / 3**: vendor JSON behind an
#: ``NNN: `` prefix, then a bare sentence that is already the whole diagnosis.
#: ``None`` is rung 5 — nothing on either channel, or a success.
PI_TIERS = {
    "auth-error": PiProvider.TIER_VENDOR_JSON,
    "invalid-model": PiProvider.TIER_VENDOR_JSON,
    "rate-limit": PiProvider.TIER_VENDOR_JSON,
    "server-error": PiProvider.TIER_VENDOR_JSON,
    "error-envelope": PiProvider.TIER_VERBATIM,
    "truncated": PiProvider.TIER_VERBATIM,
    "zero-body": PiProvider.TIER_VERBATIM,
    "context-overflow": None,
    "empty": None,
    "exit-zero-failure": None,
    "success": None,
}

#: gemini's nine failure rows split **4 / 4 / 1**: the last ``ApiError`` blob,
#: the prose after the ``Full report`` path, the first line verbatim. The single
#: tier-(iii) row is ``exit-zero-failure``, which has no blob, no path and no
#: trailing envelope — and is also the only capture with bytes on both channels.
GEMINI_TIERS = {
    "auth-error": GeminiProvider.TIER_API_ERROR,
    "context-overflow": GeminiProvider.TIER_API_ERROR,
    "rate-limit": GeminiProvider.TIER_API_ERROR,
    "server-error": GeminiProvider.TIER_API_ERROR,
    "error-envelope": GeminiProvider.TIER_POST_PATH_PROSE,
    "invalid-model": GeminiProvider.TIER_POST_PATH_PROSE,
    "truncated": GeminiProvider.TIER_POST_PATH_PROSE,
    "zero-byte-response": GeminiProvider.TIER_POST_PATH_PROSE,
    "exit-zero-failure": GeminiProvider.TIER_FIRST_LINE,
    "ephemeral-only": None,
    "multi-assistant": None,
    "success": None,
}

#: The vendor's own words, per row. These are what make pi's nine failure rows
#: and gemini's eight distinguishable from each other at all — before C4 each
#: set collapsed to one contentless string.
PI_WORDS = {
    "auth-error": ("401: Incorrect API key provided", "(invalid_api_key)"),
    "invalid-model": ("404: The model `stub-model` does not exist", "(model_not_found)"),
    "rate-limit": ("429: Rate limit reached for stub-model", "(rate_limit_exceeded)"),
    # The vendor JSON's `code` is null on a 500, so the status and message
    # stand alone — the tier reports what is there, never a placeholder.
    "server-error": ("500: The server had an error while processing your request",),
    "error-envelope": ("Stream ended without finish_reason",),
    "truncated": ("Stream ended without finish_reason",),
    "zero-body": ("Connection error.",),
}

GEMINI_WORDS = {
    "auth-error": ("401: Incorrect API key provided",),
    "context-overflow": ("400: This model's maximum context length is 8192 tokens",),
    "rate-limit": ("429: Rate limit reached for stub-model",),
    "server-error": ("500: The server had an error while processing your request",),
    "error-envelope": ("Error: Incomplete JSON segment at the end", "(code 1)"),
    "truncated": ("Error: Incomplete JSON segment at the end", "(code 1)"),
    "invalid-model": ("ModelNotFoundError: The model `gemini-2.5-flash` does not exist",),
    "zero-byte-response": ("Error: exception TypeError: fetch failed sending request",),
    "exit-zero-failure": ('Tool "write_file" not found in registry',),
}

#: Rung 5, and the three rows that make it a rule rather than an accident:
#: ``context-overflow`` is C11's silent 400 and ``empty`` and
#: ``exit-zero-failure`` are byte-identical to it. All three keep ``Empty
#: output`` exactly, with ``describe_failure`` returning ``None``.
PI_SILENT_ROWS = ("context-overflow", "empty", "exit-zero-failure")

#: The C4 acceptance snapshot: every graded row's ``classify_failure``, as it
#: stands after C3's gemini row and before nothing else. Written as a literal so
#: a future edit to either adapter's *prose* that moved a *category* fails here
#: by name, which is the one thing C4 was not allowed to do.
PI_CLASSIFY = {
    "success": None,
    "truncated": None,
    "error-envelope": None,
    "exit-zero-failure": None,
    "empty": None,
    "auth-error": "auth_failed",
    "rate-limit": "rate_limited",
    "invalid-model": "invalid_model",
    "context-overflow": None,
    "zero-body": "connection_error",
    "server-error": None,
}

GEMINI_CLASSIFY = {
    "success": None,
    "truncated": None,
    "ephemeral-only": None,
    "multi-assistant": None,
    "error-envelope": None,
    # NOT moved by C4, and deliberately: the measured markers `Error executing
    # tool` / `not found in registry` sit on stderr and would make a plausible
    # `exit_zero_failure_markers` declaration. Declaring them moves this cell,
    # so it is out of C4's scope by C4's own rule.
    "exit-zero-failure": None,
    # C3, and the ONE cell in either manifest that moved. It read None.
    "auth-error": "auth_failed",
    "rate-limit": "rate_limited",
    "invalid-model": "invalid_model",
    "context-overflow": "context_length_exceeded",
    "server-error": None,
    "zero-byte-response": None,
}


# --------------------------------------------------------------------------- #
# Rungs 2 and 3 — which tier answered
# --------------------------------------------------------------------------- #

class TestEveryCaptureLandsOnItsDeclaredTier:
    """The precedence, graded per capture rather than asserted in prose."""

    @pytest.mark.parametrize("row", sorted(PI_TIERS))
    def test_pi(self, row):
        stdout, stderr = _streams("pi", row)
        diagnosis = PI._diagnose(stdout, stderr)
        tier = None if diagnosis is None else diagnosis[0]

        assert tier == PI_TIERS[row]

    @pytest.mark.parametrize("row", sorted(GEMINI_TIERS))
    def test_gemini(self, row):
        stdout, stderr = _streams("gemini", row)
        diagnosis = GEMINI._diagnose(stdout, stderr)
        tier = None if diagnosis is None else diagnosis[0]

        assert tier == GEMINI_TIERS[row]

    @pytest.mark.parametrize("provider,table", [("pi", PI_TIERS), ("gemini", GEMINI_TIERS)])
    def test_the_table_names_every_committed_capture(self, provider, table):
        """A new capture forces a tier decision instead of going ungraded."""
        assert sorted(table) == _capture_rows(provider), (
            f"{provider}'s tier table and its committed captures disagree — a "
            f"capture must be given the rung that answers it"
        )

    def test_pi_splits_four_three(self):
        """The plan's prediction, measured: vendor JSON / bare sentence."""
        tiers = [tier for tier in PI_TIERS.values() if tier is not None]

        assert tiers.count(PiProvider.TIER_VENDOR_JSON) == 4
        assert tiers.count(PiProvider.TIER_VERBATIM) == 3
        assert len(tiers) == 7, "pi has seven stderr-bearing captures"

    def test_gemini_splits_four_four_one(self):
        """The plan's prediction, measured: blob / post-path prose / first line."""
        tiers = [tier for tier in GEMINI_TIERS.values() if tier is not None]

        assert tiers.count(GeminiProvider.TIER_API_ERROR) == 4
        assert tiers.count(GeminiProvider.TIER_POST_PATH_PROSE) == 4
        assert tiers.count(GeminiProvider.TIER_FIRST_LINE) == 1
        assert len(tiers) == 9, "gemini has nine failure captures, all with stderr"

    def test_gemini_collapses_ten_identical_retry_blobs_to_one_line(self):
        """Why tier (i) says *the LAST* blob rather than *a* blob.

        ``rate-limit.stderr.txt`` is 15,013 bytes carrying nine identical
        ``ApiError`` blobs, one per backoff attempt; ``server-error.stderr.txt``
        is 14,191 with ten. Both must come out as one sentence.
        """
        for row in ("rate-limit", "server-error"):
            _, stderr = _streams("gemini", row)
            reason = GEMINI.describe_failure("", stderr)

            assert len(stderr) > 14000
            assert stderr.count("ApiError:") >= 9
            assert reason is not None and len(reason) <= GEMINI_CAP


# --------------------------------------------------------------------------- #
# The message itself
# --------------------------------------------------------------------------- #

class TestTheMessageCarriesTheVendorsWords:

    @pytest.mark.parametrize("row", sorted(PI_WORDS))
    def test_pi(self, row):
        stdout, stderr = _streams("pi", row)
        reason = PI.describe_failure(stdout, stderr)

        assert reason is not None
        for needle in PI_WORDS[row]:
            assert needle in reason

    @pytest.mark.parametrize("row", sorted(GEMINI_WORDS))
    def test_gemini(self, row):
        stdout, stderr = _streams("gemini", row)
        reason = GEMINI.describe_failure(stdout, stderr)

        assert reason is not None
        for needle in GEMINI_WORDS[row]:
            assert needle in reason

    def test_every_pi_failure_row_gets_a_distinct_message(self):
        """The point of C4 for pi: nine rows that used to be one string.

        The two rows that stay identical are ``truncated`` and
        ``error-envelope``, whose stderr really is byte-identical — pi never
        writes the 200-with-error-object's text anywhere, which is that row's
        own recorded defect and not something a parser can fix.
        """
        messages = {
            row: PI.describe_failure(*_streams("pi", row))
            for row in PI_TIERS
            if PI_TIERS[row] is not None
        }

        assert len(set(messages.values())) == 6
        assert messages["truncated"] == messages["error-envelope"]

    def test_every_gemini_failure_row_gets_a_distinct_message(self):
        """Same for gemini, with the same one honest collision."""
        messages = {
            row: GEMINI.describe_failure(*_streams("gemini", row))
            for row in GEMINI_TIERS
            if GEMINI_TIERS[row] is not None
        }

        assert len(set(messages.values())) == 8
        assert messages["truncated"] == messages["error-envelope"]

    @pytest.mark.parametrize("row", sorted(_capture_rows("gemini")))
    def test_gemini_never_reports_object_object(self, row):
        """``error.message`` is the literal string ``[object Object]`` at 0.25.2.

        It is the only structured message field gemini offers and upstream
        stringification destroyed it, so the envelope contributes its ``code``
        and nothing else. Reporting it would be worse than reporting nothing:
        it looks like a diagnosis and is not one.
        """
        stdout, stderr = _streams("gemini", row)
        reason = GEMINI.describe_failure(stdout, stderr)

        assert "[object Object]" not in (reason or "")
        assert "object object" not in (reason or "").lower()


class TestRungFourDropsTheNoiseAndCaps:
    """A 15 KB stderr may not become the error string, and nor may a temp path."""

    @pytest.mark.parametrize(
        "provider,adapter,cap",
        [("pi", PI, PI_CAP), ("gemini", GEMINI, GEMINI_CAP)],
    )
    def test_no_message_is_long_multiline_or_noisy(self, provider, adapter, cap):
        for row in _capture_rows(provider):
            stdout, stderr = _streams(provider, row)
            reason = adapter.describe_failure(stdout, stderr)
            if reason is None:
                continue

            assert "\n" not in reason, f"{provider}/{row}: not one line"
            assert len(reason) <= cap, f"{provider}/{row}: over the cap"
            assert "    at " not in reason, f"{provider}/{row}: a Node stack frame"
            assert "/tmp/mllm-inv/" not in reason, f"{provider}/{row}: a temp path"
            assert "node_modules" not in reason, f"{provider}/{row}: a vendor path"

    @pytest.mark.parametrize(
        "provider,adapter",
        [("pi", PI), ("gemini", GEMINI)],
    )
    def test_the_same_guard_holds_for_parse_outputs_error(self, provider, adapter):
        """The message reaches a user through ``parse_output`` too (Lane B's
        rung 5 and the §6.2 rig), so the cap has to survive that path as well.
        """
        for row in _capture_rows(provider):
            stdout, stderr = _streams(provider, row)
            error = str(adapter.parse_output(stdout, stderr).get("error") or "")

            assert "    at " not in error
            assert "/tmp/mllm-inv/" not in error
            assert "\n" not in error

    def test_a_15kb_stderr_is_capped_rather_than_copied(self):
        """The caller keeps the full text; this hook is the summary."""
        _, stderr = _streams("gemini", "rate-limit")
        reason = GEMINI.describe_failure("", stderr)

        assert len(stderr) > 100 * len(reason)


# --------------------------------------------------------------------------- #
# Rung 1 — the channel, in the one place it decides anything
# --------------------------------------------------------------------------- #

class TestRungOneChannel:
    """stdout is the answer and stays on ``raw``; stderr is the diagnosis."""

    def test_geminis_exit_zero_failure_splits_the_two_channels(self):
        """The single capture with bytes on BOTH channels.

        1,174 bytes of stdout whose ``response`` is the model repeating the
        prompt's own confirmation sentence *"JSON written to …"* for a file that
        does not exist, and 193 bytes of stderr carrying the real reason. Rung 1
        is what stops the lie being reported as the diagnosis.
        """
        stdout, stderr = _streams("gemini", "exit-zero-failure")
        assert len(stdout) > 1000 and len(stderr) > 100

        result = GEMINI.parse_output(stdout, stderr)

        assert result["success"] is False
        # The answer channel: still on `raw`, untouched.
        assert "JSON written to" in result["raw"]
        # The diagnosis channel: reaches `error`, and only `error`.
        assert 'Tool "write_file" not found in registry' in result["error"]
        assert "JSON written to" not in result["error"]
        assert result["data"] is None

    @pytest.mark.parametrize(
        "provider,adapter",
        [("pi", PI), ("gemini", GEMINI)],
    )
    def test_stdout_never_becomes_the_reason(self, provider, adapter):
        """``describe_failure`` reads stderr and nothing else, on every capture."""
        for row in _capture_rows(provider):
            stdout, stderr = _streams(provider, row)

            assert adapter.describe_failure(stdout, stderr) == \
                adapter.describe_failure("", stderr)
            assert adapter.describe_failure(stdout, "") is None

    @pytest.mark.parametrize(
        "provider,adapter",
        [("pi", PI), ("gemini", GEMINI)],
    )
    def test_a_successful_run_is_byte_for_byte_what_it_was(self, provider, adapter):
        """Only the ``error`` string grows. A success has none, so nothing moves.

        stderr is empty on every success capture in both sets, but the assertion
        is made with a loud stderr anyway: a parser that folded the diagnosis
        into a *successful* result would be a far worse bug than the one C4
        fixes.
        """
        for row in _capture_rows(provider):
            stdout, stderr = _streams(provider, row)
            if not adapter.parse_output(stdout, stderr).get("success"):
                continue

            assert adapter.parse_output(stdout, "ApiError: {\"error\":{\"code\":401," \
                "\"message\":\"boom\"}}") == adapter.parse_output(stdout, "")


# --------------------------------------------------------------------------- #
# Rung 5 — both channels empty
# --------------------------------------------------------------------------- #

class TestRungFiveKeepsTodaysString:

    @pytest.mark.parametrize("row", PI_SILENT_ROWS)
    def test_pis_three_silent_rows_are_unembellished(self, row):
        stdout, stderr = _streams("pi", row)
        assert (stdout, stderr) == ("", ""), "this row is 0 bytes on both channels"

        assert PI.describe_failure(stdout, stderr) is None
        assert PI.parse_output(stdout, stderr)["error"] == "Empty output"

    def test_the_three_are_byte_identical_to_each_other(self):
        """``context-overflow`` is C11's silent 400; the other two are its twins.

        The whole reason C4 cannot improve them: at pi 0.80.6 a swallowed
        overflow, a legitimate empty answer and an exit-0 failure are the same
        event, and inventing a reason from the exit code is what rung 5 forbids.
        """
        captures = {row: _streams("pi", row) for row in PI_SILENT_ROWS}

        assert len(set(captures.values())) == 1

    @pytest.mark.parametrize("row", PI_SILENT_ROWS)
    def test_the_manifest_records_that_as_the_measurement(self, row):
        """"Not a gap" has to be written where the next reader will look."""
        notes = _manifest("pi")["rows"][row]["notes"]

        assert "rung 5" in notes.lower()
        assert "measurement" in notes.lower()

    def test_gemini_has_one_such_row_and_it_behaves_the_same(self):
        """``ephemeral-only`` is gemini's only 0-byte-stderr failure capture."""
        stdout, stderr = _streams("gemini", "ephemeral-only")
        assert stderr == ""

        assert GEMINI.describe_failure(stdout, stderr) is None
        assert GEMINI.parse_output(stdout, stderr)["error"] == "No valid JSON found in text"


# --------------------------------------------------------------------------- #
# parse_output names the stderr, and the manifests grade it
# --------------------------------------------------------------------------- #

class TestParseOutputNamesTheStderr:
    """The half that covers the exit-0 case and the §6.2 fixture rig."""

    @pytest.mark.parametrize(
        "provider,adapter",
        [("pi", PI), ("gemini", GEMINI)],
    )
    def test_every_stderr_bearing_failure_row_carries_its_reason(self, provider, adapter):
        for row in _capture_rows(provider):
            stdout, stderr = _streams(provider, row)
            reason = adapter.describe_failure(stdout, stderr)
            result = adapter.parse_output(stdout, stderr)
            if reason is None or result.get("success"):
                continue

            assert reason in result["error"], (
                f"{provider}/{row}: parse_output dropped the stderr diagnosis"
            )

    @pytest.mark.parametrize(
        "provider,adapter",
        [("pi", PI), ("gemini", GEMINI)],
    )
    def test_todays_string_is_kept_as_the_prefix(self, provider, adapter):
        """The stderr is *added*, never *substituted*.

        Every existing ``error_contains`` in both manifests keeps holding, and a
        caller matching on the old wording is not broken by the improvement.
        """
        for row in _capture_rows(provider):
            stdout, stderr = _streams(provider, row)
            result = adapter.parse_output(stdout, stderr)
            if result.get("success"):
                continue

            bare = adapter._parse_stdout(stdout)

            assert result["error"].startswith(str(bare["error"]))
            assert result["raw"] == bare["raw"]
            assert result["data"] == bare["data"]

    @pytest.mark.parametrize("provider", ["pi", "gemini"])
    def test_the_manifest_names_the_vendors_words_on_every_such_row(self, provider):
        """The improvement is graded by the rig, not asserted here.

        Every failure row with non-empty stderr must declare an
        ``error_contains`` beyond the generic parser string — otherwise the row
        would pass just as happily with the diagnosis thrown away again.
        """
        generic = {"Empty output", "JSON parse error", "No valid JSON found in text"}
        rows = _manifest(provider)["rows"]

        for name, row in rows.items():
            if "not_applicable" in row:
                continue
            _, stderr = _streams(provider, name)
            expect = row["expect"]
            if expect["success"] or not stderr.strip():
                continue

            specific = [n for n in (expect.get("error_contains") or ()) if n not in generic]

            assert specific, (
                f"{provider}/{name} has {len(stderr)} bytes of stderr and no "
                f"error_contains naming the vendor's own words"
            )

    def test_geminis_exit_zero_failure_row_pins_both_channels(self):
        """The Done-when cell: stderr in ``error_contains``, stdout in ``raw``."""
        row = _manifest("gemini")["rows"]["exit-zero-failure"]

        assert 'Tool "write_file" not found in registry' in row["expect"]["error_contains"]
        assert any("JSON written to" in n for n in row["expect"]["raw_contains"])


# --------------------------------------------------------------------------- #
# The rule C4 is bound by
# --------------------------------------------------------------------------- #

class TestNoClassifyCellMoved:
    """``describe_failure`` is diagnostics, never classification."""

    @pytest.mark.parametrize(
        "provider,adapter,expected",
        [("pi", PI, PI_CLASSIFY), ("gemini", GEMINI, GEMINI_CLASSIFY)],
    )
    def test_the_declared_cells_are_the_expected_ones(self, provider, adapter, expected):
        rows = _manifest(provider)["rows"]
        declared = {
            name: row["expect"].get("classify_failure")
            for name, row in rows.items()
            if "not_applicable" not in row
        }

        assert declared == expected

    @pytest.mark.parametrize(
        "provider,adapter,expected",
        [("pi", PI, PI_CLASSIFY), ("gemini", GEMINI, GEMINI_CLASSIFY)],
    )
    def test_the_adapter_agrees_with_them(self, provider, adapter, expected):
        rows = _manifest(provider)["rows"]

        for name, verdict in expected.items():
            stdout, stderr = _streams(provider, name)
            exit_code = rows[name]["exit_code"]

            assert adapter.classify_failure(stdout, stderr, exit_code) == verdict

    @pytest.mark.parametrize(
        "provider,adapter",
        [("pi", PI), ("gemini", GEMINI)],
    )
    def test_classification_does_not_consult_the_diagnosis_hook(
        self, provider, adapter, monkeypatch
    ):
        """Structural, not incidental: break the hook, keep every verdict.

        ``classify_failure`` reads the raw streams through the declared pattern
        tables. If it ever grew a dependency on the prose built here, a change
        to a *message* could silently move a *category* — which is the one thing
        C4 may not do.
        """
        rows = _manifest(provider)["rows"]
        before = {
            name: adapter.classify_failure(*_streams(provider, name), row["exit_code"])
            for name, row in rows.items()
            if "not_applicable" not in row
        }

        def boom(*_args, **_kwargs):
            raise AssertionError("classify_failure must not call describe_failure")

        monkeypatch.setattr(type(adapter), "describe_failure", boom)
        monkeypatch.setattr(type(adapter), "parse_output", boom)

        after = {
            name: adapter.classify_failure(*_streams(provider, name), row["exit_code"])
            for name, row in rows.items()
            if "not_applicable" not in row
        }

        assert after == before

    def test_gemini_still_declares_no_exit_zero_failure_markers(self):
        """Out of C4's scope, by C4's own rule.

        ``Error executing tool`` and ``not found in registry`` sit on stderr and
        would make a plausible declaration — and declaring them would move
        ``exit-zero-failure``'s ``classify_failure`` cell from ``null`` to
        ``runner_error``. C4 makes the sentence visible; classifying on it is a
        separate change with its own row.
        """
        assert GeminiProvider.exit_zero_failure_markers == ()

        stdout, stderr = _streams("gemini", "exit-zero-failure")

        assert 'not found in registry' in GEMINI.describe_failure(stdout, stderr)
        assert GEMINI.classify_failure(stdout, stderr, 0) is None


# --------------------------------------------------------------------------- #
# C3 — gemini's row
# --------------------------------------------------------------------------- #

#: The four adapters C3 widens, by the provider id their fixture dir carries.
#: The repo-wide sweep below derives its patterns from these classes rather than
#: spelling gemini's two out, so muse's ``api key from``, cline's ``user not
#: found`` and claude-code's ``failed to authenticate`` are swept by the same
#: assertion — and so a *fifth* wording added to any of them is swept on the day
#: it lands rather than the day someone remembers to retype it here.
C3_ADAPTERS = {
    "claude-code": ClaudeCodeProvider,
    "cline": ClineProvider,
    "gemini": GeminiProvider,
    "muse": MuseProvider,
}


def _added_auth_patterns(adapter):
    """The auth wordings ``adapter`` declares on top of the base ten."""
    return tuple(
        p for p in adapter.auth_error_patterns
        if p not in LLMProvider.auth_error_patterns
    )


class TestGeminiAuthPatterns:
    """C3: gemini's measured 401 wording, classified as the credential it is."""

    def test_the_measured_401_now_classifies_as_auth(self):
        """Driven by the existing fixture row, not by new bytes."""
        stdout, stderr = _streams("gemini", "auth-error")

        assert GEMINI.classify_failure(stdout, stderr, 145) == "auth_failed"

    def test_the_base_set_is_widened_rather_than_replaced(self):
        """Two patterns, one per measured wording, on top of the ten base ones."""
        added = _added_auth_patterns(GeminiProvider)

        assert GeminiProvider.auth_error_patterns[:len(LLMProvider.auth_error_patterns)] \
            == LLMProvider.auth_error_patterns
        assert added == ("unauthenticated", "incorrect api key")

    @pytest.mark.parametrize("pattern", ["unauthenticated", "incorrect api key"])
    def test_each_added_pattern_is_needed_and_sufficient_on_its_own(self, pattern):
        """Neither is dead weight, and neither is the base set in disguise.

        The 401 carries BOTH wordings — Google's status word and the upstream
        vendor's sentence — so each is checked alone against the capture.
        """
        _, stderr = _streams("gemini", "auth-error")
        haystack = f"\n{stderr}".lower()

        assert matches_any((pattern,), haystack)
        assert not matches_any(LLMProvider.auth_error_patterns, haystack)

    def test_no_success_capture_anywhere_matches_the_new_patterns(self):
        """C1's hazard, checked the way C3 asks: against the ECHOING adapters.

        ``goose``, ``copilot``, ``vibe`` and ``muse`` echo our own prompt back
        inside their JSON envelopes, so a widened auth table is measured against
        their *success* captures as well as their 401s — an echoed
        "authentication required" would otherwise match more wordings on the one
        class that fails a whole batch. The sweep is over every committed
        capture in the repo rather than a roster of four, so a fifth echoing
        adapter is covered on the day its fixtures land.

        Both axes are derived, not typed: the patterns come from each C3
        adapter's own :attr:`auth_error_patterns` delta, so muse's ``api key
        from`` is swept as literally as gemini's ``unauthenticated``. Spelling
        gemini's two out here would have left the other three guarded only
        indirectly, through manifest cells graded after ``vendor_text`` — which
        is the same check one hop removed, and one hop is where a future
        echoing capture gets through.
        """
        added = {
            provider: _added_auth_patterns(adapter)
            for provider, adapter in sorted(C3_ADAPTERS.items())
        }

        assert all(added.values()), (
            f"a C3 adapter declares no added auth wording to sweep: {added}"
        )

        offenders = []

        for path in sorted(FIXTURES.glob("*/*")):
            if path.suffix not in (".txt", ".jsonl", ".json") or path.name == "manifest.yaml":
                continue
            if path.name.startswith("auth-error"):
                continue  # the one place these words are supposed to appear
            haystack = "\n" + path.read_text(encoding="utf-8", errors="replace").lower()
            for provider, patterns in added.items():
                for pattern in patterns:
                    if matches_any((pattern,), haystack):
                        offenders.append(
                            f"{path.relative_to(FIXTURES)} ~ {provider}:{pattern!r}"
                        )

        assert offenders == [], (
            f"C3's added auth patterns match non-auth captures: {offenders}"
        )

    @pytest.mark.parametrize(
        "row", ["success", "multi-assistant", "ephemeral-only"]
    )
    def test_geminis_own_non_failure_rows_are_unmoved(self, row):
        stdout, stderr = _streams("gemini", row)
        exit_code = _manifest("gemini")["rows"][row]["exit_code"]

        assert GEMINI.classify_failure(stdout, stderr, exit_code) is None

    def test_the_auth_probe_haystack_is_untouched(self):
        """``classify_auth_probe`` shares the pattern set, and should.

        An auth probe never carries our prompt, and a widened set makes the
        credential verdict *more* correct there — the one verdict allowed to
        refuse a run. Pinned so the sharing is deliberate rather than noticed
        later.
        """
        _, stderr = _streams("gemini", "auth-error")

        assert GEMINI.classify_auth_probe("", stderr, 145) is False


# --------------------------------------------------------------------------- #
# C12 — gemini's row
# --------------------------------------------------------------------------- #

class TestGeminiInternalRetryEnvelope:
    """C12: the chain gemini's own docstring measured and stopped short of."""

    def test_the_numbers_are_the_measured_ones(self):
        envelope = GeminiProvider.internal_retries

        assert envelope is not None
        assert envelope.max_requests == 10
        assert envelope.max_wall_s == 243.0
        assert envelope.retries_rate_limit is True
        assert envelope.configurable_by == ()

    def test_the_capture_that_produced_them_is_committed(self):
        """10 requests / 243 s is ``rate-limit``, and the manifest says so."""
        row = _manifest("gemini")["rows"]["rate-limit"]

        assert row["requests"] == GeminiProvider.internal_retries.max_requests
        assert "10 requests over 243 s" in row["notes"]

    def test_declaring_it_removes_the_harnesss_own_rate_limit_backoff(self):
        """The budget consequence, written where a budget reader will find it.

        ``retries_rate_limit: True`` means rule 6(b) gives gemini **no**
        identical rate-limit re-dispatch: a smaller retry budget, not a larger
        one. A slot that used to end after several dispatches — each re-running
        a four-minute chain the CLI had already run — now ends after one.
        """
        from utils.retry_ladder import STOP_VENDOR_ALREADY_RETRIED

        source = Path(__file__).parent.parent / "utils" / "providers" / "gemini.py"
        text = source.read_text(encoding="utf-8")

        assert "REMOVES a retry" in text
        assert "6(b)" in text
        assert STOP_VENDOR_ALREADY_RETRIED
