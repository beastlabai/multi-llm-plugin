"""What the §6.2 fixture rig cannot grade about omp's stream reading.

``tests/fixtures/omp/manifest.yaml`` grades four cells per row — ``success``,
``error_contains``, ``raw_contains``, ``classify_failure`` — because that is the
whole of the rig's ``expect`` schema. omp's parser makes three further claims
over the same bytes, and each is load-bearing:

* the **terminal ``error_code``** it sets itself. No shipped adapter set one in
  ``parse_output`` before omp, and the guard at the assignment site
  (``if not success and "error_code" not in parsed``) exists for exactly this:
  without a code from the hard tuple, ``invoke_with_file_output`` reads a failed
  parse as "the command ran, the model may still have written the file" and goes
  on to recover ``{output_json_path}``. Lane A's review prompts write that file
  *before* the model's later tool calls, so a run whose findings landed and whose
  ``git diff`` was then cut by ``--max-time`` — exit 0, ``toolUse``-final — would
  be reported as a success with a cut-off review passed downstream as complete.
* :meth:`~utils.providers.omp.OmpProvider.describe_failure`'s **precedence**.
  It is called once, inside ``invoke_with_provider``'s ``returncode != 0``
  branch, and Lane A reaches ``parse_output`` only on exit 0 — so every
  in-stream failure that exits 1 is recovered there or nowhere. Its declared
  channel is ``STDOUT_ELSE_STDERR``, whose contract is that when stdout answers,
  stderr may not change the text.
* what :meth:`~utils.providers.omp.OmpProvider.vendor_text` **removes**, which
  is what keeps a marker quoted by the plan under review from outranking the
  run's own verdict.

Every stream below is read off ``tests/fixtures/omp/``, so the parser is graded
against the bytes omp wrote rather than against strings written beside it.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_client import ERROR_SUBPROCESS_FAILED, ERROR_TIMEOUT  # noqa: E402
from utils.providers import omp as omp_module  # noqa: E402
from utils.providers.base import (  # noqa: E402
    CATEGORY_PERMISSION_DENIED,
    CATEGORY_SERVER_ERROR,
    STATUS_RATE_LIMITED,
    STATUS_TIMEOUT,
    classify_streams,
)
from utils.providers.omp import OMP_STREAM_VERSION, OmpProvider  # noqa: E402



class TestOmpStreamReading:
    """What the §6.2 manifest rig cannot grade, over the same committed bytes.

    The fixture rig grades `success`, `error_contains`, `raw_contains` and
    `classify_failure` — the four cells its `expect` schema admits. omp's parser
    makes three claims outside that schema and each is load-bearing:

    * the **terminal code** it sets itself, which is the whole reason
      `invoke_with_file_output` refuses to recover a findings file beside a cut
      or denied run;
    * `describe_failure`'s **precedence**, which is the only voice omp has on
      every failure that exits 1 — and which may never compose its two channels;
    * what `vendor_text` **removes**, which is what keeps a marker quoted by the
      plan under review from outranking the run's own verdict.

    Every stream here is read off `tests/fixtures/omp/`, so these are graded
    against the bytes omp wrote and not against strings written beside the
    parser by its author.
    """

    @pytest.fixture
    def provider(self):
        return OmpProvider()

    @staticmethod
    def _row(name):
        base = Path(__file__).parent / "fixtures" / "omp"
        stdout = (base / f"{name}.stdout.txt")
        stderr = (base / f"{name}.stderr.txt")
        return (
            stdout.read_text(encoding="utf-8") if stdout.exists() else "",
            stderr.read_text(encoding="utf-8") if stderr.exists() else "",
        )

    # ── the terminal codes ───────────────────────────────────────────────
    @pytest.mark.parametrize("row,expected", [
        # `--max-time` landing inside omp is the harness deadline landed inside
        # omp, so both of its shapes are TIMEOUT.
        ("aborted", ERROR_TIMEOUT),
        ("exit-zero-failure", ERROR_TIMEOUT),
        # A vendor verdict in the stream, and the two shape guards: the parser
        # cannot vouch for the terminal state of a message it cannot read, so a
        # file written beside one is not read either.
        ("auth-error", ERROR_SUBPROCESS_FAILED),
        ("rate-limit", ERROR_SUBPROCESS_FAILED),
        ("stdout-error", ERROR_SUBPROCESS_FAILED),
        ("context-overflow", ERROR_SUBPROCESS_FAILED),
        ("approval-denied", ERROR_SUBPROCESS_FAILED),
        ("unrecognized-stop-reason", ERROR_SUBPROCESS_FAILED),
        ("missing-stop-reason", ERROR_SUBPROCESS_FAILED),
    ])
    def test_a_terminal_shape_carries_a_hard_code(self, provider, row, expected):
        """No shipped adapter set `error_code` in `parse_output` before omp.

        The guard at the assignment site (`if not success and "error_code" not
        in parsed`) exists for exactly this, and omp is its first user: without
        it these rows are a bare `PARSE_ERROR`, which `invoke_with_file_output`
        reads as "the command ran, the model may still have written the file".
        """
        assert provider.parse_output(*self._row(row))["error_code"] == expected

    @pytest.mark.parametrize("row", [
        # NOT `empty`, though §1.9 predicted it would be: the measurement made
        # that vehicle a terminal `error` carrying Bun's closed-socket sentence,
        # which is a verdict and carries the hard code above. What is here is
        # the genuine article — the empty-stop retry cap, a killed run, and the
        # three pre-flight refusals.
        "error-envelope", "truncated", "ephemeral-only",
        "invalid-model", "invalid-tool", "auth-error-preflight",
    ])
    def test_a_run_that_produced_nothing_carries_no_code_at_all(self, provider, row):
        """`PARSE_ERROR` is the right verdict when the model produced nothing.

        A file it did not write cannot be recovered, so leaving the key absent
        costs nothing and keeps the hard tuple meaning what it says.
        """
        assert "error_code" not in provider.parse_output(*self._row(row))

    def test_a_successful_run_carries_no_code(self, provider):
        result = provider.parse_output(*self._row("success"))
        assert result["success"] is True
        assert "error_code" not in result

    # ── describe_failure's precedence ────────────────────────────────────
    def test_when_stdout_answers_stderr_may_not_change_the_text(self, provider):
        """`STDOUT_ELSE_STDERR` is precedence, not enrichment.

        The registry-wide channel test grades the declaration over every
        committed capture pair; this pins the clause that makes the declaration
        true — the two channels are never composed.
        """
        stdout, _ = self._row("auth-error")
        alone = provider.describe_failure(stdout, "")
        with_noise = provider.describe_failure(stdout, "Working...\nand something else")
        assert alone and alone == with_noise

    def test_when_stdout_is_silent_the_stderr_diagnosis_is_the_whole_answer(
        self, provider
    ):
        stdout, stderr = self._row("invalid-tool")
        assert stdout.strip() == "", "this row is the 0-B-stdout shape"
        answer = provider.describe_failure(stdout, stderr)
        assert "CliUsageError: Unknown tool in --tools: bogus." in answer
        # The Bun source-context prelude comes BEFORE the diagnosis, so without
        # the strip the 300-char cap spends its whole budget on minified source.
        assert "19530 |" not in answer
        assert "/$bunfs/" not in answer

    def test_a_stream_with_no_verdict_answers_none(self, provider):
        """Diagnostics, never classification: `None` is always safe here.

        A clean `stop` is not a failure, and a run silent on both channels said
        nothing to report — `describe_failure` is allowed to answer `None` on
        either leg of its contract, and inventing a sentence from an exit code
        is what pi's own `_diagnose` refuses to do for the same reason.
        """
        assert provider.describe_failure(*self._row("success")) is None
        assert provider.describe_failure("", "") is None
        assert provider.describe_failure("", "   \n  ") is None

    def test_the_no_assistant_message_sentence_is_the_one_deliberate_divergence(
        self, provider
    ):
        """`parse_output` keeps the fallback; `describe_failure` drops it here.

        That fallback is the one place the two strings differ, and each side is
        forced: `error_contains` wants the stream sentence AND the stderr named
        in the error, while `describe_failure` is the ONLY voice on the exit-1
        path — so reporting the shape of a one-event stream there would bury the
        one sentence a user acts on. A run that never started said nothing on
        stdout (`_never_started`), so `STDOUT_ELSE_STDERR`'s stderr leg answers
        alone rather than being forbidden by an answer stdout never gave.
        """
        stdout, stderr = self._row("auth-error-preflight")
        assert stderr.strip(), "this row's whole point is a non-empty stderr"
        error = provider.parse_output(stdout, stderr)["error"]
        answer = provider.describe_failure(stdout, stderr)

        assert "run ended before any assistant message" in error
        assert "; stderr:" in error

        assert answer == omp_module.stderr_diagnosis(stderr)
        assert "No API key found for openrouter." in answer
        assert "run ended before any assistant message" not in answer
        # Precedence, not composition: stdout alone recovers nothing at all on
        # this row, which is what lets stderr speak without touching an answer.
        assert provider.describe_failure(stdout, "") is None

    def test_a_run_that_started_and_was_cut_keeps_the_stream_sentence(
        self, provider
    ):
        """`_never_started`'s boundary is `agent_start`, not "no assistant message".

        `ephemeral-only` is the SIGKILLed run: `session`, `agent_start`,
        `turn_start` and the user message, then nothing — and 0 bytes of stderr.
        The stream is all there is to say about it, so the fallback sentence is
        an answer there and must survive the pre-flight carve-out above.
        """
        stdout, stderr = self._row("ephemeral-only")
        assert not stderr.strip(), "this row's whole point is a silent stderr"
        answer = provider.describe_failure(stdout, stderr)
        assert answer and "run ended before any assistant message" in answer

    def test_both_hooks_read_one_stream_reader(self, provider, monkeypatch):
        """One reader, one sentence, no drift between the report and the log."""
        monkeypatch.setattr(
            omp_module, "terminal_verdict", lambda stdout, stderr: "SENTINEL-VERDICT"
        )
        stdout, stderr = self._row("auth-error")
        assert provider.describe_failure(stdout, stderr) == "SENTINEL-VERDICT"
        assert "SENTINEL-VERDICT" in provider.parse_output(stdout, stderr)["error"]

    @pytest.mark.parametrize("row", ["truncated", "error-envelope"])
    def test_the_empty_stop_sentence_comes_from_the_shared_reader(
        self, provider, monkeypatch, row
    ):
        """The empty-stop branch reads the verdict, it does not re-derive it.

        This is the branch Lane A actually takes on these two rows — exit 0, so
        `parse_output` is the production parser and `describe_failure` is never
        called — and the sentence it reports is omp's own `auto_retry_end`
        wording. A private `_retry_give_up` call underneath the wrapping would
        keep working while `terminal_verdict` moved out from under it, which is
        exactly the drift between the report and the log that one reader exists
        to make impossible.
        """
        stdout, stderr = self._row(row)
        assert provider.parse_output(stdout, stderr)["error"] == (
            "Empty output (Assistant returned empty stop after retry cap; try "
            "switching models or `/shake images` to remove archived frames)"
        )
        monkeypatch.setattr(
            omp_module, "terminal_verdict", lambda stdout, stderr: "SENTINEL-VERDICT"
        )
        assert provider.parse_output(stdout, stderr)["error"] == (
            "Empty output (SENTINEL-VERDICT)"
        )

    # ── the non-JSON lines beside the stream ─────────────────────────────
    @pytest.mark.parametrize("row", [
        "auth-error", "aborted", "exit-zero-failure",
        "approval-denied", "unrecognized-stop-reason", "missing-stop-reason",
    ])
    def test_a_banner_beside_the_stream_is_named_in_the_error(self, provider, row):
        """`_events` keeps what it could not parse; the error has to say so.

        A banner — an update notice, a deprecation warning, a runtime's own
        chatter — arrives as one non-JSON line beside an otherwise well-formed
        stream, and every branch here reports a failure a reader must act on. A
        line nobody has seen yet is visible in the reported error rather than
        dropped, and the verdict itself is unchanged: this is a suffix, never a
        substitution.
        """
        banner = "npm notice New major version of omp available! 18.2.2 -> 19.0.0"
        stdout, stderr = self._row(row)
        verdict = provider.parse_output(stdout, stderr)["error"]
        noisy = provider.parse_output(f"{banner}\n{stdout}", stderr)["error"]
        assert noisy == f"{verdict}; non-JSON output: {banner}"

    # ── classify_failure's structural read ───────────────────────────────
    def test_a_denial_outranks_the_clean_stop_that_follows_it(self, provider):
        """The fabricated-review guard, at the classifier as well as the parser.

        `approval-denied`'s last assistant message IS a `stop` carrying a
        well-formed findings array. `permission_denied` is mapped onto no ladder
        status by `_LADDER_STATUS_BY_CATEGORY` (only `connection_error` and
        `server_error` are), so it reaches `RetryLadder.record` under its own name
        and takes that method's last branch — never retried — and the slot ends
        after one dispatch, which is right, because no retry fixes a policy
        failure.
        """
        assert provider.classify_failure(*self._row("approval-denied"), 0) == \
            CATEGORY_PERMISSION_DENIED

    def test_the_exit_zero_abort_classifies_without_any_marker(self, provider):
        """`Command aborted` is deliberately NOT a marker.

        A successful run can contain it in a tool result the model recovered
        from, so the verdict is structural — and without the structural branch
        this row would be `None`, rendered `Reason: Unknown`.
        """
        stdout, stderr = self._row("exit-zero-failure")
        assert provider.classify_failure(stdout, stderr, 0) == STATUS_TIMEOUT
        assert provider.exit_zero_failure_markers == ()

    def test_a_tool_error_the_model_recovered_from_is_not_a_verdict(self, provider):
        stdout, stderr = self._row("tool-error-recovered")
        assert provider.parse_output(stdout, stderr)["success"] is True
        assert provider.classify_failure(stdout, stderr, 0) is None

    def test_the_five_hundred_row_needs_the_structural_branch(self, provider):
        """The stub's 500 sentence matches no shared `SERVER_ERROR_MARKERS` row.

        Read off `errorStatus` instead, so the wording of any one vendor's 5xx
        body is immaterial.
        """
        assert provider.classify_failure(*self._row("stdout-error"), 1) == \
            CATEGORY_SERVER_ERROR

    @pytest.mark.parametrize("row,exit_code", [
        ("approval-denied", 0),
        ("exit-zero-failure", 0),
        ("stdout-error", 1),
        # The control: omp's own words already carry a marker on this row, and
        # `_named_for_the_ladder` must not say it a second time.
        ("aborted", 1),
    ])
    def test_the_structural_verdict_survives_lane_as_string_classification(
        self, provider, row, exit_code
    ):
        r"""Lane A classifies the error STRING and never calls `classify_failure`.

        `run_single_model`'s file-output lane reaches for
        `classify_streams(str(result["error"]), "")`; `_format_failure_details`
        and the Lane B executor are the only callers of the structural read. So
        a verdict the sentence does not name arrives at that ladder as `None`,
        and `RetryLadder.record`'s last branch never retries it. `Command
        aborted` is deliberately not a marker, the denial's own text is not one,
        and the 500 body's `(type=server_error)` matches no
        `SERVER_ERROR_MARKERS` row (`\bserver error\b` wants a space) — so all
        three classified `None` while the `aborted` half of the very same
        `--max-time` said "Deadline exceeded" and took rung 2: one deadline,
        two behaviours on one ladder.
        """
        stdout, stderr = self._row(row)
        structural = provider.classify_failure(stdout, stderr, exit_code)
        assert structural is not None, "this row is graded for a structural verdict"
        assert classify_streams(
            omp_module.terminal_verdict(stdout, stderr) or "", ""
        ) == structural
        # Both channels compose that one sentence, so both reach the same rung:
        # `parse_output`'s `error` on the exit-0 rows, and the
        # `describe_failure` clause `invoke_with_provider` appends on exit 1.
        composed = (
            provider.parse_output(stdout, stderr)["error"]
            if exit_code == 0
            else (
                f"omp exited with code {exit_code}: "
                f"{provider.describe_failure(stdout, stderr)}"
            )
        )
        assert classify_streams(composed, "") == structural

    # ── vendor_text's boundary ───────────────────────────────────────────
    def test_a_marker_quoted_by_the_plan_under_review_never_outranks_the_verdict(
        self, provider
    ):
        """The regression the reduction exists for.

        `markers-in-tool-result` is a SUCCESSFUL `read` whose text quotes every
        auth marker in the tables, followed by the run's own 429. Under
        `classify_streams`' fixed order (auth before rate limit) the quoted 401
        would win — `auth_failed`, no retry, the whole batch failed on a marker
        the vendor never emitted.
        """
        stdout, stderr = self._row("markers-in-tool-result")
        assert "401 User not found." in stdout, "the decoy must really be in there"
        assert provider.classify_failure(stdout, stderr, 1) == STATUS_RATE_LIMITED

    def test_the_haystack_is_the_terminal_sentence_and_stderr_unchanged(
        self, provider
    ):
        stdout, stderr = self._row("markers-in-tool-result")
        reduced_out, reduced_err = provider.vendor_text(stdout, stderr)
        assert reduced_out == "429 Provider returned error"
        assert reduced_err == stderr
        assert "401" not in reduced_out

    def test_a_successful_run_reduces_to_nothing_whatever_its_answer_says(
        self, provider
    ):
        """A `stop` is not a failure verdict however the model narrates it."""
        reduced_out, _ = provider.vendor_text(*self._row("success"))
        assert reduced_out == ""

    def test_a_stdout_that_is_not_jsonl_comes_back_unchanged(self, provider):
        """The contract's failed-parse clause: narrow the haystack, never empty it."""
        banner = "omp v18.2.2 starting up\n"
        assert provider.vendor_text(banner, "boom") == (banner, "boom")

    def test_the_diagnostics_keep_what_the_classifier_dropped(self, provider):
        """Narrowing the haystack removes no byte from anything a human reads.

        `vendor_text` is applied inside `classify_failure` and nowhere else, so
        the quoted 401 is still in the raw stream the run log and the failure
        report's stdout tail are cut from.
        """
        stdout, _ = self._row("markers-in-tool-result")
        assert "401 User not found." in provider.parse_output(stdout, "")["raw"]

    # ── usage ────────────────────────────────────────────────────────────
    def test_usage_sums_every_assistant_message_not_only_the_last(self, provider):
        """One assistant message per API call, so a multi-turn run reports the
        whole dispatch rather than its final turn."""
        single = provider.parse_usage(*self._row("success"))
        multi = provider.parse_usage(*self._row("multi-assistant"))
        assert single["input_tokens"] > 0 and multi["input_tokens"] > single["input_tokens"]
        assert multi["tool_uses"] == 2
        assert single["tool_uses"] == 1

    def test_an_absent_usage_block_leaves_the_keys_absent_rather_than_zero(
        self, provider
    ):
        """A zero would average into per-phase totals and read as a free run."""
        stream = '{"type": "session", "version": 3}\n' + json.dumps({
            "type": "message_end",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "[]"}],
                        "stopReason": "stop"},
        }) + "\n"
        assert provider.parse_usage(stream, "") is None

    def test_usage_is_none_when_there_is_no_stream_to_read(self, provider):
        assert provider.parse_usage("", "") is None
        assert provider.parse_usage("not jsonl at all", "") is None

    # ── the answer, and only the answer ──────────────────────────────────
    def test_a_thinking_blocks_decoy_array_is_never_the_answer(self, provider):
        """Only `type == "text"` content parts are concatenated.

        A `thinking` part is the model's reasoning, and `parse_answer_text`'s
        prefer-arrays extractor would happily pick a fenced array out of it.
        """
        result = provider.parse_output(*self._row("thinking-then-text"))
        assert result["success"] is True
        titles = [item.get("title") for item in result["data"]]
        assert not any("DECOY" in (t or "") for t in titles)
        assert "DECOY" not in result.get("raw", "")

    @staticmethod
    def _with_answer(stdout, answer):
        """The `success` row's own stream carrying a different terminal answer.

        One `message_end` line is re-serialized and nothing else moves — the
        same targeted mutation the stream-version test makes, for a shape the
        §6.2 captures do not hold: neither measured truncation reaches a `stop`
        with text in it (a cut before any content is four empty stops,
        `truncated`; a cut after it is a terminal `error`, `truncated-partial`).
        """
        lines = stdout.splitlines()
        target = None
        for index, line in enumerate(lines):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = event.get("message") or {}
            if (
                event.get("type") == "message_end"
                and message.get("role") == "assistant"
                and message.get("stopReason") == "stop"
            ):
                target = index
        assert target is not None, "the success row ends in a terminal `stop`"
        event = json.loads(lines[target])
        event["message"]["content"] = [{"type": "text", "text": answer}]
        lines[target] = json.dumps(event)
        return "\n".join(lines) + "\n"

    def test_a_stop_holding_a_cut_findings_array_is_never_the_answer(self, provider):
        """§6.2's `truncated` row, second shape: a `stop` carrying the cut prefix.

        What the shared extractor does with one is not a failure, which is why
        this needs a guard of its own: `parse_answer_text` falls back from the
        failed `json.loads` to the balanced-substring scan, and a findings array
        cut after one WHOLE record still carries complete brackets inside the
        records that did arrive. Reproduced on the bytes below, the result was
        `success: True` with `data: [10, 15]` — the first finding's `line_range`
        served as the review.
        """
        cut = (
            '[\n  {\n    "title": "Off-by-one in the lap counter",\n'
            '    "file": "src/laps.py",\n    "line_range": [10, 15],\n'
            '    "importance": "high",\n    "type": "bug"\n  },\n'
            '  {\n    "title": "The stream stops he'
        )
        stdout, stderr = self._row("success")
        result = provider.parse_output(self._with_answer(stdout, cut), stderr)
        assert result["success"] is False
        assert result["data"] is None
        assert "Truncated output" in result["error"]
        assert result["raw"] == cut
        # The cut is a statement about the answer channel and not about a
        # findings file the model may have written whole before it, so the file
        # channel is left free to look — exactly as the empty stop leaves it.
        assert "error_code" not in result

    def test_a_complete_answer_that_says_a_word_afterwards_still_parses(
        self, provider
    ):
        """The guard asks about the leading container, never `json.loads`.

        A complete array followed by a closing sentence fails `json.loads` too,
        and it is precisely the shape the shared extractor exists to recover —
        so refusing every text that does not parse whole would cost every honest
        run that says a word after its findings.
        """
        stdout, stderr = self._row("success")
        answer = '[{"title": "One finding", "importance": "low"}]\n\nThat is all.'
        result = provider.parse_output(self._with_answer(stdout, answer), stderr)
        assert result["success"] is True
        assert [item["title"] for item in result["data"]] == ["One finding"]

    def test_a_successful_extraction_still_carries_the_whole_answer_as_raw(
        self, provider
    ):
        """`raw` is the answer on every outcome, success included.

        `parse_answer_text` returns `{"success": True, "data": …}` and nothing
        else, so an `ask` answer that merely *quotes* a JSON example used to
        reach the orchestrator as the quoted fragment with no text beside it.
        `recover_answer_text` refuses a `dict`/`list` `data` for exactly that
        reason, and with no `raw` to fall back on it drops the whole reply.
        """
        stdout, stderr = self._row("success")
        answer = (
            "Here is how the config looks:\n\n"
            '```json\n{"retries": 3}\n```\n\n'
            "Set it once and the rest follows."
        )
        result = provider.parse_output(self._with_answer(stdout, answer), stderr)
        assert result["success"] is True
        assert result["data"] == {"retries": 3}
        assert result["raw"] == answer

    def test_an_ask_answer_quoting_json_survives_recovery(self, provider):
        """The consumer half of the rule, over the real recovery function.

        `recover_answer_text` is the `ask` lane's reader: `data` only when it is
        a non-empty string, else `raw` when it is not an event stream, else the
        per-model log — which for this adapter holds the JSONL stream and is
        rejected by `looks_like_ndjson`. Without `raw` the answer below is lost.
        """
        from ask_orchestrator import recover_answer_text

        stdout, stderr = self._row("success")
        answer = (
            "Two options. The second one needs this block:\n\n"
            '```json\n[{"name": "b"}]\n```\n\nI would take the first.'
        )
        result = provider.parse_output(self._with_answer(stdout, answer), stderr)
        assert recover_answer_text(result) == answer

    def test_the_shape_guards_never_reach_the_answer_extractor(
        self, provider, monkeypatch
    ):
        """Fail closed: an unreadable message's text is not the answer.

        An unobserved enum may be a cut, a refusal or a rename, and the parser
        cannot tell which — so it never asks the extractor, whose prefer-arrays
        behaviour would happily serve a truncated review as a complete one.
        """
        def _boom(text):  # pragma: no cover - the assertion is that it is not called
            raise AssertionError("parse_answer_text was called on an unreadable message")

        monkeypatch.setattr(omp_module, "parse_answer_text", _boom)
        for row in ("unrecognized-stop-reason", "missing-stop-reason"):
            result = provider.parse_output(*self._row(row))
            assert result["success"] is False

    def test_the_drift_errors_quote_the_raw_value_and_the_stream_version(
        self, provider
    ):
        """The fix is a re-measurement that grows the tuple, so the error has to
        name what moved rather than say the stream was garbage."""
        enum_error = provider.parse_output(*self._row("unrecognized-stop-reason"))["error"]
        key_error = provider.parse_output(*self._row("missing-stop-reason"))["error"]
        assert "'maxTokens'" in enum_error and "stop/toolUse/error/aborted" in enum_error
        assert "without stopReason" in key_error
        for message in (enum_error, key_error):
            assert f"version {OMP_STREAM_VERSION}" in message

    def test_a_stream_version_bump_warns_once_and_reads_on(self, provider, capsys):
        """A bump with the fields intact reads fine, and refusing it would fail
        every dispatch on a benign release — so it is a WARN, not a refusal, and
        it is quoted in every shape error so a drift is never garbage."""
        from utils.providers import base as providers_base

        providers_base.reset_flag_gate_cache()
        stdout, stderr = self._row("success")
        bumped = stdout.replace('"version": 3', '"version": 4', 1).replace(
            '"version":3', '"version":4', 1)
        assert bumped != stdout, "the substitution has to have taken"

        assert provider.parse_output(bumped, stderr)["success"] is True
        first = capsys.readouterr().err
        assert "4" in first and str(OMP_STREAM_VERSION) in first

        provider.parse_output(bumped, stderr)
        assert capsys.readouterr().err == "", "the WARN is once per process"


class TestAToolUseDeadlineKeepsToolContentsOutOfTheError:
    """A ``toolUse``-final stream is a deadline, whatever the tool read.

    Lane A classifies the invocation's ``error`` string with
    ``classify_streams(error, "")`` (``review_orchestrator_base``), and the
    non-zero-exit path composes ``describe_failure`` into that string
    (``llm_client``). A tool result quoting 401/429 must not reach either —
    the same rule pi's adapter keeps. The stream is the committed
    ``exit-zero-failure`` row with its tool result swapped for the markers.
    """

    MARKERS = (
        "401 Unauthorized: invalid api key. 429 Too Many Requests: rate limit "
        "exceeded. This model's maximum context length is 8192 tokens."
    )

    @pytest.fixture
    def provider(self):
        return OmpProvider()

    @classmethod
    def _stream(cls):
        stdout = (
            Path(__file__).parent / "fixtures" / "omp" / "exit-zero-failure.stdout.txt"
        ).read_text(encoding="utf-8")
        assert "Command aborted" in stdout
        return stdout.replace("Command aborted", cls.MARKERS)

    def test_the_structural_classifier_says_timeout(self, provider):
        assert provider.classify_failure(self._stream(), "", 0) == STATUS_TIMEOUT

    def test_lane_a_classifies_the_parse_error_as_timeout(self, provider):
        result = provider.parse_output(self._stream(), "")

        assert result["success"] is False
        assert result["error_code"] == ERROR_TIMEOUT
        assert "run ended inside a tool call: bash" in result["error"]
        assert "rate limit" not in result["error"].lower()
        assert "invalid api key" not in result["error"].lower()
        assert classify_streams(str(result["error"]), "") == STATUS_TIMEOUT
        # The contents survive as a diagnostic beside the error.
        assert result["tool_name"] == "bash"
        assert "rate limit exceeded" in result["tool_result"]

    def test_lane_a_classifies_the_nonzero_exit_sentence_as_timeout(self, provider):
        reason = provider.describe_failure(self._stream(), "")
        composed = f"omp exited with code 1: {reason}"

        assert "rate limit" not in composed.lower()
        assert classify_streams(composed, "") == STATUS_TIMEOUT


class TestAnApprovalDenialKeepsToolContentsOutOfTheError:
    """The denial's result line is tool-channel text, like the deadline's.

    The committed ``approval-denied`` row with its denial notice suffixed by the
    markers: the notice still matches the denial marker, and none of the
    markers may reach the classifier-facing error.
    """

    MARKERS = TestAToolUseDeadlineKeepsToolContentsOutOfTheError.MARKERS

    @pytest.fixture
    def provider(self):
        return OmpProvider()

    @classmethod
    def _stream(cls):
        stdout = (
            Path(__file__).parent / "fixtures" / "omp" / "approval-denied.stdout.txt"
        ).read_text(encoding="utf-8")
        notice = "no interactive UI available"
        assert notice in stdout
        return stdout.replace(notice, f"{notice}. {cls.MARKERS}")

    def test_the_error_names_the_tool_and_nothing_it_returned(self, provider):
        result = provider.parse_output(self._stream(), "")

        assert result["success"] is False
        assert result["error_code"] == ERROR_SUBPROCESS_FAILED
        assert "tool write was denied approval" in result["error"]
        assert "requires approval" not in result["error"]
        assert "rate limit" not in result["error"].lower()
        assert "invalid api key" not in result["error"].lower()
        # The notice survives as a diagnostic beside the error.
        assert result["tool_name"] == "write"
        assert "requires approval" in result["tool_result"]

    def test_the_structural_classifier_still_says_permission_denied(self, provider):
        assert provider.classify_failure(self._stream(), "", 0) == \
            CATEGORY_PERMISSION_DENIED

    def test_the_nonzero_exit_sentence_carries_no_tool_text(self, provider):
        reason = provider.describe_failure(self._stream(), "")

        assert "was denied approval" in reason
        assert "rate limit" not in reason.lower()
