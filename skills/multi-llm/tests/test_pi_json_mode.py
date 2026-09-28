"""pi on ``--mode json``: the parser, the haystack, usage and the argv (§7).

``plans/opencode-validator-false-rate-limit-and-denied-turn.md`` T7.4, T7.5,
T7.7, T7.8 and T7.15. The fixture rows themselves are graded by the generic
harness (``tests/test_provider_fixture_rows.py`` over
``tests/fixtures/pi/manifest.yaml``) and the diagnosis tiers by
``tests/test_stderr_diagnosis.py``; this module pins what neither of those can
see:

* :meth:`PiProvider.vendor_text` pins **omp's reduction**, one rule per case, on
  synthetic streams built from the committed live probe
  (``tests/fixtures/pi/json-mode/usage-probe-0.87.1.stdout.jsonl``) — never
  opencode's event roster, which names events pi does not emit;
* the §7.3 step-1 envelope test in :meth:`PiProvider.parse_output`, both ways;
* the graded back-compat test of the text-mode leg, over the moved-aside 0.80.6
  ``success`` capture read by explicit path;
* the capture argv equals the landed :meth:`PiProvider.build_command`;
* usage, including the compaction leg, and omp's side of it (T7.15);
* ``ask``'s recovery of a prose answer that quotes a JSON example.
"""
import copy
import json
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils.providers import base as base_module  # noqa: E402
from utils.providers import pi as pi_module  # noqa: E402
from utils.providers.omp import OMP_DIALECT, OmpProvider  # noqa: E402
from utils.providers.pi import PI_DIALECT, PiProvider  # noqa: E402

PI_FIXTURES = SKILL_ROOT / "tests" / "fixtures" / "pi"
OMP_FIXTURES = SKILL_ROOT / "tests" / "fixtures" / "omp"
PROBE = PI_FIXTURES / "json-mode" / "usage-probe-0.87.1.stdout.jsonl"
TEXT_MODE = PI_FIXTURES / "text-mode-0.80.6"

PI = PiProvider()

#: Words every shared marker table would fire on. Planted in our own prompt's
#: echo and in pi's system prompt by :func:`_stream`, so a reduction that let
#: either reach the haystack classifies them.
MARKERS = (
    "401 Unauthorized: invalid api key. 429 Too Many Requests: rate limit "
    "exceeded. This model's maximum context length is 8192 tokens."
)


def _probe_events():
    return [json.loads(line) for line in PROBE.read_text(encoding="utf-8").splitlines()]


def _row(name):
    stdout = (PI_FIXTURES / f"{name}.stdout.txt").read_text(encoding="utf-8")
    stderr_path = PI_FIXTURES / f"{name}.stderr.txt"
    stderr = stderr_path.read_text(encoding="utf-8") if stderr_path.is_file() else ""
    return stdout, stderr


def _stream(*, stop_reason="stop", text="OK", error_message=None,
            plant_markers=True, extra_before_end=()):
    """The live probe's stream with its terminal assistant message rewritten.

    Everything else is the probe's own bytes: the ``session`` header, pi's
    system prompt (twice), our prompt's echo (twice, and again in
    ``agent_end.messages``). With ``plant_markers`` the echoes and the system
    prompt's ``preamble`` carry :data:`MARKERS`.
    """
    events = copy.deepcopy(_probe_events())
    out = []
    for event in events:
        message = event.get("message")
        if plant_markers and isinstance(message, dict):
            if message.get("role") == "user":
                message["content"] = [{"type": "text", "text": f"Reply OK. {MARKERS}"}]
            if message.get("role") == "system":
                message["sections"]["preamble"] += f" {MARKERS}"
        if event.get("type") in ("message_end", "turn_end") and isinstance(message, dict) \
                and message.get("role") == "assistant":
            message["stopReason"] = stop_reason
            message["content"] = [{"type": "text", "text": text}] if text else []
            if error_message is not None:
                message["errorMessage"] = error_message
        if event.get("type") == "agent_end":
            if plant_markers:
                for entry in event.get("messages") or []:
                    if entry.get("role") == "user":
                        entry["content"] = [{"type": "text", "text": MARKERS}]
            out.extend(extra_before_end)
        out.append(event)
    return "\n".join(json.dumps(e) for e in out) + "\n"


# --------------------------------------------------------------------------- #
# vendor_text: omp's reduction, one rule per case
# --------------------------------------------------------------------------- #

class TestVendorTextPinsOmpsReduction:

    def test_a_terminal_error_is_exactly_its_error_message(self):
        message = '429: {"message":"Rate limit reached","code":"rate_limit_exceeded"}'
        stdout = _stream(stop_reason="error", text="", error_message=message)

        assert PI.vendor_text(stdout, "") == (message, "")

    @pytest.mark.parametrize("reason", ["aborted", "deferred"])
    def test_aborted_and_an_unknown_reason_are_their_error_message(self, reason):
        stdout = _stream(stop_reason=reason, text="", error_message="Request was aborted")
        assert PI.vendor_text(stdout, "") == ("Request was aborted", "")

        bare = _stream(stop_reason=reason, text="")
        assert PI.vendor_text(bare, "") == ("", "")

    @pytest.mark.parametrize("reason", ["stop", "toolUse", "length"])
    def test_stop_tooluse_and_length_reduce_stdout_to_nothing(self, reason):
        stdout = _stream(stop_reason=reason, text="the answer quotes 429 rate limit")

        assert PI.vendor_text(stdout, "") == ("", "")

    def test_markers_in_the_committed_tool_result_row_do_not_classify(self):
        """T7.1's ``markers-in-tool-result``: a clean run that quotes 401/429."""
        stdout, stderr = _row("markers-in-tool-result")
        assert "401 Unauthorized" in stdout and "Rate limit reached" in stdout

        assert PI.classify_failure(stdout, stderr, 0) is None
        # Unreduced, the same bytes classify — the defect the reduction closes.
        assert base_module.classify_streams(stdout, stderr) is not None

    def test_markers_in_a_synthetic_tool_result_do_not_classify(self):
        """The synthetic twin: a tool result, the echo and ``agent_end`` quoting
        401 and rate limit around a clean ``stop``."""
        tool_end = {
            "type": "tool_execution_end", "toolCallId": "c1", "toolName": "read",
            "isError": False,
            "result": {"content": [{"type": "text", "text": MARKERS}]},
        }
        stdout = _stream(extra_before_end=[tool_end])

        assert base_module.classify_streams(stdout, "") is not None
        assert PI.classify_failure(stdout, "", 0) is None

    @pytest.mark.parametrize("reason,error", [
        ("stop", None), ("toolUse", None), ("length", None),
        ("error", "Connection error."), ("aborted", "Request was aborted"),
        ("deferred", None),
    ])
    def test_the_echo_and_the_system_prompt_never_reach_the_haystack(self, reason, error):
        stdout = _stream(stop_reason=reason, text="", error_message=error)
        reduced, _ = PI.vendor_text(stdout, "")

        assert "maximum context length" not in reduced
        assert "Too Many Requests" not in reduced
        assert "expert coding assistant" not in reduced

    @pytest.mark.parametrize("stdout", [
        "The plan hits a 429 rate limit.",
        '{"answer": "OK"}',
        "[1, 2]",
        "",
    ])
    def test_identity_on_a_stdout_that_is_not_a_stream(self, stdout):
        assert PI.vendor_text(stdout, "err") == (stdout, "err")

    def test_a_raised_exception_inside_the_reduction_is_identity(self, monkeypatch):
        def boom(*_a, **_k):
            raise RuntimeError("reader blew up")

        monkeypatch.setattr(pi_module, "_assistant_ends", boom)
        stdout = _stream(stop_reason="error", error_message="x")

        assert PI.vendor_text(stdout, "err") == (stdout, "err")

    @pytest.mark.parametrize("reason", ["stop", "error", "length"])
    def test_stderr_is_never_reduced(self, reason):
        stdout = _stream(stop_reason=reason, text="", error_message="x")
        stderr = f"No API key found for stub.\n\n{MARKERS}"

        assert PI.vendor_text(stdout, stderr)[1] == stderr


class TestAToolUseDeadlineKeepsToolContentsOutOfTheError:
    """A ``toolUse``-final stream is a deadline, whatever the tool read.

    Lane A classifies the invocation's ``error`` string with
    ``classify_streams(error, "")`` (``review_orchestrator_base``), and the
    non-zero-exit path composes ``describe_failure`` into that string
    (``llm_client``). A file read quoting 401/429 must not reach either.
    """

    @staticmethod
    def _tool_use_stream(is_error):
        tool_end = {
            "type": "tool_execution_end", "toolCallId": "c1", "toolName": "read",
            "isError": is_error,
            "result": {"content": [{"type": "text", "text": MARKERS}]},
        }
        return _stream(stop_reason="toolUse", text="", extra_before_end=[tool_end])

    @pytest.mark.parametrize("is_error", [False, True])
    def test_the_structural_classifier_says_timeout(self, is_error):
        stdout = self._tool_use_stream(is_error)
        assert PI.classify_failure(stdout, "", 0) == base_module.STATUS_TIMEOUT

    @pytest.mark.parametrize("is_error", [False, True])
    def test_lane_a_classifies_the_parse_error_as_timeout(self, is_error):
        result = PI.parse_output(self._tool_use_stream(is_error), "")

        assert result["success"] is False
        assert "run ended inside a tool call: read" in result["error"]
        assert "rate limit" not in result["error"].lower()
        assert "invalid api key" not in result["error"].lower()
        assert (
            base_module.classify_streams(str(result["error"]), "")
            == base_module.STATUS_TIMEOUT
        )
        # The contents survive as a diagnostic beside the error.
        assert result["tool_name"] == "read"
        assert "rate limit exceeded" in result["tool_result"]

    @pytest.mark.parametrize("is_error", [False, True])
    def test_lane_a_classifies_the_nonzero_exit_sentence_as_timeout(self, is_error):
        reason = PI.describe_failure(self._tool_use_stream(is_error), "")
        composed = f"pi exited with code 1: {reason}"

        assert "rate limit" not in composed.lower()
        assert base_module.classify_streams(composed, "") == base_module.STATUS_TIMEOUT


class TestTheOverflowRecoveryFailedSentence:
    """§7.3 step 5's recovered-then-failed shape. UNMEASURED: synthetic.

    The overflow compaction succeeds (``result`` present, ``aborted: false``)
    and the retried turn fails again with pi's *Context overflow recovery
    failed* sentence, which need not carry the vendor's 400 text. No live
    capture has produced this shape; the stream is the probe's own bytes with
    that ending grafted on.
    """

    SENTENCE = "Context overflow recovery failed after one compact-and-retry attempt"

    def _recovered_then_failed(self, error_message):
        compaction = {
            "type": "compaction_end", "reason": "overflow", "aborted": False,
            "result": {"summary": "s", "usage": {"input": 10, "output": 2}},
        }
        return _stream(stop_reason="error", text="", error_message=error_message,
                       plant_markers=False, extra_before_end=[compaction])

    def test_both_lanes_say_context_length_exceeded(self):
        stdout = self._recovered_then_failed(self.SENTENCE)

        assert PI.classify_failure(stdout, "", 1) == base_module.STATUS_CONTEXT_LENGTH_EXCEEDED
        result = PI.parse_output(stdout, "")
        assert result["success"] is False
        assert (
            base_module.classify_streams(str(result["error"]), "")
            == base_module.STATUS_CONTEXT_LENGTH_EXCEEDED
        )

    @pytest.mark.parametrize("prefix", ["429", "503", "401"])
    def test_an_overflow_with_a_status_prefix_does_not_split_the_lanes(self, prefix):
        stdout = _stream(
            stop_reason="error", text="", plant_markers=False,
            error_message=f"{prefix}: This model's maximum context length is 8192 tokens",
        )
        lane_b = PI.classify_failure(stdout, "", 1)
        lane_a = base_module.classify_streams(str(PI.parse_output(stdout, "")["error"]), "")

        assert lane_b == base_module.STATUS_CONTEXT_LENGTH_EXCEEDED
        assert lane_a == lane_b


class TestTheLengthBranchKeepsTheBanner:
    def test_a_non_json_line_rides_the_length_error(self):
        stdout = "Update available: pi 9.9.9\n" + _stream(
            stop_reason="length", text="partial", plant_markers=False,
        )
        result = PI.parse_output(stdout, "")

        assert result["success"] is False
        assert "Truncated output (stopReason length" in result["error"]
        assert "non-JSON output: Update available: pi 9.9.9" in result["error"]


# --------------------------------------------------------------------------- #
# parse_output: the step-1 envelope test
# --------------------------------------------------------------------------- #

class TestTheEnvelopeTestChoosesTheParser:

    @pytest.fixture(autouse=True)
    def _quiet_warn(self, monkeypatch):
        monkeypatch.setattr(base_module, "_GATE_WARNED", set())

    @pytest.mark.parametrize("stdout,data", [
        ('{"answer":"OK"}', {"answer": "OK"}),
        ('[{"id": 1}]', [{"id": 1}]),
        ('Here it is: [{"id": 1}] done.', [{"id": 1}]),
    ])
    def test_a_bare_answer_goes_through_parse_answer_text(self, stdout, data):
        result = PI.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == data

    def test_a_stream_cut_mid_line_fails_by_name(self):
        stdout = _stream()
        cut = stdout[: stdout.index('"role": "assistant"', stdout.index('"message_end"',
                                                                       stdout.index('"role": "user"')))]
        result = PI.parse_output(cut, "")

        assert result["success"] is False
        assert "run ended before any assistant message" in result["error"]
        assert '"type": "session"' not in str(result.get("data"))

    def test_junk_lines_between_events_stay_on_the_stream_parser(self):
        lines = _stream(stop_reason="error", text="", error_message="Connection error.").splitlines()
        lines.insert(2, "Update available: pi 0.88.0")
        result = PI.parse_output("\n".join(lines), "")

        assert result["success"] is False
        assert "Connection error." in result["error"]
        assert "non-JSON output: Update available: pi 0.88.0" in result["error"]

    def test_a_header_only_stream_fails_by_name(self):
        stdout, stderr = _row("preflight-no-key")
        result = PI.parse_output(stdout, stderr)

        assert result["success"] is False
        assert result["error"].startswith("run ended before any assistant message")
        assert "No API key found for stub." in result["error"]


class TestTheTextModeLegIsGraded:
    """§7.3 step 1's back-compat leg, over the moved-aside 0.80.6 ``success``."""

    def _success(self):
        stdout = (TEXT_MODE / "success.stdout.txt").read_text(encoding="utf-8")
        stderr = (TEXT_MODE / "success.stderr.txt").read_text(encoding="utf-8")
        return stdout, stderr

    def test_it_parses_to_the_0806_manifests_answer_with_one_warn(
        self, monkeypatch, capsys
    ):
        import yaml

        monkeypatch.setattr(base_module, "_GATE_WARNED", set())
        stdout, stderr = self._success()
        expected = yaml.safe_load(
            (TEXT_MODE / "manifest.yaml").read_text(encoding="utf-8")
        )["rows"]["success"]["expect"]["data"]

        first = PI.parse_output(stdout, stderr)
        warned = capsys.readouterr().err
        second = PI.parse_output(stdout, stderr)
        again = capsys.readouterr().err

        assert first["success"] is True and first["data"] == expected
        assert second == first
        assert warned.count("WARNING") == 1
        assert "0.87.1" in warned and "--mode json" in warned
        assert again == ""

    def test_vendor_text_is_identity_on_it(self):
        stdout, stderr = self._success()

        assert PI.vendor_text(stdout, stderr) == (stdout, stderr)


# --------------------------------------------------------------------------- #
# The argv the fixtures were captured under is the argv that ships (T7.5)
# --------------------------------------------------------------------------- #

class TestTheCaptureArgvIsTheShippedArgv:

    def test_build_command_equals_the_recorded_capture_argv(self):
        """§10 step 5: T7.1 ran before T7.5 existed, so its argv was assembled by
        hand. If this fails, the captures are redone against the landed code —
        the argv is never edited to match.

        **One known, pinned delta since the captures:** the ``--tools``
        allow-list (review-run-scope-salvage-and-failure-visibility T3.1) is
        rendered only when the installed pi's help documents it, so the argv
        the captures were made under is the one a pi *without* the flag gets.
        That is compared exactly; the shipped argv must then equal it plus the
        one ``--tools`` pair right after ``--mode json`` and nothing else. The
        captures are owed a re-run under the new argv (pi's echoed system prompt
        names its active tools); until then this keeps the difference to that
        one pair rather than letting either side drift further.
        """
        from unittest.mock import patch

        recorded = json.loads(
            (PI_FIXTURES / "capture" / "argv.json").read_text(encoding="utf-8")
        )["pi"]

        with patch.object(PiProvider, "supports_flag", lambda self, flag: False):
            declined = PI.build_command("PROMPT_PLACEHOLDER", "stub/stub-model")
        with patch.object(PiProvider, "supports_flag", lambda self, flag: True):
            shipped = PI.build_command("PROMPT_PLACEHOLDER", "stub/stub-model")

        assert declined == recorded
        tools = ["--tools", ",".join(pi_module.PI_TOOLS)]
        assert shipped == recorded[:5] + tools + recorded[5:]

    def test_the_manifest_names_the_version_the_adapter_verifies(self):
        import yaml

        manifest = yaml.safe_load((PI_FIXTURES / "manifest.yaml").read_text(encoding="utf-8"))

        assert manifest["cli_version"] == PiProvider.verified_version == "0.87.1"
        assert PI_DIALECT.verified_version == "0.87.1"


# --------------------------------------------------------------------------- #
# Usage (T7.7)
# --------------------------------------------------------------------------- #

class TestUsage:

    def test_the_probe_returns_the_five_numbers(self):
        usage = PI.parse_usage(PROBE.read_text(encoding="utf-8"), "")

        assert usage == {
            "input_tokens": 1445,
            "output_tokens": 3,
            "cached_input_tokens": 256,
            "total_tokens": 1704,
            "cost_usd": pytest.approx(0.000219318),
        }

    def test_multi_assistant_sums_both_turns_and_counts_the_tool(self):
        stdout, stderr = _row("multi-assistant")
        usage = PI.parse_usage(stdout, stderr)

        assert usage["tool_uses"] == 1
        # The stub streams no usage: both turns are pi's zeroed blocks, summed.
        assert usage["output_tokens"] == 0 and usage["input_tokens"] == 0

    def test_a_synthetic_two_turn_stream_sums(self):
        events = _probe_events()
        end = next(e for e in events if e.get("type") == "message_end"
                   and e["message"]["role"] == "assistant")
        stdout = PROBE.read_text(encoding="utf-8") + json.dumps(end) + "\n"
        usage = PI.parse_usage(stdout, "")

        assert usage["input_tokens"] == 2 * 1445
        assert usage["output_tokens"] == 2 * 3

    def test_an_overflow_with_compaction_adds_the_summarization_call(self):
        stdout, stderr = _row("context-overflow-compacted")
        usage = PI.parse_usage(stdout, stderr)

        # Assistant turns: 2000+12000+22000+32000 (tools) + 0 (the 400) + 3000
        # (the answer); the compaction's summarization call: 900 in / 40 out.
        assert usage["input_tokens"] == 71000 + 900
        assert usage["output_tokens"] == 140 + 40
        assert usage["total_tokens"] == 71140 + 940
        assert usage["tool_uses"] == 4

    def test_an_error_row_reports_its_zeroed_block_and_invents_nothing(self):
        stdout, stderr = _row("auth-error")
        assert PI.parse_usage(stdout, stderr) == {
            "input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0,
            "total_tokens": 0, "cost_usd": 0,
        }

        no_usage = "\n".join(
            json.dumps({k: v for k, v in {**e, "message": {
                k2: v2 for k2, v2 in e["message"].items() if k2 != "usage"}}.items()})
            if e.get("type") == "message_end" and e["message"].get("role") == "assistant"
            else json.dumps(e)
            for e in _probe_events()
        )
        assert PI.parse_usage(no_usage, "") is None

    def test_a_header_only_or_text_mode_stdout_has_no_usage(self):
        assert PI.parse_usage(*_row("preflight-no-key")) is None
        assert PI.parse_usage("[1, 2]", "") is None

    def test_usage_support_is_full(self):
        assert PiProvider.usage_support == base_module.USAGE_FULL


class TestOmpCompactionUsage:
    """T7.15: omp's compaction carries no usage, so its leg stays off."""

    def test_the_measured_compaction_result_has_no_usage(self):
        stdout = (OMP_FIXTURES / "compaction-summarized.stdout.txt").read_text(encoding="utf-8")
        ends = [json.loads(line) for line in stdout.splitlines()
                if '"auto_compaction_end"' in line]
        ran = [e for e in ends if isinstance(e.get("result"), dict)]

        assert [e["action"] for e in ran] == ["context-full"]
        assert "usage" not in ran[0]["result"]
        assert OMP_DIALECT.compaction_usage is False

    def test_the_run_under_counts_by_exactly_its_summarization_calls(self):
        stdout = (OMP_FIXTURES / "compaction-summarized.stdout.txt").read_text(encoding="utf-8")
        usage = OmpProvider().parse_usage(stdout, "")

        # 4 tool turns + the 400 + the answer; the two summarization requests
        # (900 in / 40 out each on the stub) are nowhere in the stream.
        assert usage["input_tokens"] == 2000 + 12000 + 22000 + 32000 + 3000
        assert usage["output_tokens"] == 4 * 20 + 60

    def test_the_existing_context_overflow_row_is_unchanged(self):
        """Its compactions were skipped (no summarization); pinned so a future
        compaction leg cannot move it."""
        stdout = (OMP_FIXTURES / "context-overflow.stdout.txt").read_text(encoding="utf-8")
        usage = OmpProvider().parse_usage(stdout, "")
        assert usage == OMP_CONTEXT_OVERFLOW_USAGE

    def test_the_docstring_records_the_under_count(self):
        doc = OmpProvider.parse_usage.__doc__ or ""
        assert "under-counts" in doc and "compaction-summarized" in doc


# --------------------------------------------------------------------------- #
# ask recovery (T7.8)
# --------------------------------------------------------------------------- #

class TestAskRecovery:

    def test_a_prose_answer_quoting_json_recovers_its_full_text(self):
        """omp's regression (``omp.py`` ``setdefault("raw", answer)``), for pi:
        the log's stdout is JSONL, which ``looks_like_ndjson`` rejects, so
        without ``raw`` the answer would recover as nothing."""
        from ask_orchestrator import recover_answer_text

        answer = (
            "Two options. The second one needs this block:\n\n"
            '```json\n[{"name": "b"}]\n```\n\nI would take the first.'
        )
        result = PI.parse_output(_stream(text=answer, plant_markers=False), "")

        assert result["success"] is True
        assert result["data"] == [{"name": "b"}]
        assert recover_answer_text(result) == answer


#: omp's ``context-overflow`` row's usage as it read before T7.15 (18.2.2): one
#: request, a zeroed ``error`` message, two skipped compactions.
OMP_CONTEXT_OVERFLOW_USAGE = {
    "input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0,
    "total_tokens": 0, "cost_usd": 0,
}


# --------------------------------------------------------------------------- #
# a2 grades pi structurally (T7.12, decision 12.9)
# --------------------------------------------------------------------------- #

class TestA2GradesPiStructurally:
    """``harness.grade_structured_model_echo`` on synthetic streams built from
    the probe, whose terminal message says ``provider: "openrouter"``,
    ``model: "deepseek/deepseek-v4.1-flash"`` for the requested
    ``openrouter/deepseek/deepseek-v4.1-flash``. The live check runs on the next
    pi live-lane row; nothing here spends."""

    PINNED = "openrouter/deepseek/deepseek-v4.1-flash"

    @staticmethod
    def _grade(stdout, pinned=PINNED):
        from tests.container import harness

        return harness.grade_structured_model_echo("pi", pinned, stdout)

    @staticmethod
    def _with_terminal(**fields):
        """The probe with its assistant ``message_end`` fields overridden."""
        out = []
        for event in _probe_events():
            message = event.get("message")
            if event.get("type") == "message_end" and isinstance(message, dict) \
                    and message.get("role") == "assistant":
                message.update(fields)
            out.append(json.dumps(event))
        return "\n".join(out) + "\n"

    def test_the_probes_own_terminal_message_passes(self):
        ok, outcome = self._grade(PROBE.read_text(encoding="utf-8"))

        assert ok, outcome
        assert "(from model)" in outcome

    def test_the_requested_model_with_a_different_response_model_fails(self):
        """The substitution a substring search would pass: ``model`` is the
        request, ``responseModel`` names what was served."""
        stdout = self._with_terminal(responseModel="qwen/qwen3.8-max")
        assert self.PINNED.partition("/")[2] in stdout  # the substring is there

        ok, outcome = self._grade(stdout)

        assert not ok
        assert "qwen/qwen3.8-max" in outcome and "responseModel" in outcome

    def test_a_response_model_equal_to_the_request_passes(self):
        ok, outcome = self._grade(
            self._with_terminal(responseModel="deepseek/deepseek-v4.1-flash"))

        assert ok, outcome
        assert "(from responseModel)" in outcome

    def test_a_mismatched_provider_fails(self):
        ok, outcome = self._grade(self._with_terminal(provider="deepseek"))

        assert not ok
        assert "provider 'deepseek'" in outcome

    def test_an_earlier_right_message_does_not_save_a_wrong_terminal_one(self):
        good = PROBE.read_text(encoding="utf-8")
        bad = self._with_terminal(responseModel="qwen/qwen3.8-max")
        end = next(line for line in bad.splitlines()
                   if '"message_end"' in line and '"role": "assistant"' in line)

        ok, _ = self._grade(good + end + "\n")

        assert not ok

    def test_no_terminal_assistant_message_fails_by_name(self):
        ok, outcome = self._grade(_row("preflight-no-key")[0])

        assert not ok
        assert "no terminal assistant message_end" in outcome

    def test_pi_is_graded_structurally_and_not_by_substring(self):
        from tests.container import harness

        source = (SKILL_ROOT / "tests" / "test_container_live_provider.py").read_text(
            encoding="utf-8")
        no_echo = source[source.index("NO_MODEL_ECHO: dict[str, str] = {"):
                         source.index("OPT_OUTS: dict")]
        drops = source[source.index("MODEL_ECHO_DROPS_FIRST_SEGMENT: dict[str, str] = {"):
                       source.index("MODEL_ECHO_ON_STDERR: dict")]

        assert set(harness.MODEL_ECHO_STRUCTURED) == {"pi"}
        assert '"pi": (' not in no_echo
        assert '"pi": (' not in drops


# --------------------------------------------------------------------------- #
# salvage_text: every committed row against its own `expect.salvage` label
# (plan: review-run-scope-salvage-and-failure-visibility §4, T4.1)
# --------------------------------------------------------------------------- #

def _labelled_rows(fixtures):
    """``(row name, row)`` for every manifest row that names captured files."""
    import yaml

    manifest = yaml.safe_load((fixtures / "manifest.yaml").read_text(encoding="utf-8"))
    return [
        (name, row) for name, row in manifest["rows"].items()
        if "not_applicable" not in row
    ]


def salvage_gate_input(adapter, fixtures, row):
    """What the salvage gate would judge for this row, through the real plumbing.

    The two ``llm_client`` attachments every losing path makes, then the one
    rule both salvage gates apply (``salvage_source``): exactly the text a
    ``[SALVAGE_NEEDED]`` decision is taken on.
    """
    from utils import llm_client
    from utils.review_orchestrator_base import salvage_source

    stdout = (fixtures / row["stdout"]).read_text(encoding="utf-8") if row.get("stdout") else ""
    stderr = (fixtures / row["stderr"]).read_text(encoding="utf-8") if row.get("stderr") else ""
    details = {"stderr": stderr}
    llm_client._attach_stdout_excerpt(details, stdout)
    llm_client._attach_salvage_text(details, adapter, stdout)
    return salvage_source(details)


def assert_meets_salvage_label(name, row, gate_input):
    """The label's rule — decided by the label, never by ``expect.success``."""
    from utils.json_extractor import extract_json_from_text, has_salvageable_json

    expect = row["expect"]
    assert "salvage" in expect, (
        f"{name}: no `expect.salvage` label. Label the row by what its bytes hold "
        f"(none / partial / answer), never by whether it succeeded."
    )
    label = expect["salvage"]
    if label == "none":
        assert not has_salvageable_json(gate_input), (
            f"{name} holds no answer text, yet the salvage gate passes on "
            f"{gate_input[:200]!r}"
        )
    elif label == "partial":
        assert has_salvageable_json(gate_input), (
            f"{name} holds a partial answer the gate rejects: {gate_input[:200]!r}"
        )
    else:
        assert label == "answer", f"{name}: unknown label {label!r}"
        assert has_salvageable_json(gate_input), name
        extracted = extract_json_from_text(gate_input)
        assert extracted.get("success"), (
            f"{name}: a whole answer that extract_json_from_text cannot recover: "
            f"{extracted.get('error')}"
        )
        if "data" in expect:
            assert extracted["data"] == expect["data"], name


PI_SALVAGE_ROWS = _labelled_rows(PI_FIXTURES)


class TestPiSalvageTextMeetsEveryRowsLabel:
    """Every pi row with files, through ``PiProvider.salvage_text`` and the gate.

    The §4.1 baseline was 18 of 29 pi captures passing the gate on the raw
    stream (``auth-error``, ``empty``, ``error-envelope``, ``context-overflow``
    …); with the reducer only the rows that really hold answer text do.
    """

    def test_the_sweep_covers_the_rows_that_matter(self):
        names = {name for name, _row in PI_SALVAGE_ROWS}
        assert {"killed-mid-toolcall", "length", "length-recovery",
                "auth-error", "success"} <= names

    @pytest.mark.parametrize(
        "name,row", PI_SALVAGE_ROWS, ids=[name for name, _row in PI_SALVAGE_ROWS]
    )
    def test_row(self, name, row):
        assert_meets_salvage_label(name, row, salvage_gate_input(PI, PI_FIXTURES, row))

    def test_the_incident_passed_the_gate_before_the_reducer(self):
        """The bug, pinned: the raw stream of the killed reviewer passes the gate
        (its ``content`` arrays read as findings) while the reduction does not."""
        from utils.json_extractor import has_salvageable_json

        stdout = (PI_FIXTURES / "killed-mid-toolcall.stdout.txt").read_text(encoding="utf-8")

        assert has_salvageable_json(stdout)
        assert PI.salvage_text(stdout) == ""

    def test_killed_mid_text_reduces_to_the_cut_array_without_the_narration(self):
        stdout = (PI_FIXTURES / "killed-mid-text.stdout.txt").read_text(encoding="utf-8")

        text = PI.salvage_text(stdout)

        assert text.startswith("[\n  {")
        assert "The plan's retry schedule" not in text
        assert "[2, 4, 8]" not in text

    @pytest.mark.parametrize("stdout", ["", "  \n"])
    def test_an_empty_stdout_keeps_an_array_shaped_stderr_off_the_gate(self, stdout):
        """A slot killed before pi printed its header leaves no stdout at all.
        ``generate`` always renders ``--mode json``, so that is a transport that
        carried nothing: the reducer says ``""`` and the gate never falls back to
        a stderr diagnostic that is itself array-shaped (D4.3)."""
        from utils import llm_client
        from utils.json_extractor import has_salvageable_json
        from utils.review_orchestrator_base import salvage_source

        stderr = '{"error": "authentication failed", "attempts": []}'
        assert has_salvageable_json(stderr)

        details = {"stderr": stderr}
        llm_client._attach_stdout_excerpt(details, stdout)
        llm_client._attach_salvage_text(details, PI, stdout)

        assert details[llm_client.SALVAGE_TEXT_KEY] == ""
        assert salvage_source(details) == ""
        assert not has_salvageable_json(salvage_source(details))
