"""Tests for the Muse adapter.

One of the per-provider test files; helpers that several of them share live in
``provider_test_support.py``. See docs/development/providers.md.
"""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.muse import MuseProvider


class TestMuseProvider:
    """Tests for the MuseProvider class."""

    @pytest.fixture
    def provider(self):
        """Create a MuseProvider instance."""
        return MuseProvider()

    @staticmethod
    def _record(payload_type, payload, seq=1):
        """Wrap a payload in muse's JSONL envelope."""
        return json.dumps({
            "schema_version": 1,
            "id": f"018f0000-0000-7000-8000-{seq:012d}",
            "stream": {"kind": "session", "id": "sess-1"},
            "sequence": seq,
            "recorded_at": 1780531400000000 + seq,
            "record_type": "event",
            "durability": "durable",
            "causation_id": "cmd-1",
            "payload_type": payload_type,
            "payload_schema_version": 1,
            "payload": payload,
        })

    @classmethod
    def _stream(cls, *records):
        """Join envelope records into a JSONL stdout blob."""
        return "\n".join(
            cls._record(payload_type, payload, seq)
            for seq, (payload_type, payload) in enumerate(records, start=1)
        )

    @classmethod
    def _delta(cls, text):
        return ("run.output.delta", {"kind": "run_output_delta", "text": text})

    @classmethod
    def _tool_result(cls, text="Read text file `note.txt`."):
        return ("tool.result", {"kind": "tool_result", "call_id": "call_1", "text": text})

    @classmethod
    def _completed(cls, text):
        return ("run.terminal.completed", {
            "kind": "run_terminal", "terminal": "completed", "reason": None, "text": text,
        })

    @classmethod
    def _failed(cls, reason):
        return ("run.terminal.failed", {
            "kind": "run_terminal", "terminal": "failed", "reason": reason, "text": "",
        })

    def test_name_property(self, provider):
        """Test that name property returns correct identifier."""
        assert provider.name == "muse"

    def test_default_timeout(self, provider):
        """No per-adapter timeout: docs/configuration.md#provider-blocks."""
        from utils.providers.base import DEFAULT_TIMEOUT_S

        # The shared fallback; the timeout a run uses is providers.yaml's row.
        assert provider.default_timeout == DEFAULT_TIMEOUT_S

    def test_muse_parse_terminal_array(self, provider):
        """Extract a JSON array from the terminal record's text."""
        inner_data = [{"title": "Finding 1", "desc": "Description", "importance": "medium"}]
        stdout = self._stream(self._delta(json.dumps(inner_data)),
                              self._completed(json.dumps(inner_data)))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_muse_parse_terminal_object(self, provider):
        """Extract a JSON object from the terminal record's text."""
        inner_data = {"first_heading": "Probe Widget"}
        stdout = self._stream(self._completed(json.dumps(inner_data)))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_muse_parse_code_block_json(self, provider):
        """Extract JSON wrapped in a markdown code block."""
        inner_data = [{"title": "Fenced", "desc": "d", "importance": "low"}]
        text = f"Here you go:\n```json\n{json.dumps(inner_data)}\n```"
        stdout = self._stream(self._completed(text))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_muse_parse_prose_with_json(self, provider):
        """Extract JSON embedded in surrounding prose."""
        inner_data = [{"title": "Inline", "desc": "d", "importance": "high"}]
        stdout = self._stream(self._completed(f"My findings: {json.dumps(inner_data)} Done."))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_muse_parse_closing_message_wins_over_draft(self, provider):
        """The terminal text glues every assistant message together with no
        separator, so an early draft precedes the answer. The deltas after the
        last tool.result isolate the closing message."""
        draft = [{"title": "draft", "desc": "early", "importance": "low"}]
        answer = [{"title": "final", "desc": "answer", "importance": "high"}]
        stdout = self._stream(
            self._delta(f"Draft: {json.dumps(draft)}"),
            self._tool_result(),
            self._delta(json.dumps(answer)),
            self._completed(f"Draft: {json.dumps(draft)}{json.dumps(answer)}"),
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == answer

    def test_muse_parse_split_deltas_reassemble(self, provider):
        """Deltas are token fragments, not messages: a single document arrives
        split mid-token and must be rejoined without a separator."""
        answer = [{"title": "split", "desc": "d", "importance": "low"}]
        blob = json.dumps(answer)
        third = len(blob) // 3
        stdout = self._stream(
            self._tool_result(),
            self._delta(blob[:third]),
            self._delta(blob[third:2 * third]),
            self._delta(blob[2 * third:]),
            self._completed(blob),
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == answer

    def test_muse_parse_prose_before_answer_same_turn(self, provider):
        """Narration and answer in the closing turn still extract the answer."""
        answer = [{"title": "final", "desc": "answer", "importance": "high"}]
        text = f"Looks good so far.{json.dumps(answer)}"
        stdout = self._stream(self._delta(text), self._completed(text))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == answer

    def test_muse_parse_falls_back_to_deltas_without_terminal(self, provider):
        """A truncated stream with no terminal record still yields the deltas."""
        answer = [{"title": "partial", "desc": "d", "importance": "low"}]
        stdout = self._stream(self._delta(json.dumps(answer)))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == answer

    def test_muse_parse_terminal_failed(self, provider):
        """A failed terminal record surfaces its reason rather than "no output"."""
        reason = ("API error 402 [request_id=abc]: Billing verification failed. "
                  "Please check your payment method. (billing_error) "
                  "(after 10 provider attempts)")
        stdout = self._stream(self._failed(reason))

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "failed" in result["error"]
        assert "Billing verification failed" in result["error"]

    def test_muse_parse_failed_with_unparseable_partial_text(self, provider):
        """A cancelled/failed run that emitted prose keeps its terminal reason."""
        stdout = self._stream(
            self._delta("I was part way through when"),
            ("run.terminal.cancelled", {
                "kind": "run_terminal", "terminal": "cancelled",
                "reason": "interrupted by signal", "text": "I was part way through when",
            }),
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "interrupted by signal" in result["error"]

    def test_muse_parse_no_events(self, provider):
        """Empty output reports a parse failure."""
        result = provider.parse_output("", "")

        assert result["success"] is False
        assert result["data"] is None

    def test_muse_parse_skips_malformed_lines(self, provider):
        """Malformed JSONL lines are skipped, not fatal."""
        answer = [{"title": "ok", "desc": "d", "importance": "low"}]
        stdout = "not json\n" + self._stream(self._completed(json.dumps(answer))) + "\n{broken"

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == answer

    def test_muse_parse_garbage_text(self, provider):
        """Terminal text with no JSON reports a parse failure."""
        stdout = self._stream(self._completed("I could not complete the review."))

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert result["data"] is None

    def test_muse_describe_failure_reads_terminal_reason(self, provider):
        """muse leaves only "run ended with Failed" on stderr; the cause is on stdout."""
        reason = "API error 402 [request_id=abc]: Billing verification failed."
        stdout = self._stream(self._failed(reason))

        assert provider.describe_failure(stdout, "run ended with Failed\n") == reason

    def test_muse_describe_failure_falls_back_to_task_failures(self, provider):
        """Without a terminal record, task-level failures carry the diagnosis."""
        stdout = self._stream(
            ("task.lifecycle.failed", {
                "kind": "task_lifecycle",
                "event": {"kind": "failed", "reason": "model failed: API error 402"},
            }),
        )

        assert provider.describe_failure(stdout, "") == "model failed: API error 402"

    def test_muse_describe_failure_returns_none_when_silent(self, provider):
        """Nothing to add beyond the exit code returns None."""
        assert provider.describe_failure("", "") is None

    def test_muse_build_command(self, provider):
        """Build the headless command without an effort suffix."""
        cmd = provider.build_command("Analyze this", "muse-spark-1.2")

        assert cmd == [
            "muse", "exec", "--json", "--disable-approval", "--disable-sandbox",
            "--model", "muse-spark-1.2",
            "Analyze this",
        ]

    def test_muse_build_command_disables_approval(self, provider):
        """--disable-approval is mandatory: an approval prompt hangs headless runs."""
        cmd = provider.build_command("Analyze this", "muse-spark-1.2")

        assert "--disable-approval" in cmd

    def test_muse_build_command_disables_sandbox(self, provider):
        """D6: muse's shell sandbox is bubblewrap, which a stock Ubuntu 24.04
        cannot start, so every sandboxed write died on `bwrap: loopback`.
        `--disable-sandbox` turns it off; `--yolo` would also grant workspace
        trust, which stays the user's own store."""
        cmd = provider.build_command("Analyze this", "muse-spark-1.2")

        assert "--disable-sandbox" in cmd
        assert cmd.index("--disable-sandbox") == cmd.index("--disable-approval") + 1
        assert "--yolo" not in cmd
        assert "--trust-workspace" not in cmd

    def test_muse_build_command_effort_suffix(self, provider):
        """A whitelisted :effort suffix becomes --reasoning-effort."""
        cmd = provider.build_command("Analyze this", "muse-spark-1.2:xhigh")

        assert cmd == [
            "muse", "exec", "--json", "--disable-approval", "--disable-sandbox",
            "--model", "muse-spark-1.2",
            "--reasoning-effort", "xhigh",
            "Analyze this",
        ]

    def test_muse_build_command_unknown_suffix_passthrough(self, provider):
        """An unrecognised suffix stays part of the model name (muse hard-errors
        on an invalid --reasoning-effort, so it must never be forwarded)."""
        cmd = provider.build_command("Analyze this", "muse-spark-1.2:turbo")

        assert cmd == [
            "muse", "exec", "--json", "--disable-approval", "--disable-sandbox",
            "--model", "muse-spark-1.2:turbo",
            "Analyze this",
        ]
        assert "--reasoning-effort" not in cmd

    def test_muse_reasoning_efforts_constant(self):
        """The whitelist matches the values muse's CLI accepts."""
        from utils.providers.muse import REASONING_EFFORTS

        assert REASONING_EFFORTS == {
            "none", "minimal", "low", "medium", "high", "xhigh", "ultra"
        }

    @patch("shutil.which")
    def test_muse_is_available_true(self, mock_which, provider):
        """Test is_available returns True when muse is found."""
        mock_which.return_value = "/usr/bin/muse"

        assert provider.is_available() is True
        mock_which.assert_called_once_with("muse")

    @patch("shutil.which")
    def test_muse_is_available_false(self, mock_which, provider):
        """Test is_available returns False when muse is not found."""
        mock_which.return_value = None

        assert provider.is_available() is False
        mock_which.assert_called_once_with("muse")
