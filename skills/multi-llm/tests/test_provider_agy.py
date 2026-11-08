"""Tests for the agy adapter.

One of the per-provider test files; helpers that several of them share live in
``provider_test_support.py``. See docs/development/providers.md.
"""

import json
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.agy import AgyProvider


class TestAgyProvider:
    """Tests for the AgyProvider class."""

    @pytest.fixture
    def provider(self):
        """Create an AgyProvider instance."""
        return AgyProvider()

    @staticmethod
    def _envelope(response, status="SUCCESS", error=None):
        """Build representative agy --output-format json stdout."""
        envelope = {
            "conversation_id": "conv-123" if status == "SUCCESS" else "",
            "status": status,
            "response": response,
            "duration_seconds": 4.2,
            "num_turns": 1,
            "usage": {"input_tokens": 50, "output_tokens": 50},
        }
        if error is not None:
            envelope["error"] = error
        return json.dumps(envelope)

    def test_name_property(self, provider):
        """Test that name property returns correct identifier."""
        assert provider.name == "agy"

    def test_default_timeout(self, provider):
        """The fallback twin of `providers.yaml`'s 1200 (§9.6.d).

        It read 600 while the yaml read 1200 and the argv read `20m`: three
        numbers for one fact. Raising it is a no-op on every real path — agy
        ships in base config, so `get_provider_timeout` returns the yaml value
        and never reaches the adapter default — which is what made it safe to
        reconcile here rather than schedule separately.
        """
        assert provider.default_timeout == 1200

    def test_agy_parse_success_array(self, provider):
        """Extract a JSON array from the response field."""
        inner_data = [
            {"title": "Finding 1", "desc": "Description", "importance": "medium"}
        ]
        stdout = self._envelope(json.dumps(inner_data))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_agy_parse_success_object(self, provider):
        """Extract a JSON object from the response field."""
        inner_data = {"first_heading": "Probe Widget"}
        stdout = self._envelope(json.dumps(inner_data))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_agy_parse_code_block_json(self, provider):
        """Extract JSON from a ```json code block in the response."""
        inner_json = [{"task": "T001", "status": "done"}]
        response = f"Analysis complete:\n\n```json\n{json.dumps(inner_json)}\n```\n"
        stdout = self._envelope(response)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_agy_parse_prose_with_json(self, provider):
        """Extract embedded JSON when prose surrounds it in the response."""
        inner_json = {"result": "success"}
        stdout = self._envelope(f"Here is the JSON you asked for: {json.dumps(inner_json)}")

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_agy_parse_trailing_newline_response(self, provider):
        """The response field carries a trailing newline; it is stripped."""
        stdout = self._envelope("[1, 2, 3]\n")

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [1, 2, 3]

    def test_agy_parse_error_envelope(self, provider):
        """ERROR status (exit 1) -> failure carrying the envelope's error."""
        error_message = "You must sign in to use the Antigravity CLI."
        stdout = self._envelope("", status="ERROR", error=error_message)

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert result["error"] == error_message
        assert result["data"] is None

    def test_agy_parse_non_success_status_without_error(self, provider):
        """A non-SUCCESS status with no error field gets a fallback message."""
        stdout = self._envelope("", status="TIMEOUT")

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "TIMEOUT" in result["error"]
        assert result["data"] is None

    def test_agy_parse_empty_response(self, provider):
        """SUCCESS with an empty response field -> failure."""
        stdout = self._envelope("")

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result

    def test_agy_parse_envelope_missing_status(self, provider):
        """A JSON dict without a status field is returned as the data itself."""
        stdout = json.dumps({"result": "success", "items": [1, 2]})

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == {"result": "success", "items": [1, 2]}

    def test_agy_parse_non_json_stdout_fallback(self, provider):
        """Fallback extraction when stdout is not a JSON envelope but embeds JSON."""
        stdout = "Some banner text [1, 2, 3] trailing text"

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [1, 2, 3]

    def test_agy_parse_garbage_stdout(self, provider):
        """Handle stdout with no envelope and no extractable JSON."""
        stdout = "not valid json at all"

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result
        assert "raw" in result

    def test_agy_build_command(self, provider):
        """Exact argv: flags order, display-name model verbatim, -p prompt."""
        cmd = provider.build_command("Analyze this", "Gemini 3.1 Pro (High)")

        assert cmd == [
            "agy",
            "--new-project",
            "--dangerously-skip-permissions",
            "--output-format",
            "json",
            "--print-timeout",
            "20m",
            "--model",
            "Gemini 3.1 Pro (High)",
            "-p",
            "Analyze this",
        ]

    # --- L11 #3: agy has a real --effort flag (§13.4 #3) --------------------
    #
    # `agy --help` (agy 1.1.13, offline, no auth):
    #   --effort   Reasoning effort for the current CLI session (low|medium|high)
    # The adapter and providers.yaml both used to state that agy has no
    # `:effort` suffix support, so every effort a user configured was refused by
    # *our* whitelist for a flag agy really has.

    def test_agy_reasoning_efforts_constant(self):
        """REASONING_EFFORTS is exactly what `agy --help` lists."""
        from utils.providers.agy import REASONING_EFFORTS

        assert set(REASONING_EFFORTS) == {"low", "medium", "high"}

    def test_agy_build_command_effort_suffix_exact(self, provider):
        """A `:effort` suffix becomes `--effort <level>`, model name intact.

        agy display names carry no colon of their own, so the last-colon split
        cannot damage one — `"Gemini 3.1 Pro (High)"` survives verbatim as a
        single argv element while `high` moves onto the flag.
        """
        cmd = provider.build_command("Analyze this", "Gemini 3.1 Pro (High):high")

        assert cmd == [
            "agy",
            "--new-project",
            "--dangerously-skip-permissions",
            "--output-format",
            "json",
            "--print-timeout",
            "20m",
            "--model",
            "Gemini 3.1 Pro (High)",
            "--effort",
            "high",
            "-p",
            "Analyze this",
        ]

    def test_agy_build_command_all_reasoning_efforts(self, provider):
        """Each accepted level renders the flag; none of them touches the model."""
        for effort in ("low", "medium", "high"):
            cmd = provider.build_command("p", f"Gemini 3.1 Pro (High):{effort}")

            assert cmd[-4:] == ["--effort", effort, "-p", "p"], (
                f"effort {effort!r} did not build the expected command: {cmd}")
            assert "Gemini 3.1 Pro (High)" in cmd

    def test_agy_build_command_effort_passthrough(self, provider):
        """An unknown suffix is not an effort: it stays part of the model name.

        agy rejects an unknown model outright with an ERROR envelope, so
        passing it through is the loud failure — inventing an `--effort xhigh`
        agy does not accept would be the quiet one.
        """
        cmd = provider.build_command("p", "Gemini 3.1 Pro (High):xhigh")

        assert "--effort" not in cmd
        assert cmd[cmd.index("--model") + 1] == "Gemini 3.1 Pro (High):xhigh"

    @patch("shutil.which")
    def test_agy_is_available_true(self, mock_which, provider):
        """Test is_available returns True when agy is found."""
        mock_which.return_value = "/usr/bin/agy"

        assert provider.is_available() is True
        mock_which.assert_called_once_with("agy")

    @patch("shutil.which")
    def test_agy_is_available_false(self, mock_which, provider):
        """Test is_available returns False when agy is not found."""
        mock_which.return_value = None

        assert provider.is_available() is False
        mock_which.assert_called_once_with("agy")


def parse_go_duration(text):
    """Seconds in a Go duration string, or ``None`` when it is not one.

    Deliberately narrow: only the spellings this adapter may emit
    (``1170s``, ``19m30s``, ``1h0m0s``). `time.ParseDuration` accepts the
    bare-seconds form, which is why §9.6.c renders whole seconds and removes
    the minute-rounding rule instead of specifying one — rounding to minutes
    is exactly what let 1200 s and `20m` become an exact tie.
    """
    import re

    match = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", str(text))
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds = (int(g or 0) for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


class TestParseGoDurationHelper:
    """The ladder invariant is only as good as the parser it reads with."""

    def test_bare_seconds(self):
        assert parse_go_duration("1170s") == 1170

    def test_minutes_and_seconds(self):
        assert parse_go_duration("19m30s") == 1170

    def test_minutes_only(self):
        assert parse_go_duration("20m") == 1200

    def test_non_duration_is_none(self):
        assert parse_go_duration("soon") is None
        assert parse_go_duration("") is None


class TestAgyPrintTimeoutDerivation:
    """§9.6.e's acceptance table, above and below 20 minutes."""

    PROMPT = "Analyze this"
    MODEL = "Gemini 3.1 Pro (High)"

    @pytest.fixture
    def provider(self):
        return AgyProvider()

    @staticmethod
    def _print_timeout(cmd):
        assert "--print-timeout" in cmd, f"no --print-timeout in {cmd}"
        return cmd[cmd.index("--print-timeout") + 1]

    def test_the_constants_are_named_not_literals(self):
        """§9.6.c: the margin is a named constant, never a literal in the builder."""
        from utils.providers.agy import INNER_SHUTDOWN_MARGIN_S, MIN_INNER_TIMEOUT_S

        assert INNER_SHUTDOWN_MARGIN_S == 30
        assert MIN_INNER_TIMEOUT_S == 60

    @pytest.mark.parametrize("configured,expected", [
        (600, "570s"),
        (1200, "1170s"),
        # ABOVE 20 minutes — the silent no-op. Today this renders `20m`, agy
        # self-terminates at 1200 s and blames itself.
        (2400, "2370s"),
    ])
    def test_derived_from_the_resolved_timeout(self, provider, configured, expected):
        cmd = provider.build_command(self.PROMPT, self.MODEL, timeout_s=configured)

        assert self._print_timeout(cmd) == expected

    def test_below_the_margin_clamps_to_the_floor_and_warns_once(self, provider, capsys):
        """§9.6.c's misconfiguration guard: one WARN naming both numbers."""
        cmd = provider.build_command(self.PROMPT, self.MODEL, timeout_s=45)

        assert self._print_timeout(cmd) == "60s"
        captured = capsys.readouterr()
        assert captured.err.count("WARN") == 1, (
            f"expected exactly one WARN, got: {captured.err!r}")
        assert "45" in captured.err and "60" in captured.err

    # Every rung is at or above MIN_INNER_TIMEOUT_S + INNER_SHUTDOWN_MARGIN_S,
    # so none of them is clamped and the strict invariant must hold on all of
    # them. 90 is the exact boundary; 1200 is the value that ties today.
    LADDER = [90, 91, 120, 300, 600, 900, 1200, 1800, 2400, 3600, 7200]

    def test_the_margin_invariant_holds_over_the_whole_ladder(self, provider):
        from utils.providers.agy import INNER_SHUTDOWN_MARGIN_S

        for configured in self.LADDER:
            rendered = self._print_timeout(
                provider.build_command(self.PROMPT, self.MODEL, timeout_s=configured)
            )
            inner = parse_go_duration(rendered)
            assert inner is not None, (
                f"{rendered!r} is not a Go duration (configured={configured})")
            assert inner > 0, f"non-positive inner timeout for {configured}"
            assert inner + INNER_SHUTDOWN_MARGIN_S <= configured, (
                f"configured={configured} rendered {rendered!r}: agy would reach "
                f"its own cutoff at or after the parent's deadline, which is the "
                f"tie the margin exists to break")

    def test_no_configured_value_still_renders_the_old_literal(self, provider):
        """The hardcoded `20m` no longer decides a live invocation."""
        for configured in (600, 1200, 2400):
            rendered = self._print_timeout(
                provider.build_command(self.PROMPT, self.MODEL, timeout_s=configured)
            )
            assert rendered != "20m", (
                f"configured={configured} still renders the hardcoded literal")

    def test_the_three_numbers_reconcile_to_one(self, provider):
        """§9.6.d: `providers.yaml` and the adapter default agree at 1200."""
        import yaml

        base = Path(__file__).parent.parent / "providers.yaml"
        config = yaml.safe_load(base.read_text(encoding="utf-8"))

        assert config["providers"]["agy"]["default_timeout"] == 1200
        assert provider.default_timeout == 1200
        # And the derived flag is that same single number, minus the margin.
        rendered = self._print_timeout(
            provider.build_command(self.PROMPT, self.MODEL, timeout_s=provider.default_timeout)
        )
        assert rendered == "1170s"

    def test_the_keyword_survives_the_effort_suffix(self, provider):
        """L11 #3's `--effort` and L5's derived timeout are independent."""
        cmd = provider.build_command(
            self.PROMPT, "Gemini 3.1 Pro (High):high", timeout_s=2400
        )

        assert self._print_timeout(cmd) == "2370s"
        assert cmd[cmd.index("--effort") + 1] == "high"
        assert cmd[cmd.index("--model") + 1] == "Gemini 3.1 Pro (High)"

    def test_the_keyword_is_keyword_only(self, provider):
        """§9.6.a: positional passing must not be possible (M7 orders after L5)."""
        with pytest.raises(TypeError):
            provider.build_command(self.PROMPT, self.MODEL, 1200)

    def test_the_transport_builder_threads_it(self, provider):
        """§9.6.b's Lane A leg: `build_command_for_transport` forwards the value."""
        argv, stdin_text = provider.build_command_for_transport(
            self.PROMPT, self.MODEL, "argv", timeout_s=2400
        )

        assert stdin_text is None
        assert self._print_timeout(argv) == "2370s"


MOCK_LLM_PATH = Path(__file__).parent / "mocks" / "mock_llm.py"


def _pid_is_alive(pid):
    """Whether ``pid`` still names a live process (POSIX)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.fixture
def agy_stub(tmp_path, monkeypatch):
    """A real `agy` on PATH, backed by ``mock_llm.py``'s print-timeout stubs."""
    import stat

    launcher = tmp_path / "agy"
    launcher.symlink_to(MOCK_LLM_PATH)
    MOCK_LLM_PATH.chmod(MOCK_LLM_PATH.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")
    monkeypatch.setenv("MOCK_LLM_PID_FILE", str(tmp_path / "sleeper.pid"))
    return launcher


def _await_pid_file(path, deadline_s=10.0):
    import time as _time

    end = _time.time() + deadline_s
    while _time.time() < end:
        try:
            text = Path(path).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            text = ""
        if text:
            return int(text)
        _time.sleep(0.05)
    raise AssertionError(f"the stub never recorded a pid at {path}")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pid liveness + symlink")
class TestNoChildOutlivesTheOuterDeadline:
    """§9.6.e's survivor check.

    Both stubs run against real processes with the margin scaled down, because
    a test cannot sleep for 20 minutes: the *mechanism* is what is under test,
    and `INNER_SHUTDOWN_MARGIN_S` / `MIN_INNER_TIMEOUT_S` are its parameters.
    """

    MODEL = "Gemini 3.1 Pro (High)"

    @pytest.fixture
    def scaled_margin(self, monkeypatch):
        from utils.providers import agy as agy_module

        monkeypatch.setattr(agy_module, "INNER_SHUTDOWN_MARGIN_S", 1)
        monkeypatch.setattr(agy_module, "MIN_INNER_TIMEOUT_S", 1)

    def test_a_stub_that_honours_the_flag_shuts_itself_down(
        self, agy_stub, scaled_margin, monkeypatch, tmp_path
    ):
        """The child exits on its own and the harness reads agy's own reason."""
        from utils.llm_client import ERROR_TIMEOUT, invoke_with_provider

        monkeypatch.setenv("MOCK_LLM_PRINT_TIMEOUT_BEHAVIOUR", "honour")

        with patch("shutil.which", return_value=str(agy_stub)):
            result = invoke_with_provider(
                prompt="Analyze this",
                model_spec=f"agy:{self.MODEL}",
                timeout=4,
            )

        assert result["success"] is False
        # agy's OWN failure reason, not a bare TIMEOUT the parent invented.
        assert result.get("error_code") != ERROR_TIMEOUT
        assert "print-timeout" in result["error"]
        # And nothing it started is still running.
        sleeper = _await_pid_file(tmp_path / "sleeper.pid")
        assert not _pid_is_alive(sleeper), (
            f"pid {sleeper} outlived the run despite the stub honouring its "
            f"own cutoff")

    # This was an `xfail(strict=False)`: the leader-only SIGKILL was the orphan
    # bug itself, and L5 cannot make a disobedient child reapable. The Lane A
    # spawn now snapshots the CLI's descendants before the kill and sweeps
    # every surviving process group (`process_control.terminate_timed_out`,
    # plan review-run-scope-salvage-and-failure-visibility D3.3), so the
    # sleeper is reaped and the test holds as an ordinary assertion.
    def test_a_stub_that_ignores_the_flag_leaves_no_descendant(
        self, agy_stub, scaled_margin, monkeypatch, tmp_path
    ):
        from utils.llm_client import ERROR_TIMEOUT, invoke_with_provider

        monkeypatch.setenv("MOCK_LLM_PRINT_TIMEOUT_BEHAVIOUR", "ignore")
        sleeper = None
        try:
            with patch("shutil.which", return_value=str(agy_stub)):
                result = invoke_with_provider(
                    prompt="Analyze this",
                    model_spec=f"agy:{self.MODEL}",
                    timeout=3,
                )

            assert result.get("error_code") == ERROR_TIMEOUT
            sleeper = _await_pid_file(tmp_path / "sleeper.pid")
            assert not _pid_is_alive(sleeper), (
                f"pid {sleeper} outlived the outer deadline — the leader-only "
                f"kill orphaned it")
        finally:
            if sleeper and _pid_is_alive(sleeper):
                try:
                    os.kill(sleeper, 9)
                except OSError:
                    pass
