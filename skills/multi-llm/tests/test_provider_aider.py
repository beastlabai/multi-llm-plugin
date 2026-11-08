"""Tests for the Aider adapter.

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

from utils.providers.aider import AiderProvider
from utils.retry_ladder import STATUS_AUTH_FAILED, STATUS_RATE_LIMITED


class TestAiderProvider:
    """Tests for the AiderProvider class."""

    @pytest.fixture
    def provider(self):
        """Create an AiderProvider instance."""
        return AiderProvider()

    BANNER = (
        "\nAider v0.86.2\n"
        "Model: openrouter/z-ai/glm-5.2 with whole edit format\n"
        "Git repo: .git with 12 files\n"
        "Repo-map: using 4096 tokens, auto refresh\n"
    )

    @classmethod
    def _stdout(cls, answer, thinking=None, tokens_line=True):
        """Build representative aider --no-pretty stdout."""
        parts = [cls.BANNER, "\n"]
        if thinking is not None:
            parts.append(f"--------------\n► **THINKING**\n\n{thinking}\n\n")
        parts.append(f"------------\n► **ANSWER**\n\n{answer}\n")
        if tokens_line:
            parts.append(
                "\nTokens: 210 sent, 703 received. "
                "Cost: $0.0023 message, $0.0023 session.\n"
            )
        return "".join(parts)

    def test_name_property(self, provider):
        """Test that name property returns correct identifier."""
        assert provider.name == "aider"

    def test_default_timeout(self, provider):
        """Test that default_timeout is set correctly."""
        assert provider.default_timeout == 600

    def test_aider_parse_answer_array(self, provider):
        """Extract a JSON array from the text after the ANSWER marker."""
        inner_data = [
            {"title": "Finding 1", "desc": "Description", "importance": "medium"}
        ]
        stdout = self._stdout(json.dumps(inner_data))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_aider_parse_answer_object(self, provider):
        """Extract a JSON object from the text after the ANSWER marker."""
        inner_data = {"first_heading": "Probe Widget"}
        stdout = self._stdout(json.dumps(inner_data))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_aider_parse_thinking_decoy_ignored(self, provider):
        """Decoy JSON in the THINKING block is ignored; post-ANSWER wins."""
        stdout = self._stdout(
            "[4, 5, 6]",
            thinking='Draft of the reply:\n[1, 2, 3]\nAlso {"wrong": 1} here.',
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [4, 5, 6]

    def test_aider_parse_multiple_answer_markers_last_wins(self, provider):
        """When several ANSWER markers appear, only the LAST one is parsed."""
        first = "------------\n► **ANSWER**\n\n[1, 2, 3]\n\n"
        stdout = self.BANNER + first + self._stdout("[7, 8, 9]")

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [7, 8, 9]

    def test_aider_parse_strips_tokens_cost_line(self, provider):
        """The trailing "Tokens: ... Cost: ..." line does not break parsing."""
        stdout = self._stdout("[1, 2, 3]", tokens_line=True)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [1, 2, 3]

    def test_aider_parse_prose_with_json(self, provider):
        """Extract embedded JSON when prose surrounds it in the answer."""
        inner_json = {"result": "success"}
        stdout = self._stdout(
            f"Here is the JSON you asked for: {json.dumps(inner_json)}"
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_aider_parse_code_block_json(self, provider):
        """Extract JSON from a ```json code block in the answer."""
        inner_json = [{"task": "T001", "status": "done"}]
        stdout = self._stdout(
            f"Analysis complete:\n\n```json\n{json.dumps(inner_json)}\n```"
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_aider_parse_file_write_envelope_unwrapped(self, provider):
        """A pseudo file-write envelope ({"file_path", "content"}) is unwrapped.

        In read-only /ask mode some models emulate the requested file write
        by answering with the target path and the JSON as a string content
        field (observed live with kimi-k2.7-code).
        """
        inner = [{"title": "Finding", "desc": "D", "importance": "high"}]
        envelope = {"file_path": "/some/out.json", "content": json.dumps(inner)}
        stdout = self._stdout(json.dumps(envelope))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner

    def test_aider_parse_code_interpreter_envelope_unwrapped(self, provider):
        """A hallucinated code_interpreter call ({"code": ...}) is unwrapped."""
        inner = [{"title": "Finding", "desc": "D", "importance": "high"}]
        code = (
            'import json\nissues = ' + json.dumps(inner, indent=2)
            + '\nwith open("/out.json", "w") as f:\n    json.dump(issues, f)\n'
        )
        stdout = self._stdout(json.dumps({"code": code}))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner

    def test_aider_parse_ordinary_object_not_unwrapped(self, provider):
        """A dict with extra keys (a real answer) is NOT treated as envelope."""
        data = {"content": "[1]", "title": "x", "extra": True}
        stdout = self._stdout(json.dumps(data))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == data

    def test_aider_parse_wrapped_json_mid_string_keeps_the_break(self, provider):
        """Hard-wrapped JSON (literal newline INSIDE a string) parses, break kept.

        This documents why get_env() sets COLUMNS=10000: without it aider
        wraps output at ~80 columns and long string values get literal
        newlines inserted. Until 2026-10-05 the strict parse rejected the
        whole document; ``extract_json_from_text`` now tolerates raw control
        characters inside strings (``_loads_reply``), so the answer survives,
        but the inserted break is kept verbatim in the value — the wrap still
        alters what the model wrote, which is why the variable stays.
        """
        wrapped = (
            '[\n  {\n    "title": "Missing Input Validation",\n'
            '    "desc": "The function responsible for parsing user-supplied \n'
            'configuration parameters lacks proper validation checks."\n  }\n]'
        )
        stdout = self._stdout(wrapped)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"][0]["desc"] == (
            "The function responsible for parsing user-supplied \n"
            "configuration parameters lacks proper validation checks."
        )

    def test_aider_parse_wrapped_between_tokens_ok(self, provider):
        """Newlines between JSON tokens (not inside strings) parse fine."""
        stdout = self._stdout('[\n  1,\n  2,\n  3\n]')

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [1, 2, 3]

    def test_aider_parse_no_answer_marker_fallback(self, provider):
        """No ANSWER marker: fall back to extraction on the full stdout."""
        stdout = self.BANNER + "\nSome text [1, 2, 3] trailing text\n"

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [1, 2, 3]

    def test_aider_parse_auth_error_stdout(self, provider):
        """aider exits 0 on auth failure; no extractable JSON -> failure."""
        stdout = (
            self.BANNER
            + "\nlitellm.AuthenticationError: AuthenticationError: "
            "OpenrouterException - Invalid credentials. Check your API key.\n"
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "litellm.AuthenticationError" in result["error"]
        assert result["data"] is None

    def test_aider_parse_empty_answer_after_marker(self, provider):
        """An empty answer after the ANSWER marker is an explicit failure."""
        stdout = self._stdout("", tokens_line=True)

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result

    def test_aider_parse_empty_stdout(self, provider):
        """Empty stdout is an explicit failure."""
        result = provider.parse_output("", "")

        assert result["success"] is False
        assert "error" in result
        assert result["data"] is None

    def test_aider_build_command(self, provider):
        """Exact argv: model verbatim, /ask prefix, all headless flags."""
        cmd = provider.build_command("Analyze this", "openrouter/z-ai/glm-5.2")

        assert cmd == [
            "aider",
            "--model",
            "openrouter/z-ai/glm-5.2",
            "--message",
            "/ask Analyze this",
            "--no-auto-commits",
            "--no-pretty",
            "--no-stream",
            "--no-check-update",
            "--no-show-model-warnings",
            "--no-analytics",
            "--no-gitignore",
            "--no-show-release-notes",
            "--no-detect-urls",
            "--no-fancy-input",
            "--chat-history-file",
            os.devnull,
            "--input-history-file",
            os.devnull,
        ]

    def test_aider_build_command_read_flags_for_existing_paths(self, provider, tmp_path):
        """Existing absolute file paths in the prompt are passed via --read."""
        plan = tmp_path / "my-plan.md"
        plan.write_text("# Plan", encoding="utf-8")
        missing = tmp_path / "review-plan" / "aider_model.json"  # not yet written
        prompt = (
            f"## Plan File\n{plan}\n\nRead this file.\n"
            f"Write your JSON output to this file: {missing}\n"
            f"Also mentioned twice: {plan}."
        )

        cmd = provider.build_command(prompt, "openrouter/z-ai/glm-5.2")

        assert cmd.count("--read") == 1  # deduped; missing path skipped
        assert cmd[cmd.index("--read") + 1] == str(plan)
        assert str(missing) not in cmd

    def test_aider_build_command_no_read_flags_without_paths(self, provider):
        """No --read args when the prompt mentions no existing files."""
        cmd = provider.build_command(
            "Analyze this plan about /nonexistent/path/plan.md", "some-model"
        )

        assert "--read" not in cmd

    def test_aider_build_command_model_verbatim(self, provider):
        """Model IDs are litellm specs passed verbatim — no splitting."""
        cmd = provider.build_command("Hi", "openrouter/moonshotai/kimi-k2.7-code")

        model_idx = cmd.index("--model") + 1
        assert cmd[model_idx] == "openrouter/moonshotai/kimi-k2.7-code"
        assert "--provider" not in cmd
        assert "-P" not in cmd

    def test_aider_get_env(self, provider):
        """get_env sets BROWSER (webbrowser no-op) and COLUMNS (no wrap)."""
        env = provider.get_env("openrouter/z-ai/glm-5.2")

        assert env == {"BROWSER": "true", "COLUMNS": "10000"}

    def test_aider_reasoning_efforts_constant(self):
        """Module-level REASONING_EFFORTS holds exactly the six valid efforts."""
        from utils.providers.aider import REASONING_EFFORTS

        assert isinstance(REASONING_EFFORTS, frozenset)
        assert REASONING_EFFORTS == {"none", "minimal", "low", "medium", "high", "xhigh"}

    def test_aider_build_command_effort_suffix_exact(self, provider):
        """Exact argv for an :effort suffix: --reasoning-effort pair after --model <base>."""
        cmd = provider.build_command("Analyze this", "openrouter/z-ai/glm-5.2:high")

        assert cmd == [
            "aider",
            "--model",
            "openrouter/z-ai/glm-5.2",
            "--reasoning-effort",
            "high",
            "--message",
            "/ask Analyze this",
            "--no-auto-commits",
            "--no-pretty",
            "--no-stream",
            "--no-check-update",
            "--no-show-model-warnings",
            "--no-analytics",
            "--no-gitignore",
            "--no-show-release-notes",
            "--no-detect-urls",
            "--no-fancy-input",
            "--chat-history-file",
            os.devnull,
            "--input-history-file",
            os.devnull,
        ]

    def test_aider_build_command_all_reasoning_efforts(self, provider):
        """Every valid effort suffix produces the --reasoning-effort form."""
        for effort in ("none", "minimal", "low", "medium", "high", "xhigh"):
            cmd = provider.build_command("Analyze this", f"openrouter/z-ai/glm-5.2:{effort}")

            assert cmd == [
                "aider",
                "--model",
                "openrouter/z-ai/glm-5.2",
                "--reasoning-effort",
                effort,
                "--message",
                "/ask Analyze this",
                "--no-auto-commits",
                "--no-pretty",
                "--no-stream",
                "--no-check-update",
                "--no-show-model-warnings",
                "--no-analytics",
                "--no-gitignore",
                "--no-show-release-notes",
                "--no-detect-urls",
                "--no-fancy-input",
                "--chat-history-file",
                os.devnull,
                "--input-history-file",
                os.devnull,
            ], f"effort {effort!r} did not build the expected command"

    def test_aider_build_command_effort_passthrough(self, provider):
        """Non-effort models pass through verbatim with no --reasoning-effort flag."""
        for model in (
            "openrouter/z-ai/glm-5.2",
            "openrouter/deepseek/deepseek-r1:free",
            "openrouter/z-ai/glm-5.2:turbo",
        ):
            cmd = provider.build_command("Analyze this", model)

            assert cmd == [
                "aider",
                "--model",
                model,
                "--message",
                "/ask Analyze this",
                "--no-auto-commits",
                "--no-pretty",
                "--no-stream",
                "--no-check-update",
                "--no-show-model-warnings",
                "--no-analytics",
                "--no-gitignore",
                "--no-show-release-notes",
                "--no-detect-urls",
                "--no-fancy-input",
                "--chat-history-file",
                os.devnull,
                "--input-history-file",
                os.devnull,
            ], f"model {model!r} was not passed through verbatim"

    @patch("shutil.which")
    def test_aider_is_available_true(self, mock_which, provider):
        """Test is_available returns True when aider is found."""
        mock_which.return_value = "/usr/bin/aider"

        assert provider.is_available() is True
        mock_which.assert_called_once_with("aider")

    @patch("shutil.which")
    def test_aider_is_available_false(self, mock_which, provider):
        """Test is_available returns False when aider is not found."""
        mock_which.return_value = None

        assert provider.is_available() is False
        mock_which.assert_called_once_with("aider")

    def test_aider_build_command_uses_os_devnull(self, provider):
        """History files are routed to os.devnull (portable null device)."""
        cmd = provider.build_command("Analyze this", "some-model")

        assert cmd[cmd.index("--chat-history-file") + 1] == os.devnull
        assert cmd[cmd.index("--input-history-file") + 1] == os.devnull

    def test_aider_abs_path_re_matches_full_backslash_windows_path(self):
        """The regex matches an entire backslash Windows path, not just C:\\."""
        from utils.providers.aider import _ABS_PATH_RE

        matches = _ABS_PATH_RE.findall(r"Read C:\Users\foo\bar.md for context.")

        assert matches == [r"C:\Users\foo\bar.md"]

    def test_aider_abs_path_re_matches_forward_slash_windows_path(self):
        """Drive-letter paths with forward slashes match in full too."""
        from utils.providers.aider import _ABS_PATH_RE

        matches = _ABS_PATH_RE.findall("Read C:/Users/foo/bar.md for context.")

        assert matches == ["C:/Users/foo/bar.md"]

    def test_aider_abs_path_re_posix_branch_unchanged(self):
        """POSIX paths match exactly as with the historical pattern."""
        from utils.providers.aider import _ABS_PATH_RE

        matches = _ABS_PATH_RE.findall(
            "See /home/user/plans/my-plan.md and /tmp/out.json"
        )

        assert matches == ["/home/user/plans/my-plan.md", "/tmp/out.json"]

    def test_aider_extract_read_files_full_windows_path(self, provider, monkeypatch):
        """_extract_read_files yields the whole Windows path, not a truncated C:\\."""
        win_path = r"C:\Users\foo\bar.md"
        monkeypatch.setattr(Path, "is_file", lambda self: str(self) == win_path)

        found = provider._extract_read_files(f"## Plan File\n{win_path}\n\nRead it.")

        assert found == [win_path]

    def test_aider_extract_read_files_posix_unchanged(self, provider, tmp_path):
        """Existing POSIX extraction behavior is unchanged by the Windows branch."""
        plan = tmp_path / "plan.md"
        plan.write_text("# Plan", encoding="utf-8")

        found = provider._extract_read_files(
            f"Read {plan} and the missing /nonexistent/file.md"
        )

        assert found == [str(plan)]


class TestAiderGlossIsNarrowedWhereItIsWritten:
    """C7: aider's own 5xx gloss stops classifying, the vendor's words do not.

    The two moved fixture rows are graded by `test_provider_fixture_rows.py`.
    What is only assertable here is the half no capture covers: that narrowing
    aider's gloss did **not** weaken `overloaded` for the vendor that genuinely
    sends it. There is no Anthropic-through-aider capture in the tree, so the
    529 case below is built from litellm's rendering of one rather than read off
    disk, and it is labelled as such.
    """

    FIXTURES = Path(__file__).parent / "fixtures" / "aider"

    #: An Anthropic 529 as litellm renders it into aider's stdout: the vendor's
    #: own error body on one line, aider's gloss on the next. Synthetic (no such
    #: capture is committed); every other assertion in this class reads bytes.
    ANTHROPIC_529 = (
        "\nAider v0.86.2\n"
        "Model: anthropic/claude-sonnet-4-5 with whole edit format\n\n"
        "litellm.InternalServerError: AnthropicException - "
        '{"type":"error","error":{"type":"overloaded_error",'
        '"message":"Overloaded"}}\n'
        "The API provider's servers are down or overloaded.\n"
        "Retrying in 0.2 seconds...\n"
    )

    def _capture(self, stem):
        return (self.FIXTURES / f"{stem}.stdout.txt").read_text(encoding="utf-8")

    def test_the_gloss_alone_no_longer_says_rate_limited(self):
        """The two rows the excision moves, from the committed bytes."""
        provider = AiderProvider()

        assert provider.classify_failure(self._capture("truncated"), "", 0) == \
            "connection_error"
        assert provider.classify_failure(
            self._capture("exit-zero-failure"), "", 0
        ) == "runner_error"

    def test_and_the_gloss_is_still_in_those_bytes(self):
        """The fix is an excision at classification time, not an edited capture."""
        for stem in ("truncated", "exit-zero-failure"):
            assert "The API provider's servers are down or overloaded." in \
                self._capture(stem)

    def test_a_vendor_that_really_says_overloaded_is_still_rate_limited(self):
        """The shared `overloaded` pattern is intact — this is C7's whole point.

        Anthropic's genuine 529 is what `overloaded` carries in
        `base.rate_limit_patterns`, and it survives the excision because the
        vendor writes it on its own line.
        """
        assert AiderProvider().classify_failure(self.ANTHROPIC_529, "", 0) == \
            STATUS_RATE_LIMITED

    def test_aiders_measured_429_is_untouched(self):
        """The committed 429 capture, which the excision must not reach."""
        assert AiderProvider().classify_failure(
            self._capture("rate-limit"), "", 0
        ) == STATUS_RATE_LIMITED

    def test_the_other_glosses_stay_because_they_are_the_only_evidence(self):
        """Why the excision names one sentence rather than aider's gloss shape.

        `not able to authenticate` — this adapter's own declared auth pattern —
        IS the 401 gloss, and no other auth wording appears in that capture. An
        excision keyed on "The API provider …" would turn a rung-1 auth failure
        into a retryable runner error.
        """
        assert AiderProvider().classify_failure(
            self._capture("auth-error"), "", 0
        ) == STATUS_AUTH_FAILED

    def test_the_excision_degrades_to_its_input(self):
        """A capture that is nothing but the gloss must not classify as None.

        An empty haystack is "nothing recognized", i.e. transport-shaped, so an
        excision that consumed a whole capture would launder a failure into a
        retry. The reduction returns its input instead.
        """
        gloss_only = "The API provider's servers are down or overloaded.\n"

        assert AiderProvider()._drop_adapter_gloss(gloss_only) == gloss_only

    def test_the_excision_only_ever_removes(self):
        """Removal-only, the same contract `vendor_text` states for the hook."""
        narrowed = AiderProvider()._drop_adapter_gloss(self._capture("truncated"))
        original = self._capture("truncated")

        assert len(narrowed) < len(original)
        for line in narrowed.splitlines():
            assert line in original
