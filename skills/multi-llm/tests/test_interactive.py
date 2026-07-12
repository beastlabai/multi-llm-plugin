"""Tests for interactive utilities."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.interactive import (
    is_tty,
    _numbered_prompt,
    _try_gum_choose,
    _try_fzf_multi,
    select_models_interactive,
    select_multi,
    UNAVAILABLE,
)


class TestIsTty:
    """Tests for is_tty function."""

    def test_returns_bool(self):
        """Test that is_tty returns a boolean."""
        result = is_tty()
        assert isinstance(result, bool)


class TestNumberedPrompt:
    """Tests for _numbered_prompt function."""

    def test_parses_single_selection(self, monkeypatch):
        """Test numbered prompt parses single selection."""
        options = ["model-a", "model-b", "model-c"]
        monkeypatch.setattr('builtins.input', lambda _: "2")

        result = _numbered_prompt(options, "Test prompt")
        assert result == ["model-b"]

    def test_parses_multiple_selections(self, monkeypatch):
        """Test numbered prompt parses multiple space-separated selections."""
        options = ["model-a", "model-b", "model-c"]
        monkeypatch.setattr('builtins.input', lambda _: "1 3")

        result = _numbered_prompt(options, "Test prompt")
        assert result == ["model-a", "model-c"]

    def test_handles_invalid_numbers(self, monkeypatch):
        """Test numbered prompt ignores invalid numbers."""
        options = ["model-a", "model-b"]
        monkeypatch.setattr('builtins.input', lambda _: "1 99 0 abc 2")

        result = _numbered_prompt(options, "Test prompt")
        assert result == ["model-a", "model-b"]

    def test_handles_empty_input(self, monkeypatch):
        """Test numbered prompt returns empty list for empty input."""
        options = ["model-a", "model-b"]
        monkeypatch.setattr('builtins.input', lambda _: "")

        result = _numbered_prompt(options, "Test prompt")
        assert result == []

    def test_handles_eof_error(self, monkeypatch):
        """Test numbered prompt returns empty list on EOFError."""
        options = ["model-a", "model-b"]

        def raise_eof(_):
            raise EOFError()

        monkeypatch.setattr('builtins.input', raise_eof)

        result = _numbered_prompt(options, "Test prompt")
        assert result == []

    def test_handles_keyboard_interrupt(self, monkeypatch):
        """Test numbered prompt returns empty list on KeyboardInterrupt."""
        options = ["model-a", "model-b"]

        def raise_interrupt(_):
            raise KeyboardInterrupt()

        monkeypatch.setattr('builtins.input', raise_interrupt)

        result = _numbered_prompt(options, "Test prompt")
        assert result == []


class TestTryGumChoose:
    """Tests for _try_gum_choose function."""

    def test_returns_unavailable_when_gum_not_installed(self, monkeypatch):
        """Test returns UNAVAILABLE when gum is not installed (cascade falls through)."""
        monkeypatch.setattr('shutil.which', lambda x: None)

        result = _try_gum_choose(["a", "b"], "prompt")
        assert result is UNAVAILABLE

    @patch('subprocess.run')
    @patch('shutil.which')
    def test_returns_selections_on_success(self, mock_which, mock_run):
        """Test returns selected items on success."""
        mock_which.return_value = "/usr/bin/gum"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="model-a\nmodel-c\n"
        )

        result = _try_gum_choose(["model-a", "model-b", "model-c"], "prompt")
        assert result == ["model-a", "model-c"]

    @patch('subprocess.run')
    @patch('shutil.which')
    def test_returns_empty_list_on_empty_selection(self, mock_which, mock_run):
        """Test returns [] (NOT UNAVAILABLE) when gum ran but nothing was selected.

        An empty selection is a deliberate cancel/Esc, not an unavailable backend,
        so it must be a concrete empty list that stops the cascade.
        """
        mock_which.return_value = "/usr/bin/gum"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=""
        )

        result = _try_gum_choose(["model-a", "model-b"], "prompt")
        assert result == []
        assert result is not UNAVAILABLE

    @patch('subprocess.run')
    @patch('shutil.which')
    def test_returns_empty_list_on_nonzero_exit(self, mock_which, mock_run):
        """Test returns [] when gum exits non-zero (Esc cancel), not UNAVAILABLE."""
        mock_which.return_value = "/usr/bin/gum"
        mock_run.return_value = MagicMock(
            returncode=130,
            stdout=""
        )

        result = _try_gum_choose(["model-a", "model-b"], "prompt")
        assert result == []
        assert result is not UNAVAILABLE

    @patch('subprocess.run')
    @patch('shutil.which')
    def test_returns_unavailable_on_subprocess_failure(self, mock_which, mock_run):
        """Test returns UNAVAILABLE when the gum subprocess fails to launch."""
        mock_which.return_value = "/usr/bin/gum"
        mock_run.side_effect = OSError("boom")

        result = _try_gum_choose(["model-a", "model-b"], "prompt")
        assert result is UNAVAILABLE


class TestTryFzfMulti:
    """Tests for _try_fzf_multi function."""

    def test_returns_unavailable_when_fzf_not_installed(self, monkeypatch):
        """Test returns UNAVAILABLE when fzf is not installed (cascade falls through)."""
        monkeypatch.setattr('shutil.which', lambda x: None)

        result = _try_fzf_multi(["a", "b"], "prompt")
        assert result is UNAVAILABLE

    @patch('subprocess.run')
    @patch('shutil.which')
    def test_returns_selections_on_success(self, mock_which, mock_run):
        """Test returns selected items on success."""
        mock_which.return_value = "/usr/bin/fzf"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="model-b\n"
        )

        result = _try_fzf_multi(["model-a", "model-b", "model-c"], "prompt")
        assert result == ["model-b"]

    @patch('subprocess.run')
    @patch('shutil.which')
    def test_returns_empty_list_on_cancel(self, mock_which, mock_run):
        """Test returns [] (NOT UNAVAILABLE) when fzf ran but the user cancelled."""
        mock_which.return_value = "/usr/bin/fzf"
        mock_run.return_value = MagicMock(
            returncode=130,
            stdout=""
        )

        result = _try_fzf_multi(["model-a", "model-b"], "prompt")
        assert result == []
        assert result is not UNAVAILABLE


class TestSelectModelsInteractive:
    """Tests for select_models_interactive function."""

    def test_raises_when_no_tty(self, monkeypatch):
        """Test raises RuntimeError when not in a TTY."""
        monkeypatch.setattr('sys.stdin.isatty', lambda: False)
        monkeypatch.setattr('sys.stdout.isatty', lambda: True)

        with pytest.raises(RuntimeError) as exc_info:
            select_models_interactive(["model-a"])

        assert "No TTY available" in str(exc_info.value)

    def test_falls_back_to_numbered_prompt(self, monkeypatch):
        """Test falls back to numbered prompt when gum and fzf unavailable."""
        monkeypatch.setattr('sys.stdin.isatty', lambda: True)
        monkeypatch.setattr('sys.stdout.isatty', lambda: True)
        monkeypatch.setattr('shutil.which', lambda x: None)  # No gum or fzf
        monkeypatch.setattr('builtins.input', lambda _: "1")

        result = select_models_interactive(["model-a", "model-b"])
        assert result == ["model-a"]


class TestSelectMultiCascade:
    """select_multi cascade: UNAVAILABLE falls through; a cancel does NOT."""

    def test_unavailable_backend_falls_through_to_next(self, monkeypatch):
        """gum UNAVAILABLE → cascade tries fzf, whose selection is returned."""
        monkeypatch.setattr('utils.interactive.is_tty', lambda: True)
        monkeypatch.setattr(
            'utils.interactive._try_gum_choose', lambda opts, prompt: UNAVAILABLE
        )
        monkeypatch.setattr(
            'utils.interactive._try_fzf_multi', lambda opts, prompt: ["model-b"]
        )

        called = {"numbered": False}

        def _numbered(opts, prompt):
            called["numbered"] = True
            return ["should-not-happen"]

        monkeypatch.setattr('utils.interactive._numbered_prompt', _numbered)

        result = select_multi(["model-a", "model-b"], "prompt")
        assert result == ["model-b"]
        assert called["numbered"] is False

    def test_gum_cancel_does_not_cascade_to_fzf_or_numbered(self, monkeypatch):
        """A cancelled gum (ran → []) stops the cascade and returns [].

        This is the regression guard for the reported bug: an Esc / empty gum
        selection must NOT fall through and re-prompt the user via fzf or the
        numbered fallback.
        """
        monkeypatch.setattr('utils.interactive.is_tty', lambda: True)
        monkeypatch.setattr(
            'utils.interactive._try_gum_choose', lambda opts, prompt: []
        )

        calls = {"fzf": False, "numbered": False}

        def _fzf(opts, prompt):
            calls["fzf"] = True
            return ["leaked-from-fzf"]

        def _numbered(opts, prompt):
            calls["numbered"] = True
            return ["leaked-from-numbered"]

        monkeypatch.setattr('utils.interactive._try_fzf_multi', _fzf)
        monkeypatch.setattr('utils.interactive._numbered_prompt', _numbered)

        result = select_multi(["model-a", "model-b"], "prompt")
        assert result == []
        assert calls["fzf"] is False
        assert calls["numbered"] is False

    def test_fzf_cancel_does_not_cascade_to_numbered(self, monkeypatch):
        """gum UNAVAILABLE, fzf ran-but-cancelled → return []; numbered not used."""
        monkeypatch.setattr('utils.interactive.is_tty', lambda: True)
        monkeypatch.setattr(
            'utils.interactive._try_gum_choose', lambda opts, prompt: UNAVAILABLE
        )
        monkeypatch.setattr(
            'utils.interactive._try_fzf_multi', lambda opts, prompt: []
        )

        called = {"numbered": False}

        def _numbered(opts, prompt):
            called["numbered"] = True
            return ["leaked"]

        monkeypatch.setattr('utils.interactive._numbered_prompt', _numbered)

        result = select_multi(["model-a", "model-b"], "prompt")
        assert result == []
        assert called["numbered"] is False

    def test_both_unavailable_falls_back_to_numbered(self, monkeypatch):
        """gum and fzf both UNAVAILABLE → numbered fallback is used."""
        monkeypatch.setattr('utils.interactive.is_tty', lambda: True)
        monkeypatch.setattr(
            'utils.interactive._try_gum_choose', lambda opts, prompt: UNAVAILABLE
        )
        monkeypatch.setattr(
            'utils.interactive._try_fzf_multi', lambda opts, prompt: UNAVAILABLE
        )
        monkeypatch.setattr(
            'utils.interactive._numbered_prompt', lambda opts, prompt: ["model-a"]
        )

        result = select_multi(["model-a", "model-b"], "prompt")
        assert result == ["model-a"]


class TestSubprocessDecoding:
    """UTF-8 + errors='replace' decoding for gum/fzf output (task 3 tier 2)."""

    def _capture_run_kwargs(self, monkeypatch, captured):
        def fake_run(cmd, **kwargs):
            captured.update(kwargs)
            mock = MagicMock()
            mock.returncode = 0
            mock.stdout = "model-a\n"
            return mock

        monkeypatch.setattr('utils.interactive.subprocess.run', fake_run)

    def test_gum_choose_requests_utf8_replace(self, monkeypatch):
        """_try_gum_choose decodes subprocess output as UTF-8 with replace."""
        monkeypatch.setattr(
            'utils.interactive.shutil.which', lambda name: '/usr/bin/gum'
        )
        captured = {}
        self._capture_run_kwargs(monkeypatch, captured)

        result = _try_gum_choose(["model-a", "model-b"], "prompt")

        assert result == ["model-a"]
        assert captured["encoding"] == "utf-8"
        assert captured["errors"] == "replace"

    def test_fzf_multi_requests_utf8_replace(self, monkeypatch):
        """_try_fzf_multi decodes subprocess output as UTF-8 with replace."""
        monkeypatch.setattr(
            'utils.interactive.shutil.which', lambda name: '/usr/bin/fzf'
        )
        captured = {}
        self._capture_run_kwargs(monkeypatch, captured)

        result = _try_fzf_multi(["model-a", "model-b"], "prompt")

        assert result == ["model-a"]
        assert captured["encoding"] == "utf-8"
        assert captured["errors"] == "replace"

    def test_gum_choose_decodes_invalid_utf8_with_replacement(self, monkeypatch):
        """A real gum stand-in emitting invalid UTF-8 decodes with U+FFFD.

        Cross-platform (no shebang/chmod/PATH tricks, so it also runs on
        Windows): swaps the gum argv for a Python byte emitter while passing
        _try_gum_choose's own kwargs through to the real subprocess.run, so
        the actual decode path is exercised: no UnicodeDecodeError,
        replacement characters in the selection result.
        """
        import subprocess

        monkeypatch.setattr(
            'utils.interactive.shutil.which', lambda name: '/usr/bin/gum'
        )
        real_run = subprocess.run
        captured_kwargs = {}
        # \xe9 (0xE9) is cp1252 'e-acute'; invalid as a standalone UTF-8 byte.
        emitter = "import sys; sys.stdout.buffer.write(b'caf\\xe9-model\\n')"

        def run_python_emitter(cmd, **kwargs):
            assert cmd[0] == "gum"
            captured_kwargs.update(kwargs)
            return real_run([sys.executable, "-c", emitter], **kwargs)

        monkeypatch.setattr(
            'utils.interactive.subprocess.run', run_python_emitter
        )

        result = _try_gum_choose(["caf\xe9-model"], "prompt")

        assert result == ["caf�-model"]
        # The lossless decode must come from _try_gum_choose's own kwargs,
        # not the locale default.
        assert captured_kwargs["encoding"] == "utf-8"
        assert captured_kwargs["errors"] == "replace"
