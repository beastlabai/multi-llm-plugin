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

    def test_returns_none_when_gum_not_installed(self, monkeypatch):
        """Test returns None when gum is not installed."""
        monkeypatch.setattr('shutil.which', lambda x: None)

        result = _try_gum_choose(["a", "b"], "prompt")
        assert result is None

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
    def test_returns_none_on_empty_selection(self, mock_which, mock_run):
        """Test returns None when no items selected."""
        mock_which.return_value = "/usr/bin/gum"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=""
        )

        result = _try_gum_choose(["model-a", "model-b"], "prompt")
        assert result is None


class TestTryFzfMulti:
    """Tests for _try_fzf_multi function."""

    def test_returns_none_when_fzf_not_installed(self, monkeypatch):
        """Test returns None when fzf is not installed."""
        monkeypatch.setattr('shutil.which', lambda x: None)

        result = _try_fzf_multi(["a", "b"], "prompt")
        assert result is None

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
