"""Tests for utils/stream_bootstrap.py (task 4 stdout/stderr encoding)."""

import io
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.stream_bootstrap import bootstrap_streams


def _cp1252_stream() -> io.TextIOWrapper:
    """A text stream mimicking piped Windows stdout (locale codec cp1252)."""
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252")


class TestBootstrapStreams:
    """Tests for bootstrap_streams()."""

    def test_non_cp1252_text_prints_without_error(self, monkeypatch):
        """Printing em dash / smart quotes through reconfigured cp1252 pipes
        must not raise UnicodeEncodeError."""
        out = _cp1252_stream()
        err = _cp1252_stream()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)

        bootstrap_streams()

        text = "em dash — smart quotes “quoted” ‘also’"
        print(text)  # would raise UnicodeEncodeError under cp1252
        print(text, file=sys.stderr)

        assert sys.stdout.encoding.lower().replace("-", "") == "utf8"
        assert sys.stderr.encoding.lower().replace("-", "") == "utf8"

    def test_output_is_utf8_encoded(self, monkeypatch):
        """After bootstrap, written text reaches the buffer as UTF-8 bytes."""
        buffer = io.BytesIO()
        out = io.TextIOWrapper(buffer, encoding="cp1252")
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", _cp1252_stream())

        bootstrap_streams()
        print("caf\xe9 —", end="")
        sys.stdout.flush()

        assert buffer.getvalue() == "caf\xe9 —".encode("utf-8")

    def test_sets_line_buffering(self, monkeypatch):
        """Streams are line-buffered after bootstrap."""
        monkeypatch.setattr(sys, "stdout", _cp1252_stream())
        monkeypatch.setattr(sys, "stderr", _cp1252_stream())

        bootstrap_streams()

        assert sys.stdout.line_buffering is True
        assert sys.stderr.line_buffering is True

    def test_errors_replace_for_unencodable_output(self, monkeypatch):
        """errors='replace' is set so even surrogates cannot crash printing."""
        monkeypatch.setattr(sys, "stdout", _cp1252_stream())
        monkeypatch.setattr(sys, "stderr", _cp1252_stream())

        bootstrap_streams()

        assert sys.stdout.errors == "replace"
        assert sys.stderr.errors == "replace"
        # Lone surrogate (the surrogateescape failure mode) must not raise.
        print("bad \udce9 byte")

    def test_non_reconfigurable_streams_are_tolerated(self, monkeypatch):
        """Streams without reconfigure (StringIO, test doubles) are skipped."""
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        monkeypatch.setattr(sys, "stderr", io.StringIO())

        bootstrap_streams()  # must not raise

        print("still works")
        assert "still works" in sys.stdout.getvalue()
