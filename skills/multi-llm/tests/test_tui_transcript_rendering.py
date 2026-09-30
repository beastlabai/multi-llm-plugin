"""``entrypoint._plain`` renders a full-screen TUI repaint stream, offline (T1).

`test_container_documented_invocation.py`'s
`test_a_non_resolving_host_refused_it_visibly_and_sent_nothing[codex]` failed
against codex 0.159.2: the host_matrix.yaml cell's `refusal` pattern is
``Unrecognized command '/multi-llm'`` (real spaces), but the transcript the row
captured read ``Unrecognizedcommand'/multi-llm'.Type"/"foralistofsupportedcommands.``
— every word run together, so the regex never matched and the row was graded a
silent non-resolution instead of the refusal it actually was.

The cause is in `_plain`, not the regex. codex 0.159.2's TUI repaints by
positioning the cursor and overwriting only the cells that changed: each word
of the refusal line arrives as its own `text` run preceded by an absolute
`CSI row;colH`, and the one-column gap between two words is never a byte on the
wire — it is the *distance* the next `H` places the cursor, past wherever the
previous run's last character landed. The old `_plain` was `_ANSI.sub("", …)`:
delete every escape sequence with no memory of what it did. That deletes the
distance along with the sequence that encoded it, so `Unrecognized` and
`command` land adjacent with nothing between them. Loosening the refusal regex
to tolerate missing spaces would have hidden the same bug from every other
assertion that ever reads a transcript.

The fix renders onto a small grid instead (`_render_screen`): a cursor moves
for CR/LF/backspace and for the CSI moves `_apply_csi` interprets (`H`/`f`,
`A`/`B`/`C`/`D`, `G`, `K`, `J`), and each run of plain text is written at the
cursor's current cell, one column per character. Flattening the grid at the end
recovers the gap as real space characters, because the gap is exactly the
distance between where one run's write left the cursor and where the next
`CSI …H` moved it before its own run began.

This module is offline (no daemon, no credential, no network) and pins that
render directly against captured bytes, the way `test_host_model_pin.py` pins
`entrypoint`'s pure functions without a container. **The filename is
deliberately not `test_container_*`**, for the reason that module and
`test_stray_write_paths.py` both give: that glob is what
`tests/container/context_manifest.txt` ships into the image and what the
marker-gating check enumerates, and a plain-CI module answering to it would be
shipped for nothing and would fail that assertion.
"""

from __future__ import annotations

import re

from .container import entrypoint


def _render(raw: bytes) -> str:
    return entrypoint._plain(raw)


class TestWordsPositionedApartOnOneRowGetTheirSpaceBack:
    """The exact shape codex 0.159.2 sends: one `CSI row;colH` per word."""

    def test_the_refusal_line_renders_with_real_spaces(self):
        # A trimmed, hand-built replay of the byte pattern captured from a real
        # codex 0.159.2 container row: six words and the leading bullet, each
        # positioned by its own absolute CUP rather than joined by a literal
        # space. Column gaps match what was measured (one blank column between
        # each pair of words).
        raw = (
            b"\x1b[8;1H\x1b[2m\xe2\x80\xa2 \x1b[22mUnrecognized"
            b"\x1b[8;16Hcommand"
            b"\x1b[8;24H'/multi-llm'."
            b"\x1b[8;38HType"
            b"\x1b[8;43H\"/\""
            b"\x1b[8;47Hfor"
            b"\x1b[8;51Ha"
            b"\x1b[8;53Hlist"
            b"\x1b[8;58Hof"
            b"\x1b[8;61Hsupported"
            b"\x1b[8;71Hcommands."
        )
        text = _render(raw)
        assert re.search(r"Unrecognized command '/multi-llm'", text), text
        assert ("Unrecognized command '/multi-llm'. Type \"/\" for a list of "
                "supported commands." in text), text

    def test_a_bare_cursor_forward_move_also_becomes_a_gap(self):
        """`CSI n C` (relative), not just `CSI row;colH` (absolute).

        `one` ends at column 3 (0-indexed); `CSI 3C` advances the cursor three
        more columns to 6, and `two` is written there — a three-column gap,
        with nothing on the wire to say "space" three times over.
        """
        raw = b"\x1b[3;1Hone\x1b[3Ctwo"
        text = _render(raw)
        assert "one   two" in text, text


class TestContiguousTextIsUnaffected:
    """The case every prior version of `_plain` already handled correctly —
    opencode's ready banner (measured 2026-08-24) is a contiguous same-row
    write with real space characters and no repositioning between words, and
    the render must not disturb it."""

    def test_plain_contiguous_text_with_real_spaces_survives(self):
        raw = b"\x1b[5;3Hctrl+p commands \xc2\xb7 ? for shortcuts"
        text = _render(raw)
        assert "ctrl+p commands \xb7 ? for shortcuts" in text, text

    def test_ansi_colour_and_mode_codes_leave_no_residue(self):
        raw = (b"\x1b[1;1H\x1b[38;5;75;49mhello\x1b[39;49m\x1b[0m "
               b"\x1b[?25l\x1b[?2026hworld\x1b[?25h")
        text = _render(raw)
        assert "hello world" in text, text
        assert "\x1b" not in text
        assert "[38;5" not in text


class TestEraseAndOverwriteBehaveLikeARealTerminal:
    def test_erase_in_line_clears_stale_text_before_a_shorter_redraw(self):
        # First paint a long line, then erase-to-end-of-line at column 6 and
        # paint a shorter replacement — the trailing "world" of the first paint
        # must not survive into the final frame.
        raw = b"\x1b[1;1Hhello world\x1b[1;6H\x1b[K!!"
        text = _render(raw)
        assert "world" not in text
        assert "hello!!" in text.splitlines()[0]

    def test_a_later_frame_overwrites_an_earlier_one_at_the_same_cell(self):
        raw = b"\x1b[2;3Hloading\x1b[2;3Hready!!"
        text = _render(raw)
        assert "loading" not in text
        assert "ready!!" in text


class TestOtherCallersOfPlainStillWork:
    """`_plain` backs both the `ready_rx` check and the saved transcript file
    (`drive_documented_invocation`) — a regex anchored on real spaces must keep
    matching both hosts' ready banners after the rewrite."""

    def test_codex_ready_marker_still_matches(self):
        raw = b"\x1b[2;3H\x1b[38;5;75;49m>_ \x1b[1mOpenAI Codex\x1b[22m (v0.159.2)"
        text = _render(raw)
        assert re.search(r"OpenAI Codex \(v", text), text

    def test_opencode_ready_marker_still_matches(self):
        raw = b"\x1b[50;3Hctrl+p commands"
        text = _render(raw)
        assert re.search(r"ctrl\+p commands", text), text
