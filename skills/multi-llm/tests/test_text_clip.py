"""``utils.text_clip.clip``: the one cut for a model's words in a bounded line.

Fix round 22 (live run 9, #4): ``status`` cut a declined fix's reason to
"… would reverse that dec)" and, on OpenCode, "… (catch-up may reach 60 m/s), )";
the notice cut T002's finding to "… but scripts/c)" and a Codex host relayed
"truncated at 'but scripts/c'". The rows below are those texts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.text_clip import ELLIPSIS, clip, plain_line, unclosed  # noqa: E402

CODEX_REASON = (
    "The implementation record deliberately allows the catch-up boost to exceed "
    "MAX_SPEED, so this fix would reverse that decision"
)
OPENCODE_REASON = (
    "Implementation record T001 and the plan's Decisions deliberately leave "
    "nextSpeed unclamped (catch-up may reach 60 m/s), and test 'a trailing "
    "opponent may exceed MAX_SPEED' pins it; adding Math.min(..., MAX_SPEED) "
    "would break it"
)
T002_FINDING = (
    "node scripts/check.mjs exited 1: check failed: retries must be 3 (found 2); "
    "src/config.json has retries 2, but scripts/check.mjs requires 3."
)
CORPUS = [CODEX_REASON, OPENCODE_REASON, T002_FINDING]


def _words(text: str) -> set:
    return set(text.split())


class TestTheCut:
    def test_text_that_fits_is_unchanged(self):
        assert clip(T002_FINDING, len(T002_FINDING)) == T002_FINDING
        assert clip("  two   words ", 20) == "two words"

    @pytest.mark.parametrize("text", CORPUS)
    @pytest.mark.parametrize("limit", [40, 60, 80, 100, 120, 160])
    def test_a_cut_keeps_whole_words_and_says_so(self, text, limit):
        if len(text) <= limit:
            assert clip(text, limit) == text
            return
        shown = clip(text, limit)
        assert len(shown) <= limit
        assert ELLIPSIS in shown
        kept = shown.split(ELLIPSIS)[0]
        # Every kept word is a word of the text: none was cut in two.
        assert _words(kept) <= _words(text) | {w.rstrip(",;:.") for w in _words(text)}, shown

    def test_the_run_9_status_lines(self):
        assert clip(CODEX_REASON, 120) == (
            "The implementation record deliberately allows the catch-up boost "
            "to exceed MAX_SPEED, so this fix would reverse that…"
        )
        assert clip(T002_FINDING, 120).endswith("src/config.json has retries 2, but…")

    @pytest.mark.parametrize("limit", range(8, 160))
    def test_no_dangling_separator_and_brackets_pair_up(self, limit):
        for text in CORPUS:
            shown = clip(text, limit)
            if shown == text:
                continue
            kept = shown.split(ELLIPSIS)[0]
            assert kept and kept[-1] not in " ,;:—-/(", shown
            # The caller's own brackets around the preview still pair up.
            assert unclosed(f"({shown})") == [], shown

    def test_a_bracket_the_cut_left_open_is_closed_after_the_mark(self):
        text = "unclamped (catch-up may reach 60 m/s and more besides) here"
        assert clip(text, 32) == "unclamped (catch-up may reach…)"
        assert clip('run "npm test --watch" now', 16) == 'run "npm test…"'
        assert clip("a `git diff --stat` call", 14) == "a `git diff…`"

    def test_an_apostrophe_is_not_a_quote(self):
        assert clip("the plan's own decision is final here", 22) == "the plan's own…"
        assert clip("say 'never clamp it here' twice", 20) == "say 'never clamp…'"

    def test_one_long_token_is_cut_inside(self):
        path = "see /very/long/" + "x" * 80 + "/file.js now"
        shown = clip(path, 40)
        assert len(shown) == 40 and shown.endswith(ELLIPSIS)
        assert shown.startswith("see /very/long/xxx")

    @pytest.mark.parametrize("limit", [0, 1, 2, 3])
    def test_tiny_budgets(self, limit):
        assert len(clip(CODEX_REASON, limit)) <= limit

    def test_control_characters_never_survive(self):
        assert clip("a\x1b[2K b\tc", 50) == plain_line("a\x1b[2K b\tc") == "a[2K b c"
