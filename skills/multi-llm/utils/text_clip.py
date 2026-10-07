"""One rule for cutting text a model or a user wrote down to a line's budget.

Fix round 22 (live run 9, #4): every surface that showed a model's words in
a bounded line — ``status``'s progress line naming a declined fix, the
"check findings for review-code" line and the result's ``notice``, the stop
``error``, the batch log lines and the native leg's printed records — cut
them with a bare ``text[:n]``. That stopped mid-word with no mark ("… would
reverse that dec)", "… but scripts/c)"), left a dangling ", )" when the cut
fell after a comma inside the caller's own brackets, and a Codex host relayed
"truncated at 'but scripts/c'" as if that were the text. :func:`clip` is the
one cut: at a word boundary, marked with :data:`ELLIPSIS`, never ending on a
separator, and with every bracket or quote the kept part opened closed again
after the mark, so the caller's own ``(…)`` around it still pairs up.

Pure and import-free (``status`` stays sub-second); :func:`plain_line` moved
here from ``cli_common``, which re-exports it, so the result line's ``error``
and every clipped preview drop the same control characters.
"""

from __future__ import annotations

import re
from typing import Any, List, Tuple

#: The mark a cut text ends with (before any bracket it closes again).
ELLIPSIS = "…"

#: What :func:`plain_line` removes after collapsing whitespace: the C0
#: controls, DEL, the C1 controls and the bidirectional overrides and
#: isolates. A terminal acts on any of them — an ESC sequence can erase or
#: rewrite the line it sits on — and none belongs in a line a host shows.
_CONTROL_CHARACTERS = re.compile(
    "[\x00-\x1f\x7f-\x9f‎‏‪-‮⁦-⁩]"
)

#: Characters a cut never ends on before its mark: a separator or a closing
#: sentence mark left there reads as the start of something the cut dropped
#: (", …", "; …", " — …", "2.…").
DANGLING = " \t,;:.!?/\\|&+=*-–—·"

#: Bracket pairs :func:`clip` closes again after its mark.
_PAIRS = {"(": ")", "[": "]", "{": "}", "“": "”"}
_CLOSING = {closer: opener for opener, closer in _PAIRS.items()}

#: Marks that open and close with the same character.
_TOGGLES = ('"', "`")

#: A cut backs off to the last word boundary only while that keeps at least
#: two thirds of the room (``3 * kept >= 2 * room``, integers, so the
#: OpenCode plugin's copy of this rule decides alike); past it the text is
#: one long token (a path, a URL), which is cut inside rather than dropped.


def plain_line(text: Any) -> str:
    """``text`` as one plain line: whitespace runs collapsed to one space and
    every control character removed (:data:`_CONTROL_CHARACTERS`).

    Fix round 17 (the Fable review of round 16): the stop headline carries a
    task title from tasks.md and the runner's own summary, and ``set_error``
    only collapsed whitespace, so an ANSI sequence in either reached the
    OpenCode tool line, the codex tool result and ``status`` intact."""
    collapsed = " ".join(str(text).split()) if text else ""
    return " ".join(_CONTROL_CHARACTERS.sub("", collapsed).split())


def _closes_quote(text: str, index: int) -> bool:
    """Whether the ``'`` at *index* can close a single quote (not a letter's
    apostrophe, as in "plan's"): something before it and no letter after."""
    after = text[index + 1] if index + 1 < len(text) else " "
    before = text[index - 1] if index else " "
    return not before.isspace() and not after.isalnum()


def _opens_quote(text: str, index: int) -> bool:
    """Whether the ``'`` at *index* opens a single quote: no letter before it
    and something after it."""
    after = text[index + 1] if index + 1 < len(text) else " "
    before = text[index - 1] if index else " "
    return not before.isalnum() and not after.isspace()


def unclosed(text: str) -> List[Tuple[int, str]]:
    """The brackets and quotes *text* opens and never closes, outermost first,
    each as ``(index, mark)``.

    Brackets pair by kind, a closer with no opener is ignored, a closer whose
    opener sits under other open marks closes those too (they were left open
    inside it). ``"`` and backticks toggle. A single quote counts only where
    it cannot be a letter's apostrophe ("plan's", "models'" are not quotes).
    """
    stack: List[Tuple[int, str]] = []

    def close(mark: str) -> bool:
        for depth in range(len(stack) - 1, -1, -1):
            if stack[depth][1] == mark:
                del stack[depth:]
                return True
        return False

    for index, char in enumerate(text):
        if char in _PAIRS:
            stack.append((index, char))
        elif char in _CLOSING:
            close(_CLOSING[char])
        elif char in _TOGGLES:
            if not close(char):
                stack.append((index, char))
        elif char == "'":
            open_here = any(mark == "'" for _, mark in stack)
            if open_here and _closes_quote(text, index):
                close("'")
            elif _opens_quote(text, index):
                stack.append((index, char))
    return stack


def _closers(stack: List[Tuple[int, str]]) -> str:
    return "".join(_PAIRS.get(mark, mark) for _, mark in reversed(stack))


def _cut_at_word(text: str, room: int) -> str:
    """The first *room* characters of *text*, backed off to the last word
    boundary unless that drops more than a third (one long token)."""
    head = text[:room]
    if text[room].isspace():
        return head
    for index in range(len(head) - 1, -1, -1):
        if head[index].isspace():
            return head[:index] if 3 * index >= 2 * room else head
    return head


def _trimmed(kept: str) -> Tuple[str, List[Tuple[int, str]]]:
    """*kept* without a dangling separator or a mark it opens last with
    nothing after it, and the marks still open in what is left."""
    while True:
        kept = kept.rstrip(DANGLING)
        stack = unclosed(kept)
        if stack and stack[-1][0] == len(kept) - 1:
            kept = kept[:-1]
            continue
        return kept, stack


def clip(text: Any, limit: int, *, flatten: bool = True) -> str:
    """*text* within *limit* characters, cut at a word boundary.

    Text that fits comes back unchanged (as a :func:`plain_line` under
    *flatten*, the default). A longer one keeps as many whole words as fit,
    drops a trailing separator (``,``, ``;``, ``:``, ``—``, a full stop…),
    ends in :data:`ELLIPSIS` and then closes every bracket or quote the kept
    part left open, so ``f"({clip(reason, 120)})"`` never reads ``(…, )``
    and never leaves a ``(`` or ``"`` unpaired. A single token longer than
    two thirds of the room (a path, a URL) is cut inside. The result is never
    longer than *limit*.
    """
    flat = plain_line(text) if flatten else ("" if text is None else str(text))
    if limit is None or len(flat) <= limit:
        return flat
    if limit <= 0:
        return ""
    room = limit - len(ELLIPSIS)
    while room > 0:
        kept, stack = _trimmed(_cut_at_word(flat, room))
        if not kept:
            break
        clipped = kept + ELLIPSIS + _closers(stack)
        if len(clipped) <= limit:
            return clipped
        room -= len(clipped) - limit
    return ELLIPSIS[:limit]


def was_clipped(text: Any, shown: str, *, flatten: bool = True) -> bool:
    """Whether *shown* (a :func:`clip` of *text*) is shorter than *text*."""
    flat = plain_line(text) if flatten else ("" if text is None else str(text))
    return shown != flat
