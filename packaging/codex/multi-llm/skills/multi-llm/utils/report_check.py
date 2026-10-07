#!/usr/bin/env python3
"""``util report-check``: does this turn's final message carry the next command?

The Claude Code plugin's ``Stop`` hook (``hooks/report-check.sh``) runs this
after every turn whose transcript mentions the block at all
(plans/next-phase-command.md T11). A phase result, or ``util next-command``,
prints the next step as a block of marker lines above its result line::

    [NEXT_COMMAND] <command>
    [NEXT_COMMAND_OPTIONAL_WHY] <why>
    [NEXT_COMMAND_OPTIONAL] <command>
    [NEXT_COMMAND_ANSWERS_WHY] <why>
    [NEXT_COMMAND_ANSWERS] <command>

(the answers pairs only from ``util next-command`` on a stopped implement,
fix round 14: the combined answer first when more than one task waits, then
each answer alone since fix round 20; the check holds the report to the
commands printed, in any order, so one waiting task means no combined
command to carry), or ``[WORKFLOW_COMPLETE]``. Either kind is followed by one
``[MANUAL_TASK] <id> <title>`` line per implement task the run deferred to the
user, while any is owed, and one ``[MANUAL_TASK_NOTE]`` line saying how to
record them done (after the commands of a command block since fix round 16),
then the ``[MANUAL_TASK_COMMAND]`` line that does it, a command the report
fences like the others (fix round 22).
The model is told to end its report with each command in a fenced block,
copied character for character, or with "Workflow complete.", and then those
tasks listed. This checks the copy, and only the copy:

* It reads the transcript entries since the last **real** user prompt: a user
  entry whose content is a string or carries a text block. A user entry that
  only carries ``tool_result`` blocks is part of the turn, and an ``isMeta``
  entry (a loaded skill body, a local-command caveat) or a compaction summary
  is not a prompt. ``isSidechain`` entries belong to a subagent, not the turn.
* Among that turn's tool results it takes the **last** block. A block printed
  only in an earlier turn is not this turn's to copy, so it is never checked.
* A marker counts only at the start of a line, which is where the emitter puts
  it. An indented example of the block (a plan or a doc quoting it, read by a
  tool) is not a phase result.
* It passes when every command of that block (the ``[NEXT_COMMAND]`` one,
  each ``[NEXT_COMMAND_OPTIONAL]`` one and each ``[NEXT_COMMAND_ANSWERS]`` one)
  is, byte for byte, a line inside a
  fenced code block of the final assistant message. Only the whitespace around
  a line is ignored, so a fence nested in a list item still counts. For
  ``[WORKFLOW_COMPLETE]`` it passes when the message says "Workflow complete."
  A block of either kind with ``[MANUAL_TASK]`` lines also needs each of their
  ids named as a word of its own, anywhere in the message (fix round 17: after
  a command block too, which SKILL.md and summary-format.md have asked for
  since fix round 16). Only the ids: a report that lists the tasks as the
  rule says always names them, while a check of the note's wording would
  block a report that put its command in backticks. A why line is never a
  command.
* The final message is the text of the turn's trailing assistant entries. The
  ``Stop`` hook input's ``last_assistant_message`` is read too, when present,
  and the turn passes when either carries the copy: a stale transcript must
  never produce a block. It is stale in practice. Observed on a live headless
  ``claude -p`` hook (2.1.289, 2026-10-05), the input carries ``session_id``,
  ``transcript_path``, ``cwd``, ``prompt_id``, ``permission_mode``,
  ``effort``, ``hook_event_name``, ``stop_hook_active``,
  ``last_assistant_message``, ``background_tasks`` and ``session_crons``, and
  the transcript still ends at the turn's last tool result: the reply is
  written after the hook returns. So the field is what gets checked, and a
  transcript with no reply text yet, with no field beside it, passes.
* ``last_assistant_message`` is the text of the **last** assistant message
  Claude Code holds in memory, not the whole reply (hooks review of fix round
  10, read from the harness, not measured live). A reply split into several
  text blocks hands the hook only the last one, so a command copied into an
  earlier block reads as missing. That costs at most one needless block: the
  model's extra turn ends with ``stop_hook_active`` set, and the hook passes
  at once then. The transcript's own trailing assistant entries are checked
  beside the field and would carry the earlier blocks, but at ``Stop`` time
  they are not written yet, so they cannot prevent that one block.
* A turn that ends on an API error fires Claude Code's ``StopFailure`` event,
  not ``Stop``, so this check never runs on it (``hooks/hooks.json`` runs it
  on ``Stop`` alone). :func:`ended_on_api_error` still passes such a turn, for
  a transcript checked by hand or read later: the model wrote no report there.
* On a miss it prints exactly one line, ``{"decision": "block", "reason": …}``,
  the one form Claude Code reads as a ``Stop`` decision. The reason names each
  missing command in its own fenced block and asks for it copied exactly (and
  repeats the block's other commands), and it asks for the whole report again
  (:data:`WHOLE_REPORT`), since the reply becomes the final message: racing
  run 5's block got back the `### Next Steps` section alone, and the run's
  result lost its report. The model gets one more turn: the hook script exits
  at once when ``stop_hook_active`` is true, and so does this.
* The interactive UI labels a block "Stop hook error occurred" (a
  ``stop-hook-error`` notification in a ``stream-json`` stream). Measured on
  racing run 5 (Claude Code 2.1.289): it fired only in the round this hook
  blocked, while the hook's own record read exit 0, empty stderr, the one
  JSON line on stdout and ``outcome: success`` — so the label is Claude
  Code's word for a block, not a fault in this hook, and no documented
  output field turns it off.

**Advisory, so every error passes silently.** An unreadable or malformed
transcript, a bad argument, a missing file: exit 0 and no output. A check that
could break a session, or block on data it could not read, is worse than the
miss it exists to catch.

**Stdout is the hook's decision channel.** Called through the dispatcher, the
decision line is followed by the dispatcher's own ``MULTI_LLM_RESULT`` line,
and Claude Code parses hook stdout as JSON only when the whole of it is JSON.
So the decision is printed on a line of its own that starts with
:data:`DECISION_PREFIX`, and ``hooks/report-check.sh`` passes that line alone
through (``grep '^{"decision"'``). Nothing else here prints to stdout.

Import is side-effect free: argparse and all I/O live inside :func:`main`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

try:
    from .stream_bootstrap import bootstrap_streams
except ImportError:  # direct script invocation
    from stream_bootstrap import bootstrap_streams  # type: ignore

#: The block's four line kinds (plans/next-phase-command.md §2.2(2)). The
#: trailing space is part of each command prefix: ``[NEXT_COMMAND] `` must never
#: match ``[NEXT_COMMAND_OPTIONAL] ``.
NEXT_COMMAND = "[NEXT_COMMAND] "
NEXT_COMMAND_OPTIONAL = "[NEXT_COMMAND_OPTIONAL] "
NEXT_COMMAND_OPTIONAL_WHY = "[NEXT_COMMAND_OPTIONAL_WHY] "
#: A stopped implement's answers (``util next-command``, fix round 14; each
#: alone after the combined one since fix round 20), after the primary and
#: any optional entries, each below its why line.
NEXT_COMMAND_ANSWERS = "[NEXT_COMMAND_ANSWERS] "
NEXT_COMMAND_ANSWERS_WHY = "[NEXT_COMMAND_ANSWERS_WHY] "
WORKFLOW_COMPLETE = "[WORKFLOW_COMPLETE]"
#: One per implement task the run deferred, under ``[WORKFLOW_COMPLETE]`` or
#: after a command block's commands: ``<id> <title>``
#: (``next_phase.MANUAL_TASK_MARKER``).
MANUAL_TASK = "[MANUAL_TASK] "
#: After those lines: how the user records them done
#: (``next_phase.MANUAL_TASK_NOTE_MARKER``). Quoted in a block's reason.
MANUAL_TASK_NOTE = "[MANUAL_TASK_NOTE] "
#: After the note: the command that records them done, alone
#: (``next_phase.MANUAL_TASK_COMMAND_MARKER``, fix round 22). A command the
#: report carries in a fenced block like the block's others.
MANUAL_TASK_COMMAND = "[MANUAL_TASK_COMMAND] "

#: What the report says when the block is ``[WORKFLOW_COMPLETE]`` (§2.3).
COMPLETE_SENTENCE = "Workflow complete."

#: Every decision line starts with this, so the hook script can select it from
#: a stdout that also carries the dispatcher's result line.
DECISION_PREFIX = '{"decision"'

#: A fence opener: three or more backticks or tildes. Any indentation, so a
#: fence nested in a list item counts; the info string (```bash) is ignored.
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")


class _Unreadable(Exception):
    """The transcript or hook input could not be read as what it claims to be."""


class _ArgumentError(Exception):
    """argparse refused the command line; the caller passes silently."""


class _QuietParser(argparse.ArgumentParser):
    """An ``ArgumentParser`` whose usage error raises instead of exiting 2.

    A hook must never print a usage error into the session or exit non-zero,
    so a bad command line becomes a silent pass. ``--help`` still prints and
    exits 0, as it does for every other util command.
    """

    def error(self, message: str):  # type: ignore[override]
        raise _ArgumentError(message)


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------

class Block:
    """One ``[NEXT_COMMAND]`` block, or one ``[WORKFLOW_COMPLETE]`` line."""

    def __init__(self, complete: bool, command: Optional[str] = None):
        self.complete = complete
        self.command = command
        #: ``(why, command)`` pairs, in the order the block printed them.
        self.optional: List[tuple] = []
        #: ``(why, command)`` pairs: the answers lines, in the order printed.
        self.answers: List[tuple] = []
        #: ``(id, title)`` pairs the block names: the manual tasks the user
        #: still owes (``title`` may be ``""``), on either kind of block.
        self.manual: List[tuple] = []
        #: The ``[MANUAL_TASK_NOTE]`` text after them, when printed.
        self.note: Optional[str] = None
        #: The ``[MANUAL_TASK_COMMAND]`` after the note, when printed.
        self.manual_command: Optional[str] = None

    def commands(self) -> List[str]:
        """Every command the report has to carry: the primary one first, the
        manual tasks' close command last. A complete block's only command is
        that last one (its primary is the sentence)."""
        manual = [self.manual_command] if self.manual_command else []
        if self.complete or self.command is None:
            return manual
        return (
            [self.command]
            + [command for _why, command in self.optional]
            + [command for _why, command in self.answers]
            + manual
        )


def find_blocks(text: str) -> List[Block]:
    """Every block in one tool result's text, in the order printed.

    A ``[NEXT_COMMAND]`` line or a ``[WORKFLOW_COMPLETE]`` line starts a block;
    each ``[NEXT_COMMAND_OPTIONAL]`` line joins the block started before it in
    the same text, with the why line printed directly above it. An optional
    line with no primary above it is ignored: it is not a block the emitter
    prints.
    """
    blocks: List[Block] = []
    current: Optional[Block] = None
    pending_why: Optional[str] = None
    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if line.startswith(NEXT_COMMAND):
            command = line[len(NEXT_COMMAND):].strip()
            current = Block(complete=False, command=command) if command else None
            if current is not None:
                blocks.append(current)
            pending_why = None
        elif line.rstrip() == WORKFLOW_COMPLETE:
            current = Block(complete=True)
            blocks.append(current)
            pending_why = None
        elif line.startswith(NEXT_COMMAND_OPTIONAL_WHY):
            pending_why = line[len(NEXT_COMMAND_OPTIONAL_WHY):].strip()
        elif line.startswith(NEXT_COMMAND_ANSWERS_WHY):
            pending_why = line[len(NEXT_COMMAND_ANSWERS_WHY):].strip()
        elif line.startswith(NEXT_COMMAND_ANSWERS):
            command = line[len(NEXT_COMMAND_ANSWERS):].strip()
            if current is not None and not current.complete and command:
                current.answers.append((pending_why, command))
            pending_why = None
        elif line.startswith(NEXT_COMMAND_OPTIONAL):
            command = line[len(NEXT_COMMAND_OPTIONAL):].strip()
            if current is not None and not current.complete and command:
                current.optional.append((pending_why, command))
            pending_why = None
        elif line.startswith(MANUAL_TASK_NOTE):
            note = line[len(MANUAL_TASK_NOTE):].strip()
            if current is not None and current.manual and note:
                current.note = note
        elif line.startswith(MANUAL_TASK_COMMAND):
            command = line[len(MANUAL_TASK_COMMAND):].strip()
            if current is not None and current.manual and command:
                current.manual_command = command
        elif line.startswith(MANUAL_TASK):
            task_id, _, title = line[len(MANUAL_TASK):].strip().partition(" ")
            if current is not None and task_id:
                current.manual.append((task_id, title.strip()))
    return blocks


# ---------------------------------------------------------------------------
# The message
# ---------------------------------------------------------------------------

def fenced_lines(text: str) -> Set[str]:
    """Every line inside a fenced code block of *text*, whitespace-stripped.

    CommonMark's closing rule: the same fence character, at least as many of
    it, and nothing else on the line. An unclosed fence runs to the end of the
    message, as it renders. A backtick fence whose info string holds a
    backtick is not a fence (CommonMark again), which keeps an inline
    ```` ```code``` ```` span from opening one.
    """
    lines: Set[str] = set()
    fence: Optional[str] = None
    for raw in text.splitlines():
        if fence is None:
            match = _FENCE_RE.match(raw)
            if match is None:
                continue
            marker, info = match.group(1), match.group(2)
            if marker[0] == "`" and "`" in info:
                continue
            fence = marker
            continue
        stripped = raw.strip()
        if (
            stripped
            and set(stripped) == {fence[0]}
            and len(stripped) >= len(fence)
        ):
            fence = None
            continue
        lines.add(stripped)
    return lines


def missing_commands(block: Block, message: str) -> List[str]:
    """The block's commands that *message* does not carry, in block order.

    For a ``[WORKFLOW_COMPLETE]`` block the "commands" are the sentence, missing
    when the message does not contain it. On either kind of block, the id of
    each manual task it names follows, missing when the message does not name
    it as a word of its own (``T011``, not ``T0110``): a final message that
    lists them passes, and one that drops them does not.
    """
    missing = (
        [] if not block.complete or COMPLETE_SENTENCE in message
        else [COMPLETE_SENTENCE]
    )
    carried = fenced_lines(message)
    for command in block.commands():
        if command not in carried and command not in missing:
            missing.append(command)
    for task_id in missing_manual_tasks(block, message):
        if task_id not in missing:
            missing.append(task_id)
    return missing


def missing_manual_tasks(block: Block, message: str) -> List[str]:
    """The ids of *block*'s manual tasks that *message* does not name."""
    return [
        task_id for task_id, _title in block.manual
        if re.search(rf"(?<![\w]){re.escape(task_id)}(?![\w])", message) is None
    ]


#: What every block reason asks for (fix round 14, racing run 5 Claude Code
#: 08): the reply after a block is what the run's final message becomes, and
#: a reason that asked for "your report's `### Next Steps` section" got back
#: that section alone, so the run's result lost the report it had written
#: (what was fixed, skipped and left out). The whole report is asked for again.
WHOLE_REPORT = (
    "Your next reply replaces your last message as the run's final message, "
    "so send your whole report again — everything your last message said "
    "(what ran, what changed, what was skipped, failed or left out, the files "
    "and the counts), unchanged — and end it with"
)


def _manual_list(block: Block) -> str:
    """The block's manual tasks as a list, then its note when it printed one,
    and on a complete block the command after it, fenced (a command block
    lists it among its commands)."""
    listed = "\n".join(
        f"- {task_id} {title}".rstrip() for task_id, title in block.manual
    )
    if block.note:
        listed += (
            "\n\nThen this note, verbatim, and the command after it alone in its "
            "own fenced block (the user's to type, never yours to run):\n\n"
            + block.note
        )
        if block.complete and block.manual_command:
            listed += f"\n\n```\n{block.manual_command}\n```"
    return listed


def _tasks_noun(ids: Sequence[str]) -> str:
    return ("task " if len(ids) == 1 else "tasks ") + ", ".join(ids)


def block_reason(block: Block, missing: Sequence[str]) -> str:
    """The text the model reads when the hook blocks its stop."""
    if block.complete:
        if not block.manual:
            return (
                "multi-llm report check: the last phase result in this turn printed "
                f"{WORKFLOW_COMPLETE}, and your final message does not say "
                f'"{COMPLETE_SENTENCE}". {WHOLE_REPORT} its '
                f'`### Next Steps` section saying "{COMPLETE_SENTENCE}" and '
                "carrying no multi-llm command."
            )
        # The reply replaces the final message, so it is told the whole
        # section again, the tasks it already named included.
        lacks = []
        if COMPLETE_SENTENCE in missing:
            lacks.append(f'does not say "{COMPLETE_SENTENCE}"')
        dropped = [item for item in missing
                   if item not in (COMPLETE_SENTENCE, block.manual_command)]
        if dropped:
            lacks.append(f"does not name the {_tasks_noun(dropped)}")
        if block.manual_command and block.manual_command in missing:
            lacks.append(
                "does not carry the command that records them done alone in a "
                "fenced block")
        return (
            "multi-llm report check: the last phase result in this turn printed "
            f"{WORKFLOW_COMPLETE} with the tasks implement deferred to the "
            f"user, and your final message {' and '.join(lacks)}. {WHOLE_REPORT} "
            f"its `### Next Steps` section saying "
            f'"{COMPLETE_SENTENCE}" and then listing each of these '
            "tasks, its id and title, as the user's to do, carrying no "
            + ("other " if block.note else "")
            + "multi-llm command:\n\n" + _manual_list(block)
        )
    commands = block.commands()
    lost = [item for item in missing if item in commands]
    dropped = [item for item in missing if item not in commands]
    tasks_after = (
        "After the commands, list each of these tasks implement deferred to "
        "the user, its id and title, as the user's to do; never write "
        f'"{COMPLETE_SENTENCE}" for them: while a command is next, the '
        "workflow goes on."
    )
    if not lost:
        # Fix round 17: every command is there and the tasks after them are
        # not. The reply replaces the final message, so it carries the
        # commands again too.
        parts = [
            "multi-llm report check: the last phase result in this turn printed "
            "the next step and, after it, the tasks implement deferred to the "
            f"user, and your final message does not name the {_tasks_noun(dropped)}. "
            f"{WHOLE_REPORT} its `### Next Steps` section, with each command "
            "below in its own fenced code block, copied character for character "
            "as before. " + tasks_after,
        ]
        for command in commands:
            parts.append(f"```\n{command}\n```")
        parts.append(_manual_list(block))
        return "\n\n".join(parts)
    noun = "command" if len(lost) == 1 else "commands"
    parts = [
        "multi-llm report check: the last phase result in this turn printed "
        f"the next step, and your final message does not carry the {noun} "
        f"below exactly. {WHOLE_REPORT} its `### Next Steps` "
        "section, with each command in its own fenced code block, "
        "copied character for character from its [NEXT_COMMAND], "
        "[NEXT_COMMAND_OPTIONAL], [NEXT_COMMAND_ANSWERS] or "
        "[MANUAL_TASK_COMMAND] line (optional "
        "commands after the main one, under the line \"Optional, before "
        "that:\", each with its why; the answers commands last, each with its why). "
        "Do not "
        "rebuild, shorten, re-quote or rephrase a command, and do not run it: "
        "it is for the user to type.",
        f"Missing {noun}:",
    ]
    for command in lost:
        parts.append(f"```\n{command}\n```")
    # The reply replaces the final message, so it has to carry the commands
    # the report already had right as well.
    others = [command for command in commands if command not in lost]
    if others:
        parts.append(
            "Carry the block's other "
            + ("command" if len(others) == 1 else "commands")
            + " in the reply too, unchanged:"
        )
        for command in others:
            parts.append(f"```\n{command}\n```")
    if block.manual:
        parts.append(tasks_after)
        parts.append(_manual_list(block))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# The transcript
# ---------------------------------------------------------------------------

def _content(entry: Dict[str, Any]) -> Any:
    message = entry.get("message")
    if not isinstance(message, dict):
        return None
    return message.get("content")


def _is_real_prompt(entry: Dict[str, Any]) -> bool:
    """A user entry that starts a turn: typed text, not a tool-result carrier."""
    if entry.get("type") != "user":
        return False
    if entry.get("isMeta") or entry.get("isCompactSummary"):
        return False
    content = _content(entry)
    if isinstance(content, str):
        return True
    if isinstance(content, list):
        return any(
            isinstance(block, dict) and block.get("type") == "text"
            for block in content
        )
    return False


def _tool_result_texts(entry: Dict[str, Any]) -> Iterable[str]:
    """The text of every ``tool_result`` block a user entry carries, in order."""
    content = _content(entry)
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        inner = block.get("content")
        if isinstance(inner, str):
            yield inner
        elif isinstance(inner, list):
            texts = [
                part.get("text", "")
                for part in inner
                if isinstance(part, dict) and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            ]
            if texts:
                yield "\n".join(texts)


def _assistant_text(entry: Dict[str, Any]) -> List[str]:
    content = _content(entry)
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    return [
        block["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    ]


def read_turn(path: Path) -> List[Dict[str, Any]]:
    """The transcript entries since the last real user prompt, in file order.

    Read from the end, so a long session parses only its last turn. A line
    inside that stretch that is not a JSON object makes the whole transcript
    unreadable (:class:`_Unreadable`): the check does not guess at a turn it
    cannot see in full.
    """
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise _Unreadable(str(exc)) from exc
    turn: List[Dict[str, Any]] = []
    for raw in reversed(data.splitlines()):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError as exc:
            raise _Unreadable(f"not JSON: {exc}") from exc
        if not isinstance(entry, dict):
            raise _Unreadable("a transcript line is not a JSON object")
        if entry.get("isSidechain"):
            continue
        if _is_real_prompt(entry):
            break
        turn.append(entry)
    turn.reverse()
    return turn


def last_block(turn: Sequence[Dict[str, Any]]) -> Optional[Block]:
    """The last block printed in the turn's tool results, if any."""
    found: Optional[Block] = None
    for entry in turn:
        if entry.get("type") != "user":
            continue
        for text in _tool_result_texts(entry):
            blocks = find_blocks(text)
            if blocks:
                found = blocks[-1]
    return found


def _trailing_assistant(turn: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The assistant entries after the turn's last user entry, newest first.

    An ``isMeta`` user entry (a hook's feedback, a loaded skill body) does not
    end the run: it is not a turn of the conversation.
    """
    trailing: List[Dict[str, Any]] = []
    for entry in reversed(turn):
        kind = entry.get("type")
        if kind == "user":
            if entry.get("isMeta"):
                continue
            break
        if kind == "assistant":
            trailing.append(entry)
    return trailing


def ended_on_api_error(turn: Sequence[Dict[str, Any]]) -> bool:
    """Whether the turn ended on an API error the harness synthesized.

    True when the newest trailing assistant entry that carries text, or is an
    API error, is the API error: the model wrote no report there to check.
    """
    for entry in _trailing_assistant(turn):
        if entry.get("isApiErrorMessage"):
            return True
        if _assistant_text(entry):
            return False
    return False


def final_message(turn: Sequence[Dict[str, Any]]) -> Optional[str]:
    """The text of the turn's trailing assistant entries.

    Claude Code writes one transcript entry per content block, so one reply
    can span several entries (thinking, then text). The final message is every
    assistant entry after the turn's last user entry. ``None`` when those
    entries carry no text: the transcript has not written the reply yet, which
    is how it stands when the ``Stop`` hook runs (measured on 2.1.289,
    2026-10-05: it ends at the turn's last tool result).
    """
    texts: List[str] = []
    for entry in reversed(_trailing_assistant(turn)):
        texts.extend(_assistant_text(entry))
    if not texts:
        return None
    return "\n".join(texts)


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

def check(
    transcript: Path,
    last_assistant_message: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """The hook decision for one turn: ``None`` to pass, else the block object.

    The turn passes when either the transcript's trailing assistant text or
    *last_assistant_message* carries the copy. The second is only the last
    text block of the reply (see the module docstring), so a copy made in an
    earlier block of a split reply is a miss here, once: the hook's next run
    sees ``stop_hook_active`` and passes.

    Raises :class:`_Unreadable` for a transcript it cannot read; the caller
    turns that into a silent pass.
    """
    turn = read_turn(transcript)
    block = last_block(turn)
    if block is None or ended_on_api_error(turn):
        return None
    # A reply this check cannot see is never a miss: with the transcript not
    # yet holding it and no hook-input copy of it, the turn passes.
    messages = [
        message
        for message in (final_message(turn), last_assistant_message)
        if message is not None
    ]
    if not messages:
        return None
    misses = [missing_commands(block, message) for message in messages]
    if any(not missing for missing in misses):
        return None
    # Name what the freshest message lacks: the hook input's, when it has one.
    missing = misses[-1]
    return {"decision": "block", "reason": block_reason(block, missing)}


def _read_hook_input(source: str) -> Dict[str, Any]:
    try:
        if source == "-":
            raw = sys.stdin.buffer.read() if hasattr(sys.stdin, "buffer") else (
                sys.stdin.read().encode("utf-8")
            )
        else:
            raw = Path(source).read_bytes()
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except (OSError, ValueError) as exc:
        raise _Unreadable(f"hook input: {exc}") from exc
    if not isinstance(data, dict):
        raise _Unreadable("hook input is not a JSON object")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = _QuietParser(
        prog="multi_llm.py util report-check",
        description=(
            "Check that a Claude Code turn's final message carries the next "
            "command a phase result printed ([NEXT_COMMAND] lines), each in a "
            "fenced code block and copied exactly, or says \"Workflow "
            "complete.\" for [WORKFLOW_COMPLETE]. Run by the Claude Code "
            "plugin's Stop hook. Prints nothing on a pass, and one "
            '{"decision": "block", "reason": ...} line on a miss. Always '
            "exits 0: every error is a silent pass."
        ),
    )
    parser.add_argument(
        "--hook-input",
        metavar="PATH",
        help="The Stop hook's input JSON, as a file or - for stdin. Supplies "
             "transcript_path, stop_hook_active and last_assistant_message.",
    )
    parser.add_argument(
        "--transcript",
        metavar="PATH",
        help="The session transcript (JSONL). Takes precedence over the hook "
             "input's transcript_path.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    bootstrap_streams()
    parser = build_parser()
    try:
        args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    except _ArgumentError:
        return 0
    try:
        hook: Dict[str, Any] = {}
        if args.hook_input:
            hook = _read_hook_input(args.hook_input)
            if hook.get("stop_hook_active") is True:
                return 0
        transcript = args.transcript or hook.get("transcript_path")
        if not isinstance(transcript, str) or not transcript:
            return 0
        path = Path(transcript)
        if not path.is_file():
            return 0
        last_message = hook.get("last_assistant_message")
        decision = check(
            path, last_message if isinstance(last_message, str) else None
        )
    except Exception:  # noqa: BLE001 - advisory: every failure passes silently
        return 0
    if decision is not None:
        sys.stdout.write(json.dumps(decision) + "\n")
        try:
            sys.stdout.flush()
        except (AttributeError, ValueError):  # pragma: no cover - closed stream
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
