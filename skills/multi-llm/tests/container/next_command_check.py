"""T10 of plans/next-phase-command.md: what a live host did with the next command.

Pure functions over one host-lane bundle, and nothing else. The credentialed
rows that run them are `live`-marked beneath a container gate, so they only
ever answer inside a paid Docker run; every reader here is therefore written
against the hosts' recorded stream shapes and unit-tested offline
(`tests/test_next_command_live_check.py`) before a cent is spent on it.

Three observations, one per check the plan names:

* **The copy (§5 criterion 4).** The host's FINAL assistant message, read out of
  the host's own recorded stream (`host-stdout.txt`), must carry the command the
  dispatcher printed — `user_next_step.command` — as the one line of a fenced
  block, byte for byte, plus one fenced line per `optional_before` entry. On a
  `complete` answer it must say "Workflow complete." and carry no fenced command.
  On claude-code two further facts are recorded, so a skipped lookup is never
  scored as a bad copy: whether a `util next-command` call (or a phase result
  carrying the block) followed the phase's last dispatcher step, and whether the
  T11 Stop hook blocked.
* **The no-call (criterion 5).** From the host's recorded tool calls: no
  `multi_llm` call after the phase's terminal result, and none whose mode is
  `user_next_step.phase`. A receipt for that mode in the session's own receipt
  directory counts as a call too, whoever made it — a model that shelled out to
  the dispatcher instead of calling the tool still ran the command.
* **The typed template (§2.1, criterion 6).** A FRESH session of the same host
  and route was handed the printed command verbatim (`typed-run.json`, written by
  `entrypoint.run_typed_followup`), and its own receipts must show the
  dispatcher running the named mode with a plan argument that names the fixture
  plan and holds no quote character.

**Where `user_next_step` is read from.** The dispatcher's own receipts —
`wrapper.run_uv` tees the dispatcher's stdout into the tool result the model is
handed and parses the same `MULTI_LLM_RESULT` line the model reads, with the
product's own parser. On the tool routes the servers lift the same value into
their result (`lift_user_next_step`); the receipt holds the payload it was lifted
from. That is the rig's anchor everywhere else too (C8), because a host's event
stream carries the line at best JSON-escaped inside a nested envelope.

Nothing here raises on a malformed stream: a reader that cannot find a thing
returns ``None`` or an empty list, and the caller turns that into a named
failure. A crash would hide which half of the evidence was missing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

#: The tool both packaged servers register (`DEGRADED_TOOL_NAME` in
#: `packaging/codex/mcp_server.py` and `packaging/opencode/src/index.mjs`).
TOOL_NAME = "multi_llm"

#: A fence opener or closer: three or more backticks or tildes, optionally
#: indented (a report's `### Next Steps` list item indents its fence).
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")

#: Any quote character a typed plan path could have kept. A model that copies
#: `"plans/next phase.md"` into a shell argument the shell does not strip, or
#: into a tool property the server does not strip, leaves one of these in argv.
QUOTE_CHARS = ("\"", "'", "“", "”", "‘", "’", "`")

#: The words `util` subcommands a phase's own continuation uses. `next-command`
#: is the lookup itself and is never one of the phase's steps.
_LOOKUP = ("util", "next-command")

#: The literal the report writes for a `complete` answer (summary-format.md).
WORKFLOW_COMPLETE_TEXT = "Workflow complete."


# --------------------------------------------------------------------------- #
# Small readers
# --------------------------------------------------------------------------- #

def _json_lines(text: str) -> list[dict]:
    """Every line of *text* that parses as a JSON object, in order."""
    out: list[dict] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            out.append(value)
    return out


def _json_object(text: Any) -> dict | None:
    """*text* parsed as one JSON object, or ``None``."""
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        value = json.loads(stripped)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def fenced_lines(message: str | None) -> list[str]:
    """Every line inside a fenced code block of *message*, stripped.

    Stripped of surrounding whitespace only: a fence nested in a list item is
    indented, and that indentation is the report's layout, not part of the
    command. Everything between the first and last non-space character is
    compared byte for byte by the caller.
    """
    lines: list[str] = []
    inside: str | None = None
    for raw in (message or "").splitlines():
        match = _FENCE.match(raw)
        if inside is None:
            if match:
                inside = match.group(1)[0]
            continue
        if match and match.group(1)[0] == inside and not raw.strip()[len(match.group(1)):].strip():
            inside = None
            continue
        if raw.strip():
            lines.append(raw.strip())
    return lines


# --------------------------------------------------------------------------- #
# The host stream: final message and tool calls, per host
# --------------------------------------------------------------------------- #

def _codex_items(stdout: str) -> list[dict]:
    """`item.completed` items of a `codex exec --json` stream, in order."""
    return [event["item"] for event in _json_lines(stdout)
            if event.get("type") == "item.completed"
            and isinstance(event.get("item"), dict)]


def _opencode_events(stdout: str) -> list[dict]:
    """`text` and `tool_use` events of an `opencode run --format json` stream."""
    return [event for event in _json_lines(stdout)
            if event.get("type") in ("text", "tool_use")
            and isinstance(event.get("part"), dict)]


def claude_envelope(stdout: str) -> dict:
    """claude-code's `--output-format json` result envelope, or ``{}``."""
    for event in reversed(_json_lines(stdout)):
        if event.get("type") == "result" or "result" in event:
            return event
    whole = _json_object(stdout)
    return whole or {}


def final_message(host: str, stdout: str) -> str | None:
    """The host's final assistant message: the text after its last tool call.

    codex and opencode emit one text item per step, and a closing report can
    arrive as more than one item after the last tool call, so every text item
    after the last tool item is joined. claude-code prints one envelope whose
    `result` is the final message.
    """
    if host == "claude-code":
        result = claude_envelope(stdout).get("result")
        return result if isinstance(result, str) else None
    if host == "codex":
        texts: list[str] = []
        for item in _codex_items(stdout):
            kind = item.get("type")
            if kind == "agent_message":
                texts.append(str(item.get("text") or ""))
            elif kind in ("command_execution", "mcp_tool_call", "file_change",
                          "web_search"):
                texts = []
        return "\n".join(texts) if texts else None
    if host == "opencode":
        texts = []
        for event in _opencode_events(stdout):
            part = event["part"]
            if event.get("type") == "tool_use":
                texts = []
            elif part.get("type") == "text":
                texts.append(str(part.get("text") or ""))
        return "\n".join(texts) if texts else None
    return None


def _shell_mode(command: str) -> str | None:
    """The dispatcher mode a shell command line runs, or ``None``."""
    match = re.search(r"multi_llm\.py[\"']?\s+(\S+)(?:\s+(\S+))?", command or "")
    if not match:
        return None
    mode = match.group(1).strip("\"'")
    if mode == "util" and match.group(2):
        return f"util {match.group(2).strip(chr(34) + chr(39))}"
    return mode


def tool_calls(host: str, stdout: str) -> list[dict]:
    """The host's tool calls in order: ``{index, kind, name, mode, arguments, output}``.

    ``kind`` is ``tool`` for a call to the packaged `multi_llm` tool, ``shell``
    for a shell command (its ``mode`` set when the command runs the dispatcher),
    and ``other`` for anything else. claude-code's stream is one envelope with no
    tool calls in it, so it answers ``[]``; that host has no tool route.
    """
    calls: list[dict] = []
    if host == "codex":
        for item in _codex_items(stdout):
            kind = item.get("type")
            if kind == "mcp_tool_call":
                arguments = item.get("arguments")
                arguments = arguments if isinstance(arguments, dict) else (
                    _json_object(arguments) or {})
                result = item.get("result") or {}
                texts = [str(c.get("text") or "") for c in
                         (result.get("content") or []) if isinstance(c, dict)]
                output = result.get("structured_content") or "\n".join(texts)
                name = str(item.get("tool") or "")
                calls.append({
                    "index": len(calls),
                    "kind": "tool" if name == TOOL_NAME else "other",
                    "name": name,
                    "mode": arguments.get("mode") if name == TOOL_NAME else None,
                    "arguments": arguments,
                    "output": output,
                })
            elif kind == "command_execution":
                command = str(item.get("command") or "")
                calls.append({
                    "index": len(calls), "kind": "shell", "name": "shell",
                    "mode": _shell_mode(command), "arguments": {"command": command},
                    "output": str(item.get("aggregated_output") or ""),
                })
    elif host == "opencode":
        for event in _opencode_events(stdout):
            if event.get("type") != "tool_use":
                continue
            part = event["part"]
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            arguments = state.get("input")
            arguments = arguments if isinstance(arguments, dict) else (
                _json_object(arguments) or {})
            name = str(part.get("tool") or "")
            if name == TOOL_NAME:
                kind, mode = "tool", arguments.get("mode")
            elif name == "bash":
                kind, mode = "shell", _shell_mode(str(arguments.get("command") or ""))
            else:
                kind, mode = "other", None
            calls.append({"index": len(calls), "kind": kind, "name": name,
                          "mode": mode, "arguments": arguments,
                          "output": state.get("output")})
    return calls


def _payload_of(output: Any) -> dict | None:
    """A tool result's JSON body, when it is one."""
    if isinstance(output, dict):
        return output
    return _json_object(output)


def carries_next_step(call: dict) -> bool:
    """Whether this tool result handed the model a `user_next_step`.

    A packaged tool's result carries it as a top-level key (lifted by the
    server, §2.4) or inside its `result` payload; a shell result carries the
    dispatcher's own `MULTI_LLM_RESULT` line, or the `[NEXT_COMMAND]` /
    `[WORKFLOW_COMPLETE]` block printed above it.
    """
    payload = _payload_of(call.get("output"))
    if payload is not None:
        if payload.get("user_next_step") is not None:
            return True
        inner = payload.get("result")
        if isinstance(inner, dict) and inner.get("user_next_step") is not None:
            return True
    text = call.get("output") if isinstance(call.get("output"), str) else ""
    return bool(text) and ("\"user_next_step\"" in text
                           or "[NEXT_COMMAND]" in text
                           or "[WORKFLOW_COMPLETE]" in text)


# --------------------------------------------------------------------------- #
# Receipts
# --------------------------------------------------------------------------- #

def read_receipts(directory: Path) -> list[dict]:
    """``NNN.receipt.json`` files of *directory*, in claim order."""
    if not directory.is_dir():
        return []
    out: list[dict] = []
    for path in sorted(directory.glob("*.receipt.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(value, dict) and "install-host" not in (value.get("argv") or []):
            out.append(value)
    return sorted(out, key=lambda r: (int(r.get("seq") or 0)))


def receipt_words(receipt: dict) -> list[str]:
    """The dispatcher argv after `multi_llm.py` (mode word first), or ``[]``."""
    argv = [str(token) for token in (receipt.get("argv") or [])]
    for index, token in enumerate(argv):
        if token.endswith("multi_llm.py"):
            return argv[index + 1:]
    return []


def receipt_mode(receipt: dict) -> str | None:
    """`implement`, `generate-tasks`, `util next-command`, … or ``None``."""
    words = receipt_words(receipt)
    if not words:
        return None
    if words[0] == "util" and len(words) > 1:
        return f"util {words[1]}"
    return words[0]


def receipt_plan_arg(receipt: dict) -> str | None:
    """The plan argument a plan-mode receipt carried, exactly as argv held it.

    `--plan-file <value>` / `--plan-file=<value>` when the caller used the flag
    form, else the first positional after the mode word — which is where every
    plan mode takes it (`modes.POS_PLAN`), and which `multi_llm._normalize_
    positionals` copies into `--plan-file` unchanged.
    """
    words = receipt_words(receipt)
    for index, token in enumerate(words):
        if token == "--plan-file" and index + 1 < len(words):
            return words[index + 1]
        if token.startswith("--plan-file="):
            return token.split("=", 1)[1]
    start = 2 if words[:1] == ["util"] else 1
    for token in words[start:]:
        if token.startswith("-"):
            return None
        return token
    return None


def last_user_next_step(receipts: Iterable[dict]) -> tuple[dict | None, int | None]:
    """The last receipt payload's `user_next_step`, and that receipt's position."""
    found: tuple[dict | None, int | None] = (None, None)
    for position, receipt in enumerate(receipts):
        payload = receipt.get("result")
        if isinstance(payload, dict) and isinstance(payload.get("user_next_step"), dict):
            found = (payload["user_next_step"], position)
    return found


# --------------------------------------------------------------------------- #
# The three checks
# --------------------------------------------------------------------------- #

def copy_check(message: str | None, step: dict | None,
               prefixes: Iterable[str] = ()) -> dict:
    """Criterion 4 over one final message and the printed `user_next_step`.

    `prefixes` are the host's `next_command` template prefixes: on a `complete`
    answer, a fenced line starting with one of them is a command the report
    should not carry.
    """
    out: dict = {"message_present": message is not None}
    if not isinstance(step, dict):
        out.update(observable=False, reason="no user_next_step was printed")
        return out
    lines = fenced_lines(message)
    out["observable"] = True
    if step.get("complete"):
        stray = [line for line in lines
                 if any(line.startswith(p) for p in prefixes if p)]
        says = WORKFLOW_COMPLETE_TEXT in (message or "")
        out.update(complete=True, says_complete=says, stray_commands=stray,
                   passes=bool(says and not stray))
        return out
    command = str(step.get("command") or "")
    optional = [str((entry or {}).get("command") or "")
                for entry in (step.get("optional_before") or [])]
    copied = command in lines
    optional_copied = [entry in lines for entry in optional]
    out.update(
        complete=False,
        command=command,
        command_copied=copied,
        # Diagnostics, never a pass: present somewhere but outside a fence, or
        # a fenced line that is a near miss of the command.
        command_anywhere=bool(command) and command in (message or ""),
        optional_commands=optional,
        optional_copied=optional_copied,
        near_misses=[line for line in lines if line != command
                     and line not in optional
                     and any(line.startswith(p) for p in prefixes if p)],
        passes=bool(copied and all(optional_copied)),
    )
    return out


def no_call_check(calls: list[dict], step: dict | None,
                  session_receipts: list[dict], terminal_receipt: int | None) -> dict:
    """Criterion 5: nothing ran the printed command, and nothing ran after the end.

    The terminal result is the last tool result that handed the model a
    `user_next_step`. A dispatcher call after it is a violation whatever its
    mode; a call — or a receipt — whose mode is the next phase is one anywhere.
    """
    out: dict = {}
    if not isinstance(step, dict):
        out.update(observable=False, reason="no user_next_step was printed")
        return out
    phase = step.get("phase")
    dispatching = [c for c in calls if c.get("kind") == "tool"
                   or (c.get("kind") == "shell" and c.get("mode"))]
    terminal = None
    for call in dispatching:
        if carries_next_step(call):
            terminal = call["index"]
    after = [c for c in dispatching
             if terminal is not None and c["index"] > terminal]
    named = [c for c in dispatching if phase and c.get("mode") == phase]
    later_receipts = [receipt_mode(r) for i, r in enumerate(session_receipts)
                      if terminal_receipt is not None and i > terminal_receipt]
    named_receipts = [receipt_mode(r) for r in session_receipts
                      if phase and receipt_mode(r) == phase]
    out.update(
        observable=terminal is not None,
        phase=phase,
        terminal_call_index=terminal,
        calls=[{"index": c["index"], "kind": c["kind"], "mode": c.get("mode")}
               for c in dispatching],
        calls_after_terminal=[{"index": c["index"], "mode": c.get("mode")}
                              for c in after],
        calls_naming_next_phase=[{"index": c["index"], "kind": c["kind"]}
                                 for c in named],
        receipts_after_terminal=later_receipts,
        receipts_naming_next_phase=named_receipts,
    )
    out["holds"] = bool(terminal is not None and not after and not named
                        and not named_receipts)
    if terminal is None:
        out["reason"] = ("no tool result in the host stream handed the model a "
                         "user_next_step, so there is no terminal result to "
                         "count from")
    return out


def names_fixture(plan_arg: str | None, fixture: str, workspace: str) -> bool:
    """Whether *plan_arg* is the fixture plan, relative or absolute."""
    if not plan_arg:
        return False
    fixture = fixture.lstrip("./")
    if plan_arg in (fixture, f"./{fixture}"):
        return True
    return plan_arg == f"{workspace.rstrip('/')}/{fixture}"


def typed_check(typed_run: dict, typed_receipts: list[dict], fixture: str,
                workspace: str) -> dict:
    """The typed-template check: the named mode dispatched, the plan unquoted."""
    out: dict = {"exercised": bool(typed_run.get("exercised"))}
    if not out["exercised"]:
        out.update(passes=False,
                   reason=str(typed_run.get("reason") or "no typed session ran"))
        return out
    phase = str(typed_run.get("phase") or "")
    hits = [r for r in typed_receipts if receipt_mode(r) == phase]
    plan_args = [receipt_plan_arg(r) for r in hits]
    quote_free = bool(plan_args) and all(
        arg is not None and not any(q in arg for q in QUOTE_CHARS)
        for arg in plan_args)
    fixture_named = bool(plan_args) and all(
        names_fixture(arg, fixture, workspace) for arg in plan_args)
    out.update(
        typed=typed_run.get("typed"),
        surface=typed_run.get("surface"),
        phase=phase,
        stop_reason=typed_run.get("stop_reason"),
        modes_dispatched=[receipt_mode(r) for r in typed_receipts],
        dispatched=bool(hits),
        plan_args=plan_args,
        quote_free=quote_free,
        names_fixture=fixture_named,
        passes=bool(hits and quote_free and fixture_named),
    )
    return out


def lookup_check(receipts: list[dict], phase: str) -> dict:
    """claude-code's extra: was the command looked up after the phase's last step?

    The phase's steps are its own mode's receipts (the first call, any
    `--reaggregate`) and the `util` calls between them; the lookup is a
    `util next-command` receipt after the last of those, or the last phase
    receipt itself carrying `user_next_step` (a phase result with the block).
    """
    last_phase = None
    for position, receipt in enumerate(receipts):
        if receipt_mode(receipt) == phase:
            last_phase = position
    if last_phase is None:
        return {"lookup": "none", "reason": f"no {phase} receipt"}
    lookups = [position for position, receipt in enumerate(receipts)
               if position > last_phase
               and receipt_words(receipt)[:2] == list(_LOOKUP)]
    payload = receipts[last_phase].get("result")
    carried = isinstance(payload, dict) and isinstance(
        payload.get("user_next_step"), dict)
    if lookups:
        return {"lookup": "util-next-command", "lookup_receipts": len(lookups),
                "phase_result_carried_block": carried}
    if carried:
        return {"lookup": "phase-result", "phase_result_carried_block": True}
    return {"lookup": "none", "phase_result_carried_block": False}


def stop_hook_state(digest_text: str, plugin_installed: bool) -> str:
    """Whether T11's Stop hook blocked: `blocked`, `ran`, `silent` or `not-installed`.

    The hook ships in the PLUGIN's `hooks/hooks.json`; a plain-skill install
    loads no hooks at all, so on that route it cannot have run.
    """
    if not plugin_installed:
        return "not-installed"
    text = digest_text or ""
    if "report-check" in text or "Stop hook" in text or "stop_hook" in text:
        return "blocked" if ("decision" in text and "block" in text) else "ran"
    return "silent"


def route_of(command: str | None, templates: dict[str, str]) -> str | None:
    """The `next_command` route(s) whose template prefix *command* starts with.

    The longest matching prefix wins. Two routes that share one spelling (both
    of opencode's are `/multi-llm {args}`) cannot be told apart by the command,
    so both are named, `|`-joined in sorted order, rather than one guessed.
    """
    if not command:
        return None
    longest, routes = -1, []
    for route, template in (templates or {}).items():
        prefix = str(template).split("{args}", 1)[0]
        if not prefix or not command.startswith(prefix):
            continue
        if len(prefix) > longest:
            longest, routes = len(prefix), [route]
        elif len(prefix) == longest:
            routes.append(route)
    return "|".join(sorted(routes)) or None


# --------------------------------------------------------------------------- #
# One bundle
# --------------------------------------------------------------------------- #

def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def observe(bundle: Path, host: str, row_id: str, matrix_row: dict,
            templates: dict[str, str], workspace: str = "/work/sample-project",
            route: str = "file") -> dict:
    """Every T10 fact one bundle holds, for the record and the tests alike."""
    check = dict(matrix_row.get("next_command_check") or {})
    fixture = str((matrix_row.get("fixture_plan") or {}).get("path") or "")
    receipts = read_receipts(bundle / "receipts" / row_id)
    step, terminal_receipt = last_user_next_step(receipts)
    stdout = _read(bundle / "host-stdout.txt")
    message = final_message(host, stdout)
    prefixes = [str(t).split("{args}", 1)[0] for t in (templates or {}).values()]
    census = _json_object(_read(bundle / "credential-census.json")) or {}
    observed: dict = {
        # The slug the host really ran: the census records the override
        # (`MLLM_HOST_MODEL`) when there was one, where the record's own
        # `model` column carries the checked-in pin.
        "model": census.get("model"),
        "route_observed": route_of((step or {}).get("command"), templates),
        "printed": step,
        "copy": copy_check(message, step, prefixes),
    }
    if host == "claude-code":
        phase = str(check.get("phase") or "")
        observed["lookup"] = lookup_check(receipts, phase) if phase else {
            "lookup": "none", "reason": "the row names no phase"}
        observed["stop_hook"] = stop_hook_state(
            _read(bundle / "host-session-digest.jsonl"),
            plugin_installed=(route == "package"))
    else:
        observed["no_call"] = no_call_check(
            tool_calls(host, stdout), step, receipts, terminal_receipt)
    typed_run = _json_object(_read(bundle / "typed-run.json")) or {}
    typed_receipts = read_receipts(bundle / "receipts" / f"{row_id}-typed")
    observed["typed"] = typed_check(typed_run, typed_receipts, fixture, workspace)
    return observed


def record_block(observed: dict) -> dict:
    """The sanitized summary a run record carries (`rows[].next_command`)."""
    copy = observed.get("copy") or {}
    printed = observed.get("printed") or None
    block: dict = {
        "model": observed.get("model"),
        "route_observed": observed.get("route_observed"),
        "printed_phase": (printed or {}).get("phase"),
        "printed_command": (printed or {}).get("command"),
        "printed_optional": [str((e or {}).get("command") or "")
                             for e in ((printed or {}).get("optional_before") or [])],
        "printed_complete": bool((printed or {}).get("complete")) if printed else None,
        "copied": copy.get("passes") if copy.get("observable") else None,
        "copied_unfenced": (bool(copy.get("command_anywhere"))
                            and not copy.get("command_copied"))
        if copy.get("observable") and not copy.get("complete") else None,
    }
    if "lookup" in observed:
        block["lookup"] = (observed["lookup"] or {}).get("lookup")
        block["stop_hook"] = observed.get("stop_hook")
    if "no_call" in observed:
        no_call = observed["no_call"] or {}
        block["no_call"] = no_call.get("holds") if no_call.get("observable") else None
        block["calls_after_terminal"] = len(no_call.get("calls_after_terminal") or [])
    typed = observed.get("typed") or {}
    block["typed"] = {
        "exercised": bool(typed.get("exercised")),
        "surface": typed.get("surface"),
        "dispatched": typed.get("dispatched"),
        "plan_args": typed.get("plan_args") or [],
        "quote_free": typed.get("quote_free"),
        "names_fixture": typed.get("names_fixture"),
        "passes": bool(typed.get("passes")),
        "stop_reason": typed.get("stop_reason") or typed.get("reason"),
    }
    return block
