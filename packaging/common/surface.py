#!/usr/bin/env python3
"""The generated CLI surface — ``dist/cli-surface.json`` — for every host package.

Lifted out of ``packaging/opencode/build.py`` (D3e) with the host id as a
parameter, so the OpenCode npm package and the codex plugin ship one emitter and
one :data:`SURFACE_SCHEMA_VERSION`. Each host's ``build.py`` is a thin driver
that calls :func:`cli_surface_json` with its own id and writes the result to
``<package>/dist/cli-surface.json``.

**The plugin holds no word of its own.** Every mode word, alias, summary, flag
name, exit code, state word and message a host layer shows a model or composes
into an argv is generated here from ``modes.py``, ``utils/cli_common.py`` and
their siblings. A host layer that spells one is the drift this module exists to
prevent, and each host's no-prose grep holds that line.

**The surface is a function of the source tree alone.** Nothing here reads
``os.environ`` or calls an accessor that does: the uv flag list is
``unattended.DEFAULT_UV_RUN_ARGS`` — the constant — and never
``unattended.uv_run_args()``, which would bake one builder's
``MULTI_LLM_UV_RUN_ARGS`` into every installation. Runtime overrides are read by
the server, at invocation, from its own environment.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SKILL_DIR = _REPO_ROOT / "skills" / "multi-llm"

# The skill tree is the library this emitter reads its single sources out of.
# It is not installed, so it goes on the path the way each host's build.py puts
# it there.
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

import modes as mode_table  # noqa: E402
import multi_llm  # noqa: E402  (side-effect free at import: main() does the work)
from utils import (  # noqa: E402
    cli_common, detach, host_capabilities, next_phase, provider_registry,
    subagent_executor, subagent_tasks, unattended,
)

#: The token a generated artifact carries in place of an absolute skill dir.
#: The host resolves it at startup from wherever the package landed, which is
#: the whole point of a package route: no baked path to go stale.
SKILL_DIR_TOKEN = "{SKILL_DIR}"

#: The option a poll and a re-attach name one run by. One spelling here, read
#: by the ``dispatcher.run_id_option`` key.
RUN_ID_OPTION = "--run-id"

#: Bumped when the *shape* of ``cli-surface.json`` changes. The plugin refuses a
#: version it does not know, so a payload/plugin mismatch is a named error rather
#: than a ``TypeError`` inside a ``.map``.
#:
#: 2: ``needs_human_message``, ``needs_human_option`` and ``outcomes_line``
#: joined the surface (the plugin's exit-3 message and its tool description's
#: outcomes line are read from here, never authored in JS), every mode row
#: carries its ``phase``, and ``dispatcher`` gained ``states`` and
#: ``review_gate`` — the data the exit-3 message is selected from.
#:
#: 3: ``dispatcher`` gained ``session_model_env`` — the name of the variable the
#: plugin exports the *session's own model* on, so the resolution ladder can
#: prefer what the operator selected over the shipped credential-free floor. The
#: name's owner is ``provider_registry.HOST_SESSION_MODEL_ENV``; the plugin reads
#: it from here rather than spelling it, exactly like every flag and mode word.
#:
#: 4: the emitter moved to ``packaging/common/surface.py`` with the host id as a
#: parameter (the codex plugin ships it too), and the words a model reads in
#: the tool definition became surface data: ``tool_name``, ``tool_description``
#: (the whole joined description — lede, mode block, default-mode line,
#: outcomes line, worked examples; ``description_lede``/``outcomes_line`` stay
#: as the pieces it subsumes) and ``argument_descriptions`` (exactly ``mode``,
#: ``plan``, ``question``, ``extra_flags``). Plus what the codex server reads:
#: per-mode ``writes_worktree``, ``write_posture_refusal``,
#: ``anchor_unresolved``, ``shell_missing`` and ``runbook`` (the
#: ``util instructions`` sentinel and banner opening). A host layer composes
#: none of it.
#:
#: 5: ``dispatcher`` gained ``route_env`` — ``{"name", "value"}``, the route
#: signal a server sets on every child it spawns (polls included), so the
#: dispatcher prints the next-phase command in this route's spelling
#: (plans/next-phase-command.md §2.1, "Knowing the route"). The name is
#: ``modes.ROUTE_ENV`` and the value the one ``modes.TOOL_ROUTES`` member the
#: host's ``next_command`` cell carries; a server spells neither. And every
#: mode row gained ``optional_why`` beside ``summary``: ``Mode.optional_why``,
#: the generic reason an optional phase is offered before the next required
#: one (``None`` on every row whose phase is not optional).
SURFACE_SCHEMA_VERSION = 5


def _aliases_for(name: str) -> List[str]:
    """Every alias that resolves to *name*, in table order."""
    return [word for word, target in mode_table.ALIASES.items() if target == name]


def _exposed_modes() -> List[Dict[str, Any]]:
    """What the tool exposes: ``user_facing_modes()`` plus nothing wider.

    That is the ten work modes, ``init``, and ``status`` — which is plumbing but
    must stay exposed, because re-attaching to a detached run is a tool call with
    ``mode: "status"`` and stopping one is the ``abort``/``cancel``/``stop``
    alias. Not ``util`` (a routing word, never user-facing; ``execute`` reaches
    ``util instructions`` itself on a path the model cannot address) and not
    ``install-host``/``uninstall-host`` (already ``user_facing=False``, and on
    this route installing is npm's job).
    """
    rows: List[Dict[str, Any]] = []
    for mode in mode_table.user_facing_modes():
        rows.append({
            "name": mode.name,
            "aliases": _aliases_for(mode.name),
            "positionals": list(mode.positionals),
            "supports_detach": mode.supports_detach,
            "summary": mode.summary,
            # The generic `why` an optional phase carries beside its command in
            # a finished phase's next step (schema v5), `None` on every row
            # whose phase is not optional. Vocabulary only: a server relays the
            # `why` that arrives in a result, never one read from here.
            "optional_why": mode.optional_why,
            # The phase word the exit-3 message names. From the *launch*, never
            # from the payload: a poll's result line is `status`'s own and
            # declares no phase, so the payload cannot say which phase died.
            "phase": mode.phase,
            # How the user names this mode on the host (`--<mode>`), the word
            # a server re-spells a redacted dispatcher command with
            # (`dispatcher.user_command`). The same spelling
            # `next_phase.resume_command` and `render_argument_hint` print.
            "user_flag": f"--{mode.name}",
            # Whether this mode's `--exec-subagents` dispatch can reach an
            # `agent-write` profile — derived, never typed (D3b).
            "writes_worktree": _writes_worktree(mode),
        })
    return rows


def _write_phases() -> FrozenSet[str]:
    """The phases whose dispatch runs a write kind, from the executor's table.

    ``subagent_tasks.PHASE_BY_WRITE_KIND`` is the one owner of "which kinds edit"
    and of the phase each belongs to; every write kind runs ``agent-write``
    (``subagent_executor.PROFILE_BY_KIND``). A phase there that no mode row
    owns is a table drift, and the build says so rather than classifying
    around it.
    """
    phases = frozenset(subagent_tasks.PHASE_BY_WRITE_KIND.values())
    owned = {mode.phase for mode in mode_table.MODES if mode.phase}
    orphans = sorted(phases - owned)
    if orphans:
        raise ValueError(
            "subagent_tasks.PHASE_BY_WRITE_KIND names phase(s) no modes.py row "
            f"owns: {', '.join(orphans)}. Fix the phase spelling in one table "
            "or the other."
        )
    return phases


def _writes_worktree(mode: "mode_table.Mode") -> bool:
    """Can *mode*'s ``--exec-subagents`` dispatch reach ``agent-write``?

    Directly, when its phase dispatches a write kind (the apply family and
    ``implement``); or by containment, when the row is a composite and **any**
    phase of its ``sequence`` writes. Any, not every and not the last alone: the
    question is whether the dispatch can *reach* ``agent-write``, and one
    writing phase reaches it, wherever it sits in the sequence.
    """
    phases = _write_phases()
    if mode.phase in phases:
        return True
    if mode.is_composite:
        return any(phase in phases for phase in mode.sequence_phases)
    return False


# ---------------------------------------------------------------------------
# The model-facing definition (schema v4): tool_description and
# argument_descriptions, rendered here so no host layer composes them
# ---------------------------------------------------------------------------

#: The flag the aliased worked example carries, and the ``unattended`` site
#: whose row names the modes that accept it. Read from that table rather than
#: assumed, because models copy examples: a flag the aliased mode's parser does
#: not define is an argparse usage exit, which the dispatcher maps to 3
#: needs-human. (``--run-id`` was the old spelling here, and only ``status``
#: parses it.)
_EXAMPLE_FLAG = "--quick"
_EXAMPLE_FLAG_SITE = "model-selection"

#: Host-keyed clause of the ``plan`` description: the file-attachment sigil a
#: host prepends to a path the user names, which the plugin strips. OpenCode's
#: is ``@`` (measured 2026-09-16); a host absent here has none recorded, and
#: its description simply lacks the clause — the schema does not change.
_PLAN_SIGIL_CLAUSE: Dict[str, str] = {
    "opencode": ', and one leading "@" (this host\'s file-attachment sigil),',
}


#: Host-keyed paragraph of the tool description, placed after the outcomes
#: line. Codex's is D3b's "the tool says what it is": its server runs outside
#: the session's exec sandbox and the ``writes_worktree`` modes edit the tree.
#: ``{modes}`` is filled with those modes, comma-joined, from the table.
_DESCRIPTION_NOTE: Dict[str, str] = {
    "codex": mode_table.TOOL_WRITE_CAPABLE_NOTE,
}


#: The codex plugin's ``.mcp.json`` source, which owns the tool-call budget
#: (``tool_timeout_sec``) that the server enforces and :func:`long_call_rule`
#: quotes.
CODEX_MCP_JSON = _REPO_ROOT / "packaging" / "codex" / "mcp.json"

#: The hosts whose model calls this tool from a yielding ``exec`` cell.
_LONG_CALL_HOSTS = frozenset({"codex"})


def codex_tool_budget_s() -> int:
    """``tool_timeout_sec`` of the codex plugin's one MCP server, in seconds."""
    servers = json.loads(CODEX_MCP_JSON.read_text(encoding="utf-8"))["mcpServers"]
    (entry,) = servers.values()
    budget = entry["tool_timeout_sec"]
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ValueError(f"{CODEX_MCP_JSON}: tool_timeout_sec is not a positive int")
    return budget


def long_call_rule(host_id: str) -> Optional[str]:
    """``modes.TOOL_LONG_CALL_RULE`` filled from the tool budget, or ``None``.

    The cell's ``yield_time_ms`` is the whole budget, so a cell never hands
    control back before the server answers or hands the run off (fix round 16).
    """
    if host_id not in _LONG_CALL_HOSTS:
        return None
    budget = codex_tool_budget_s()
    minutes = f"{budget // 60}" if budget % 60 == 0 else f"{budget / 60:g}"
    return (mode_table.TOOL_LONG_CALL_RULE
            .replace("{BUDGET_MIN}", minutes)
            .replace("{YIELD_MS}", str(budget * 1000)))


def _resolve_row(word: str, rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The exposed row *word* names, by mode word first and then by alias."""
    for row in rows:
        if row["name"] == word:
            return row
    for row in rows:
        if word in row["aliases"]:
            return row
    return None


def _example(mode: str, plan: str, question: str, extra_flags: List[str]) -> str:
    """One worked example, spelled exactly as the tool receives it."""
    return json.dumps(
        {"mode": mode, "plan": plan, "question": question,
         "extra_flags": extra_flags},
        separators=(",", ":"), ensure_ascii=False,
    )


def _worked_examples(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """The four worked spellings, **selected by shape rather than by name** (rule f).

    Between them they cover every arity rule: the default mode with one plan, a
    mode declaring no positional (the three-empties spelling a flat "an empty
    plan is invalid" rule would forbid), a mode declaring a question as well as
    a plan, and an alias travelling unexpanded beside extra flags. Each is found
    by what the table says the mode *declares*, so a renamed mode changes the
    examples without an edit here. The only written words are
    ``modes.EXAMPLE_PLAN_PATH``, ``modes.EXAMPLE_QUESTION`` and
    ``_EXAMPLE_FLAG`` — and the aliased row is chosen among the modes the
    ``unattended`` table says accept that flag, so the example never teaches
    a call the mode's parser refuses.
    """
    plan_kind, question_kind = mode_table.POS_PLAN, mode_table.POS_QUESTION
    plan = mode_table.EXAMPLE_PLAN_PATH
    examples: List[str] = []

    fallback = _resolve_row(mode_table.DEFAULT_MODE, rows)
    if fallback:
        examples.append(_example(fallback["name"], plan, "", []))

    nullary = next((row for row in rows if not row["positionals"]), None)
    if nullary:
        examples.append(_example(nullary["name"], "", "", []))

    asking = next(
        (row for row in rows
         if plan_kind in row["positionals"] and question_kind in row["positionals"]),
        None,
    )
    if asking:
        examples.append(
            _example(asking["name"], plan, mode_table.EXAMPLE_QUESTION, [])
        )

    flag_site = unattended.row_for_site(_EXAMPLE_FLAG_SITE)
    if _EXAMPLE_FLAG not in flag_site.flags:
        raise RuntimeError(
            f"unattended site {_EXAMPLE_FLAG_SITE!r} no longer declares "
            f"{_EXAMPLE_FLAG}; the aliased worked example needs a flag its "
            "mode accepts"
        )
    aliased = next(
        (row for row in rows
         if row["aliases"] and plan_kind in row["positionals"]
         and row["name"] in flag_site.modes),
        None,
    )
    if aliased:
        examples.append(_example(
            aliased["aliases"][0], plan, "", [_EXAMPLE_FLAG],
        ))
    return examples


def _tool_description(host_id: str, rows: Sequence[Dict[str, Any]]) -> str:
    """The whole ``tools/list`` description, already joined.

    The lede (``modes.TOOL_DESCRIPTION_LEDE``), one line per exposed mode with
    its aliases and summary, the default-mode line, the outcomes line — after
    the mode block, so a model meets the exit codes in the same glance as the
    modes — and directly under it the ``user_next_step`` line
    (``modes.TOOL_USER_NEXT_STEP_LINE``: a lifted next-phase command is the
    user's to type, never a call; plans/next-phase-command.md §2.4) and its
    sibling ``modes.TOOL_NO_COMPOSED_COMMAND_LINE`` (those keys and
    ``resume_command`` are the only commands, never one the model composes)
    and ``modes.TOOL_NOTICE_LINE`` (a finished run's ``notice`` is relayed),
    then any host-keyed note, then the worked examples. Examples travel inside the
    description because a JSON-Schema fallback has nowhere to put an
    ``examples`` key a host would render.
    """
    lines = [mode_table.TOOL_DESCRIPTION_LEDE, "", "Modes:"]
    for row in rows:
        names = " | ".join([row["name"], *row["aliases"]])
        lines.append(f"  {names} — {row['summary']}")
    lines.append("")
    lines.append(f"The default mode is {mode_table.DEFAULT_MODE}.")
    lines.append("")
    lines.append(mode_table.TOOL_OUTCOMES_LINE)
    lines.append(mode_table.TOOL_USER_NEXT_STEP_LINE)
    lines.append(mode_table.TOOL_NO_COMPOSED_COMMAND_LINE)
    lines.append(mode_table.TOOL_NOTICE_LINE)
    note = _DESCRIPTION_NOTE.get(host_id)
    if note:
        writers = ", ".join(row["name"] for row in rows if row["writes_worktree"])
        lines.append("")
        lines.append(note.replace("{modes}", writers))
    waiting = long_call_rule(host_id)
    if waiting:
        lines.append("")
        lines.append(waiting)
    lines.append("")
    lines.append("Examples:")
    for example in _worked_examples(rows):
        lines.append(f"  {example}")
    return "\n".join(lines)


def _argument_descriptions(
    host_id: str, rows: Sequence[Dict[str, Any]], reserved: Dict[str, Any],
) -> Dict[str, str]:
    """One rendered description per tool property — exactly the four of rule 1.

    ``mode`` is described rather than enumerated: a fallback schema of
    ``{type, properties, required}`` has no ``enum`` to carry the vocabulary, so
    legality is the pre-spawn check against the table and the description is
    what conveys it.
    """
    words = [word for row in rows for word in (row["name"], *row["aliases"])]
    plan_modes = [
        row["name"] for row in rows if mode_table.POS_PLAN in row["positionals"]
    ]
    question_modes = [
        row["name"] for row in rows
        if mode_table.POS_QUESTION in row["positionals"]
    ]
    reserved_words = [
        *reserved["exact"],
        *reserved["named_exception"],
        *(f"{prefix}*" for prefix in reserved["prefixes"]),
    ]
    redundant = reserved["redundant"]
    restated = [
        *redundant["flags"],
        *(f"{name} {value}" for name, value in redundant["valued"].items()),
    ]
    sigil = _PLAN_SIGIL_CLAUSE.get(host_id, "")
    return {
        "mode": (
            f"One of: {', '.join(words)}. An alias travels unexpanded — do not "
            "expand it and do not add the flag it already contributes. Never "
            "empty. The default when the user named no mode is "
            f"{mode_table.DEFAULT_MODE}."
        ),
        "plan": (
            "The plan path, as exactly one argv element — never split, never "
            "shell-quoted, never escaped. Strip at most one layer of matching "
            f"quotes{sigil} from what the user typed and send the value. A path "
            "containing spaces needs no escaping. Empty (\"\") when and only "
            "when the mode declares no plan positional. Modes taking a plan: "
            f"{', '.join(plan_modes)}. Relative paths resolve against the "
            "workspace."
        ),
        "question": (
            "The free-text question, as exactly one argv element — newlines "
            "preserved verbatim, no truncation, no escaping. Empty (\"\") for "
            f"every mode except: {', '.join(question_modes) or '(none)'}. A "
            "question whose first character is \"-\" is refused, because the "
            "dispatcher's grammar would read it as a flag and end the "
            "positional scan; a question merely containing a flag is fine."
        ),
        "extra_flags": (
            "Additional dispatcher flags, one token per array element — "
            '["--view", "run"], never ["--view run"]. Use [] for none. Passed '
            "verbatim, in order, never reordered or deduplicated. This tool "
            f"composes these itself: {', '.join(reserved_words)}. Restating one "
            f"with the same meaning ({', '.join(restated)}) is fine: it is "
            "ignored, because the call already does what it says. Every other "
            'one is rejected — including their abbreviations and "=" forms.'
        ),
    }


def _redundant_flags(host_id: str) -> Dict[str, Any]:
    """The composed flags a call may restate as a no-op (racing re-run 2026-10-04).

    Derived from ``baked_flags`` and the parent parser, never typed: every
    spelling of a valueless composed flag (``--yes``/``-y``/
    ``--non-interactive``, ``--exec-subagents``) is the same action twice, and
    a valued one (``--host``) is the same only with this host's value, so it is
    listed with that value. ``--detach`` is composed for some modes only and is
    a refusal on the rest, ``--interactive`` contradicts ``--yes``, and the
    detach controls and ``--internal-*`` are never the caller's: none of them is
    here, so each stays refused.

    The drop is quiet (run 6, #10a/#14): there is no ``note``. The result used
    to name each dropped token ("leave them out next time"), and the host
    relayed it as "`--yes` was dropped from your flags" on every call of a user
    who typed ``--yes`` on purpose. Every flag here means exactly what the tool
    already does, so nothing changed that the host or the user needs to hear;
    a flag whose meaning would differ is refused, never dropped.
    """
    baked = _baked_flags(host_id)
    actions = {
        name: action
        for action in cli_common.build_parent_parser()._actions
        for name in action.option_strings
    }
    flags: List[str] = []
    valued: Dict[str, str] = {}
    index = 0
    while index < len(baked):
        action = actions[baked[index]]
        if action.nargs == 0:
            flags.extend(action.option_strings)
            index += 1
        else:
            valued[baked[index]] = baked[index + 1]
            index += 2
    return {"flags": flags, "valued": valued}


def _baked_flags(host_id: str) -> List[str]:
    """The flags ``execute`` appends to every dispatch, in composition order."""
    return ["--host", host_id, "--yes", "--exec-subagents"]


def _reserved_flags(host_id: str) -> Dict[str, Any]:
    """The flags ``extra_flags`` may not carry, generated from their owners.

    **The set is closed by construction: it is exactly the flags the plugin
    composes** — not a hand-picked list that happens to hold four entries today.
    Any flag ``execute`` appends to the dispatcher argv is reserved, so a model
    can never duplicate one, contradict one, or change behaviour by restating
    one.

    Two flags the plugin *also* composes are deliberately absent, and the reason
    is where they sit rather than a judgement: ``--frozen`` and ``--no-dev`` are
    **uv** arguments, composed before the ``--`` separator, while ``extra_flags``
    is appended after it and reaches only ``multi_llm.py``. A model sending
    ``--no-dev`` there cannot duplicate the uv-side flag or re-enable the
    dev-group sync; it hands the dispatcher an unknown option and gets exit 3.
    Reserving it would be reserving a name the plugin does not own on that side
    of the line.

    ``--interactive``/``-i`` are the one entry that is *not* a flag the plugin
    composes, so they ride beside the closure as a **named exception** rather
    than as a quiet addition: with ``--detach`` the dispatcher rejects the pair
    at parse time, and without it the forced model picker has no TTY on this
    route, so the flag has nothing behind it in any posture.
    """
    return {
        # Matched by normalized option head — split at the first `=`, take the
        # head — and then by argparse's own binding rule, because no parser in
        # this tree passes ``allow_abbrev=False``: `--ho`, `--ye` and `--det`
        # all bind to their full options, so a guard that scanned for exact
        # spellings was bypassable by typing three fewer characters.
        "exact": [
            "--host",
            "--yes", "-y", "--non-interactive",
            "--exec-subagents",
            "--detach", "--if-not-running", "--max-detached",
        ],
        "named_exception": ["--interactive", "-i"],
        "prefixes": [cli_common.INTERNAL_FLAG_PREFIX],
        # A long option that is a strict prefix of another real one. argparse
        # resolves an exact match before it considers any abbreviation, so
        # these never read as an abbreviation of something longer.
        "exact_options": sorted(cli_common._EXACT_OPTIONS),
        # Additive within schema v5: the subset of the above a call may
        # restate as a no-op, dropped quietly. A server whose surface lacks
        # the key refuses every one of them, as before.
        "redundant": _redundant_flags(host_id),
    }


def _uninstall_command(host_id: str) -> str:
    """``host_install.uninstall_line(host_id, SKILL_DIR_TOKEN)``, environment-free.

    Calling ``uninstall_line`` here leaked the builder's environment into the
    shipped file: it composes through ``cli_common.dispatcher_prefix``, which
    reads ``unattended.uv_run_args()`` — so a maintainer with
    ``MULTI_LLM_UV_RUN_ARGS=--no-sync`` exported shipped ``uv run --no-sync …``
    as every installation's uninstall line. Composed here from the
    ``DEFAULT_UV_RUN_ARGS`` constant in the prefix's own shape and quoting
    (``cli_common._quote``); ``test_cli_surface_emitter`` holds it equal to
    ``uninstall_line`` with the override unset, so the spelling keeps one owner.
    """
    base = Path(SKILL_DIR_TOKEN)
    flags = "".join(f" {flag}" for flag in unattended.DEFAULT_UV_RUN_ARGS)
    return (
        f"uv run{flags} --project {cli_common._quote(base)} -- "
        f"python {cli_common._quote(base / 'multi_llm.py')} "
        f"uninstall-host {host_id}"
    )


def _route_env(host_id: str) -> Dict[str, str]:
    """``dispatcher.route_env`` for *host_id*: ``{"name", "value"}`` (schema v5).

    The name is ``modes.ROUTE_ENV``. The value is the one member of
    ``modes.TOOL_ROUTES`` that is a key of the host's ``next_command`` cell,
    read through ``host_capabilities.entry`` — the source tree's
    ``hosts/capabilities.yaml``, no environment — so the surface stays a
    function of the source tree (plans/next-phase-command.md T2). None, or more
    than one, is a table drift: the build fails naming both tables rather than
    writing a guess, because a wrong value would print a plausible wrong
    command on this route.
    """
    cell = host_capabilities.entry(host_id).get("next_command") or {}
    members = [route for route in mode_table.TOOL_ROUTES if route in cell]
    if len(members) != 1:
        raise ValueError(
            f"host '{host_id}': expected exactly one modes.TOOL_ROUTES member "
            f"({', '.join(mode_table.TOOL_ROUTES)}) among its next_command keys "
            f"in hosts/capabilities.yaml ({', '.join(cell) or 'none'}), found "
            f"{len(members)}. Fix one table or the other; the surface's "
            "route_env is never guessed."
        )
    return {"name": mode_table.ROUTE_ENV, "value": members[0]}


def _user_command(host_id: str) -> Dict[str, str]:
    """``dispatcher.user_command``: the route's template and its args slot."""
    route = _route_env(host_id)["value"]
    return {
        "template": host_capabilities.next_command_template(host_id, route),
        "args_slot": next_phase.ARGS_SLOT,
    }


def cli_surface(host_id: str) -> Dict[str, Any]:
    """Everything a host's plugin needs to know about the CLI, generated.

    The plugin reads this at load and holds no mode word, no alias, no summary
    text and no flag name of its own — the same rule that makes the preflight
    message ``hosts/uv-missing.txt`` rather than a JS string literal. Since
    schema v4 that includes the model-facing definition itself: the tool's
    name, its whole joined description and its four argument descriptions are
    rendered here, from the table and the authored ``modes.py`` constants
    beside it, and a host passes them through unchanged.

    *host_id* is the host this surface is a distribution for (``opencode``,
    ``codex``); it is the only input besides the imported tables. Every field
    below is generated from those tables, and none is read out of a skill tree
    on disk or out of the builder's environment — the surface is a function of
    the source tree alone (D3e). The one field that *was* read from disk — the
    three resolved capability-verb sentences — is gone. The plugin never read
    them from here: it asks ``util instructions <mode> --host <id>`` at run
    time, which is the single owner of that text, and a second copy in a file
    nothing consumes is a copy that can only go stale. One field reads data
    rather than a module: ``dispatcher.route_env``'s value comes from the
    source tree's ``hosts/capabilities.yaml`` (the host's ``next_command``
    keys) through ``host_capabilities.entry`` — still the source tree, still
    no environment.
    """
    rows = _exposed_modes()
    reserved = _reserved_flags(host_id)
    return {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "host": host_id,
        # The model-facing definition, whole (schema v4). A host registers the
        # tool under `tool_name`, shows `tool_description` as its description
        # and `argument_descriptions[<property>]` as each property's, byte for
        # byte, and composes none of them.
        "tool_name": mode_table.TOOL_NAME,
        "tool_description": _tool_description(host_id, rows),
        "argument_descriptions": _argument_descriptions(host_id, rows, reserved),
        # The one runnable spelling of "remove the now-dead file shim", carried
        # as data so the plugin's collision warning composes nothing: it is
        # ``host_install.uninstall_line()``'s string, the same one ``--check``
        # prints for the both-routes posture, so the installer's note and the
        # warning can never name two different commands — or, as the warning
        # did, a placeholder host and a bare ``multi_llm.py`` nobody can run.
        # ``{SKILL_DIR}`` stays a token for exactly the reason
        # :func:`command_body` leaves one: the package lands in whatever cache
        # directory npm gives it, and only the plugin knows where that is.
        # See :func:`_uninstall_command` for why it is not *called*.
        "uninstall_command": _uninstall_command(host_id),
        # The single owner of "no mode flag means review-plan".
        "default_mode": mode_table.DEFAULT_MODE,
        "description_lede": mode_table.TOOL_DESCRIPTION_LEDE,
        # The lede's sibling: the four exit codes and what exit 3 decides,
        # pushed into the description after the mode block.
        "outcomes_line": mode_table.TOOL_OUTCOMES_LINE,
        # The exit-3 `message`, as a template plus the clauses its tokens are
        # filled with. The plugin selects and substitutes; it authors nothing,
        # not even a fallback — the same rule that keeps the uv preflight text
        # in `hosts/uv-missing.txt`.
        "needs_human_message": dict(mode_table.NEEDS_HUMAN_MESSAGE),
        # The one `options` entry that is not a retry.
        "needs_human_option": mode_table.NEEDS_HUMAN_OPTION,
        # The sentence a server sets as `no_command_reason` beside
        # `user_next_step: null` on a failed or needs-human result without a
        # `resume_command`. Additive within v5: a server whose surface lacks it
        # still sets the null and omits the sentence.
        "no_command_reason": mode_table.NO_COMMAND_REASON,
        # The OpenCode tool part's title at a stop (``tool.execute.after``),
        # which ``opencode run`` prints without the model. Additive within v5:
        # a plugin whose surface lacks it sets no title.
        "stop_title": dict(mode_table.STOP_TITLE),
        "modes": rows,
        "positional_kinds": {
            "plan": mode_table.POS_PLAN,
            "question": mode_table.POS_QUESTION,
        },
        "reserved_flags": reserved,
        # The flags ``execute`` appends to every dispatch, in composition order.
        # ``--yes`` is unconditional: the ``--interactive`` substitution is
        # retired and the flag is refused before the spawn.
        "baked_flags": _baked_flags(host_id),
        "detach_flag": "--detach",
        "uv": {
            "env": unattended.UV_ENV,
            "run_args_env": unattended.UV_RUN_ARGS_ENV,
            "default_run_args": list(unattended.DEFAULT_UV_RUN_ARGS),
            # Element for element, in probe order. A JS reimplementation that
            # drops the two ``.exe`` entries recreates precisely the Windows
            # failure this route exists to remove: they are what the official
            # installer creates, and Git Bash is the only supported Windows
            # shell.
            "candidates": list(unattended.UV_CANDIDATES),
            "missing_message_file": unattended.UV_MISSING_FILE.replace(
                os.sep, "/"
            ),
        },
        "dispatcher": {
            "script": "multi_llm.py",
            "result_marker": cli_common.RESULT_MARKER,
            # Read from cli_common's constants, never as JS literals.
            "exit_codes": {
                "ok": cli_common.EXIT_OK,
                "error": cli_common.EXIT_ERROR,
                "nothing_to_do": cli_common.EXIT_NOTHING_TO_DO,
                "needs_human": cli_common.EXIT_NEEDS_HUMAN,
            },
            # The `state` word the result line carries for each exit, keyed
            # like `exit_codes`. The exit-3 message claims "did not complete"
            # only when the result line's state is the needs-human one.
            "states": {
                "ok": cli_common._STATE_BY_EXIT[cli_common.EXIT_OK],
                "error": cli_common._STATE_BY_EXIT[cli_common.EXIT_ERROR],
                "nothing_to_do": cli_common._STATE_BY_EXIT[
                    cli_common.EXIT_NOTHING_TO_DO
                ],
                "needs_human": cli_common._STATE_BY_EXIT[
                    cli_common.EXIT_NEEDS_HUMAN
                ],
                # The two override words exit 2 also carries, keyed by the word
                # rather than by an exit they share with ``nothing_to_do``.
                # They are here because ``_STATE_BY_EXIT``'s own comment makes
                # this dict the place a host learns the vocabulary, and because
                # a predicate that decides what an ``ok: true`` state owes the
                # model (``rowNeedsInstructions``) must be able to name every
                # word it can be handed without spelling one in JS.
                "running": cli_common.STATE_RUNNING,
                "died": cli_common.STATE_DIED,
                # The exit-0 override word: a finished run with at least one
                # failed model slot (plan §5). Terminal, ``next: null``, so it
                # owes no further instructions and is not an incomplete run.
                "partial": cli_common.STATE_PARTIAL,
            },
            # The one exit 3 that is a *completed* run: a review held for a
            # decision. Read from the launch, because its result line carries
            # the same state as every other exit 3.
            "review_gate": dict(mode_table.REVIEW_GATE),
            # ``--wait`` is a store_true and the plugin passes the bare flag,
            # never a ``--timeout``: the duration is the dispatcher's to
            # resolve. What that resolution *is* changed underneath this row —
            # ``_resolve_wait_timeout`` now takes the project's
            # ``progress.poll_s`` (90 s as shipped, so a default install polls
            # exactly as it always did) and then clamps the resolved value to
            # the named host's ``timeout_ceiling_s`` less ``WAIT_CAP_MARGIN_S``.
            #
            # The clamp therefore *does* bind this route, whatever was once
            # written here: ``baked_flags`` above puts ``--host`` on every
            # dispatch and the poll is composed through the same function, so
            # the runtime ``--host`` the clamp reads is never absent. It is
            # accepted rather than routed around, even though
            # ``timeout_ceiling_s`` is documented as the cap on a *foreground
            # shell command* and binds no tool execution: the plugin's poll is
            # a child spawned with the call's remaining budget as its timeout
            # and killed by process group the moment the call is cancelled, so
            # a long cadence costs neither abort latency nor budget overshoot
            # nor a killed child — it costs a thinned progress relay, one
            # update per poll, and the clamp is the one thing bounding that.
            #
            # The alternative, weighed and refused, was to compute a number here
            # from the host entry this surface is already built against and bake
            # it in as ``--timeout <n>``. That keeps the duration Python-owned
            # too, but it makes this the *second* place that computes it: it
            # would have to track the margin, the tightest-cap fallback for an
            # unrouted poll and the ``poll_s`` floor by hand, and the two would
            # drift the first time one of them moved.
            "poll": ["status", "--view", "run", "--wait"],
            "abort": ["status", "--abort"],
            "instructions": ["util", "instructions"],
            # The one lookup a server makes after an exit 3, to hand the user a
            # command they can type: `util next-command <plan> --blocked
            # <mode> --host <id>` prints the same-phase `--resume` command in
            # this route's spelling and carries it in the result line's
            # `user_resume`. Additive within schema v5 (no bump): a server that
            # finds the key absent skips the lookup, so an older surface only
            # loses the command, never the result.
            "resume": {
                "argv": ["util", "next-command"],
                "option": "--blocked",
                "result_key": "user_resume",
            },
            # How a user types a command on this route: the host's
            # `next_command` template for `route_env.value`, and the slot the
            # arguments fill. A needs-human result's redacted dispatcher
            # commands are re-spelled through it (each row's `user_flag`, the
            # plan as the call typed it, `next_phase.quote_plan_arg`'s quoting)
            # instead of being replaced by a removal note. Additive within v5:
            # a server whose surface lacks it keeps the note.
            "user_command": _user_command(host_id),
            "host_option": "--host",
            "run_id_option": RUN_ID_OPTION,
            # The variable the plugin sets to the model the host session is
            # itself running on, read back off the OpenCode session's newest
            # assistant message, or codex's per-call turn metadata `model`.
            # Python owns the name (`provider_registry`), because
            # `resolve_subagent_runner` is the only thing that reads it: the
            # plugin's job is to *observe* the session model and the ladder's is
            # to decide what that means. Without this the run dispatches the
            # host's credential-free floor while the picker two inches above the
            # prompt says something else.
            "session_model_env": provider_registry.HOST_SESSION_MODEL_ENV,
            # The route signal (schema v5): a server sets `env[name] = value`
            # on every child it spawns, the polls included, so the dispatcher
            # spells a finished phase's `user_next_step` command the way this
            # route types it. The same precedent as `session_model_env` — a
            # Python-owned name the server reads from here — plus the value,
            # which is picked from `modes.TOOL_ROUTES` by the host's
            # `next_command` cell. It picks a spelling, never what runs.
            "route_env": _route_env(host_id),
            # The block lines a result's `user_next_step` already says (fix
            # round 27): a server drops a line that is one of these markers,
            # or starts with one and a space, from the `output` of a result
            # that carries the key, so the model reads each sentence once.
            # Additive within v5: a server whose surface lacks it keeps every
            # line.
            "next_step_markers": list(next_phase.CARRIED_BLOCK_MARKERS),
            # The result-line key naming the repository root a command typed
            # in a subdirectory ran from (fix round 27): a server names a plan
            # typed from that root by the root's file for its polls, its abort
            # and its options, only when a launch's line says so. Additive
            # within v5: a server whose surface lacks it names the plan from
            # its anchor.
            "ran_from_root_key": cli_common.RAN_FROM_ROOT_KEY,
            # A `next` carrying one of these is a *mechanical* continuation —
            # the same phase finishing itself off — so `execute` follows it
            # internally under its own cap rather than handing it back for a
            # round of turns. Anything else is returned to the model as a tool
            # call. Both markers are flags the dispatcher already owns; the
            # plugin never invents a continuation of its own.
            "mechanical_continuations": ["--reaggregate", "--consolidate"],
            # The banner is the ONLY place a launch reports its run id — the
            # result payload has no such key, and the launch's `next` carries
            # `--run-id` for `ask` runs alone. So the field names ship as data
            # and the plugin parses them rather than spelling them in JS.
            "banner": {
                "headline": detach.BANNER_HEADLINE,
                "fields": list(detach.BANNER_FIELDS),
                "run_id_field": "run_id",
                "log_field": "log",
            },
        },
        # The executor's structured refusals, paired with the capability field
        # each is a verdict about. Generated so the plugin can name both without
        # holding a reason-code table of its own.
        "nested_exec_reasons": dict(subagent_executor.NESTED_EXEC_FIELDS),
        # The pre-spawn refusals a package server returns in its ordinary
        # envelope instead of spawning anything. Templates: `{name}` tokens
        # are replaced by plain substitution, never `str.format`.
        #
        # D3b: a `writes_worktree` mode under a read-only or unreadable
        # session posture. `{mode}` is the word as sent, `{posture}` the
        # posture word read or `posture_unreadable`. `lever_env` is read from
        # the server's own environment and lifts the refusal only when it is
        # exactly `lever_value`.
        "write_posture_refusal": {
            "template": mode_table.WRITE_POSTURE_REFUSAL,
            "posture_unreadable": mode_table.WRITE_POSTURE_UNREADABLE,
            "lever_env": mode_table.WRITE_POSTURE_LEVER_ENV,
            "lever_value": mode_table.WRITE_POSTURE_LEVER_VALUE,
        },
        # D3f: no workspace-anchor candidate answered. `{probed}`.
        "anchor_unresolved": mode_table.ANCHOR_UNRESOLVED,
        # D3a: the login shell does not resolve. `{shell}`, `{platform}`,
        # `{path}` (the PATH searched).
        "shell_missing": mode_table.SHELL_MISSING,
        # `util instructions` stdout, as the dispatcher writes it: the head
        # banner opens with `banner_opening` and is dropped before a runbook is
        # attached (its character count describes the unscrubbed core); the
        # core closes with `core_end_marker` + " " + the mode word, and a
        # stdout lacking that line is a partial payload, never attached.
        "runbook": {
            "core_end_marker": multi_llm.CORE_END_MARKER,
            "banner_opening": multi_llm.TRUNCATION_BANNER_OPENING,
        },
    }


def cli_surface_json(host_id: str) -> str:
    """:func:`cli_surface` serialized exactly as every host's build writes it."""
    return json.dumps(cli_surface(host_id), indent=2, sort_keys=False) + "\n"
