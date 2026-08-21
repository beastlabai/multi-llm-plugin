#!/usr/bin/env python3
"""The reader for ``hosts/capabilities.yaml`` (D5b).

Three things live here, and nothing else:

* **Loading and validating** the capability data, cached per base directory
  because ``status`` must stay sub-second and every mode pays this on the way in.
* **Resolving a host id** — ``--host <id>`` (baked into a shim at install time),
  else ``$MULTI_LLM_HOST``, else ``claude-code``. An unknown id is exit 3 listing
  the ones that exist; Python **never sniffs the environment to guess its host**.
* **Rendering the sentences a host's capability decides** — the three verb
  sentences a shim bakes in (one fixed phrasing per enum value) and, on the same
  principle, the recovery instruction a failed batch prints (:func:`retry_instruction`,
  addressed by ``multi_turn`` to whoever that host leaves able to follow it).
  This is the whole point of the indirection: the capability is resolved to one
  concrete imperative *here*, in Python, so the driving agent resolves nothing
  and the matrix is never on an agent read path.

``providers.yaml`` ``driver.*`` is user config and always wins over anything here;
this file states *facts about hosts* and ships with the skill.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # normal import (utils is a package)
    from .cli_common import skill_dir
    from .unattended import NeedsHuman
except ImportError:  # pragma: no cover - direct script runs with utils/ on path
    from cli_common import skill_dir  # type: ignore
    from unattended import NeedsHuman  # type: ignore


#: Relative to the skill root. Ships inside the skill so a shim can name the
#: instruction core and the dispatcher by path from one installed artifact.
CAPABILITIES_FILE = os.path.join("hosts", "capabilities.yaml")
SCHEMA_FILE = os.path.join("schemas", "host_capabilities.schema.json")

#: The reference host. An omitted ``--host`` reproduces pre-capability-data
#: behavior exactly, which is what keeps the flag strictly additive.
DEFAULT_HOST = "claude-code"

#: Env fallback, for a host that invokes ``multi_llm.py`` directly rather than
#: through a rendered shim. Explicitly *set* by a user or an installer — never
#: detected.
HOST_ENV = "MULTI_LLM_HOST"

#: The closed verb vocabulary, identical in all three places it appears: the
#: wording in ``instructions/*.md``, the fields below, and the three shim slots.
VERBS: Tuple[str, ...] = ("subagents", "ask", "detach")

#: Verb -> the wording the instruction core uses for it. Asserted against the
#: core's own text by ``tests/test_instruction_portability.py``.
VERB_WORDING: Dict[str, str] = {
    "subagents": "spawn subagents",
    "ask": "ask the user",
    "detach": "run detached",
}

#: The profiles the Python-side executor needs from a runner. A runner that can
#: only ``generate`` cannot stand in for a subagent (D3e).
EXECUTOR_PROFILES: Tuple[str, ...] = ("agent-readonly", "agent-write")

_CACHE: Dict[str, Dict[str, Any]] = {}


class UnknownHost(NeedsHuman):
    """An unrecognized ``--host`` id: exit 3, never a silent default."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load(base: Optional[str] = None) -> Dict[str, Any]:
    """The parsed capability document, cached per base directory."""
    root = str(base) if base else str(skill_dir())
    cached = _CACHE.get(root)
    if cached is not None:
        return cached
    import yaml

    path = os.path.join(root, CAPABILITIES_FILE)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
    except OSError as exc:
        raise NeedsHuman(
            f"ERROR: host capability data is missing or unreadable: {path}\n"
            f"       ({exc})\n"
            "       The install looks corrupt or partial; reinstall the skill."
        ) from exc
    _CACHE[root] = document
    return document


def clear_cache() -> None:
    """Drop the cache (tests that point at a fixture tree)."""
    _CACHE.clear()


def host_ids(base: Optional[str] = None) -> Tuple[str, ...]:
    return tuple(load(base).get("hosts", {}))


def entry(host_id: str, base: Optional[str] = None) -> Dict[str, Any]:
    """One host's capability profile, or exit 3 naming the ids that exist."""
    hosts = load(base).get("hosts", {})
    if host_id not in hosts:
        raise UnknownHost(
            f"ERROR: unknown host '{host_id}'.\n"
            f"       Known hosts: {', '.join(sorted(hosts))}\n"
            "       --host names an entry in hosts/capabilities.yaml; it is baked "
            "into a shim by `install-host`, not guessed at run time."
        )
    return hosts[host_id]


def resolve_host_id(
    argv: Optional[Sequence[str]] = None,
    *,
    explicit: Optional[str] = None,
) -> str:
    """``--host`` → ``$MULTI_LLM_HOST`` → :data:`DEFAULT_HOST`.

    Nothing here inspects the environment for *evidence* of a host — an env var a
    user or installer sets is a declaration, a sniffed process tree would be a
    guess (D5b: "Python never sniffs the environment to guess its host").
    """
    if explicit:
        return explicit
    if argv:
        try:
            from .cli_common import flag_value
        except ImportError:  # pragma: no cover
            from cli_common import flag_value  # type: ignore
        found = flag_value(argv, "--host")
        if found:
            return found
    return os.environ.get(HOST_ENV) or DEFAULT_HOST


# ---------------------------------------------------------------------------
# Derived facts consumers ask for
# ---------------------------------------------------------------------------

def max_concurrent_subagents(host_id: str, base: Optional[str] = None) -> int:
    """The host-side leg of D3i's three-way minimum."""
    return int(entry(host_id, base)["max_concurrent_subagents"])


def executor_capable_runners(
    host_id: str, base: Optional[str] = None
) -> List[str]:
    """Runners in this host's entry that can actually back the executor (D3e)."""
    runners = entry(host_id, base).get("runners", {})
    return sorted(
        name
        for name, spec in runners.items()
        if set(spec.get("profiles", ())) & set(EXECUTOR_PROFILES)
    )


def require_executor_runner(host_id: str, base: Optional[str] = None) -> None:
    """Refuse ``--exec-subagents`` on a host with no executor-capable runner.

    A flag that silently does nothing is worse than a refusal: the phase would
    stop at a dispatch marker waiting for a subagent nobody is going to start,
    until the host's shell cap kills it. The list comes from the data, so adding a
    runner is a YAML edit rather than a code change.
    """
    if executor_capable_runners(host_id, base):
        return
    host = entry(host_id, base)
    raise NeedsHuman(
        f"ERROR: --exec-subagents is not available on host '{host_id}'.\n"
        "       The executor needs a runner declaring the "
        f"{' or '.join(EXECUTOR_PROFILES)} profile, and this host's entry "
        f"declares none (runners: {', '.join(sorted(host.get('runners', {}))) or 'none'}).\n"
        "       Configure one under `driver.subagent_runner` in providers.yaml, or "
        "drop --exec-subagents and let this host's own subagent mechanism run the "
        "phase."
    )


#: The three lists every ``env_guards`` block declares (D3c/D5b).
ENV_GUARD_LISTS: Tuple[str, ...] = ("guards", "session_scoped", "passthrough")


def env_guards(host_id: str, base: Optional[str] = None) -> Dict[str, List[str]]:
    """This host's declared child-env guards (D3c/D5b), always three lists.

    ``guards`` are the nested-session markers a dispatched runner must not
    inherit, ``session_scoped`` the ids that would make a child join or overwrite
    the parent's session, and ``passthrough`` the names the host *sanctions* for
    the runner (a credential the runner legitimately needs), which therefore
    survive the strip. Literal names only — the schema rejects a ``*``, because a
    prefix sweep would take the runner's own credentials with it.
    """
    declared = entry(host_id, base).get("env_guards") or {}
    return {
        name: [str(value) for value in (declared.get(name) or ())]
        for name in ENV_GUARD_LISTS
    }


#: The three independent ways a host can stop a dispatched runner (D3c), in the
#: order every consumer reports them.
NESTED_EXEC_LEGS: Tuple[str, ...] = ("spawn", "network", "credentials")


def nested_exec(host_id: str, base: Optional[str] = None) -> Dict[str, str]:
    """This host's declared ``nested_exec`` verdict, always the three legs + notes.

    ``spawn`` is "may this host's shell start a third-party agent CLI at all",
    ``network`` is "can that CLI reach its provider API from inside the sandbox",
    ``credentials`` is "can it read its own auth store" — each
    ``allowed | needs-config | denied | unknown``, with ``notes`` naming the flag
    or setting that grants it. A missing or unreadable block reads as ``unknown``
    on every leg rather than as ``allowed``: the executor's preflight probes an
    ``unknown`` leg with a throwaway dispatch (D3c), and defaulting to ``allowed``
    would silently skip exactly that probe.
    """
    declared = entry(host_id, base).get("nested_exec") or {}
    states = {
        leg: str(declared.get(leg) or "unknown") for leg in NESTED_EXEC_LEGS
    }
    states["notes"] = str(declared.get("notes") or "").strip()
    return states


def is_multi_turn(host_id: str, base: Optional[str] = None) -> bool:
    """Does this host leave the driving agent able to act on a command's output?

    ``multi_turn`` is one boolean over two mechanisms — a further tool call
    inside the same invocation, or a conversation the next invocation resumes —
    because every consumer asks the same single question: *after this command
    returns, is an agent still there to read it and do something?* Either
    mechanism answers that yes; a host that answers no ends its invocation with
    the command it ran, so the output is read by a human and by nobody else.
    """
    return bool(entry(host_id, base)["multi_turn"])


def is_free_text_ask(host_id: str, base: Optional[str] = None) -> bool:
    return entry(host_id, base)["ask"] == "free-text"


def ask_class(host_id: str, base: Optional[str] = None) -> str:
    return str(entry(host_id, base)["ask"])


# ---------------------------------------------------------------------------
# Verb resolution — one fixed phrasing per enum value
# ---------------------------------------------------------------------------
# These strings ARE the shim's answer to each capability verb. `install-host`
# fills `{{spawn_subagents}}` / `{{ask_user}}` / `{{run_detached}}` from them at
# the same moment it bakes `{SKILL_DIR}` and `--host <id>`, and the Claude Code
# shim — the one nobody renders — is asserted byte-identical to what they produce
# for the `claude-code` entry. One phrasing per value, no interpolation beyond the
# tool name, so "the reference host cannot drift from the data that governs every
# other host" is a byte comparison rather than a review note.

_SUBAGENTS = {
    "native": (
        "**Spawn subagents** — call your host's own subagent tool `{tool}`: pass "
        "`task.prompt` verbatim, take the role from `task.subagent_type` when your "
        "tool needs one, and write the JSON it returns to `task.output_path`. "
        "Never pass `--exec-subagents` on this host."
    ),
    "executor": (
        "**Spawn subagents** — run the dispatcher with `--exec-subagents`: it "
        "dispatches every materialized task itself and continues past each marker "
        "in the same process. Never spawn a subagent by hand on this host."
    ),
    "none": (
        "**Spawn subagents** — run the dispatcher with `--exec-subagents`. This "
        "host has no subagent mechanism of its own, so a run that cannot use the "
        "executor is refused rather than degraded."
    ),
}

_ASK = {
    "structured": (
        "**Ask the user** — call your host's own structured question tool "
        "`{tool}`, then copy the selected option's command verbatim. Never compose "
        "a flag from the user's words."
    ),
    "free-text": (
        "**Ask the user** — print the numbered option block exactly as the "
        "dispatcher rendered it and copy the selected option's command verbatim, "
        "and only under `--interactive`. Otherwise the documented `--yes` default "
        "applies and nothing is asked."
    ),
    "none": (
        "**Ask the user** — never. This host has no reliable user channel, so "
        "every gate resolves from its documented `--yes` default and "
        "`--interactive` is exit 3."
    ),
}

_DETACH = {
    "native": (
        "**Run detached** — use your host's own background mechanism `{tool}`, "
        "redirecting stdout and stderr to the phase dir's `orchestrator-run.log`. "
        "Never pass `--detach` on this host."
    ),
    # R4 (2026-08-21): the previous wording ended "the foreground stays legal
    # for short calls", and codex-apply-executor read that as permission to skip
    # `--detach` entirely — measured as a miss on three consecutive host-lane
    # runs (20260820T093838Z, 20260821T020823Z, 20260821T024227Z), each time
    # leaving no `.status.json` run block for the row to grade. The permission
    # was accurate and is kept, but it is no longer the clause the reader ends
    # on: `--detach` is now stated as the default for every fan-out mode, and
    # the foreground exception is narrowed to the calls it was always meant for.
    "python": (
        "**Run detached** — pass `--detach` (which implies `--yes`) on every "
        "fan-out mode, then poll `status <plan> --view run`. Do this even when "
        "you expect the run to be quick: a fan-out started in the foreground "
        "cannot be handed back to once your turn ends. Only a single-model, "
        "non-fan-out call may stay in the foreground. Never combine `--detach` "
        "with a host background mechanism — pick one."
    ),
    "required": (
        "**Run detached** — pass `--detach` (which implies `--yes`) on every "
        "fan-out mode and poll `status <plan> --view run`. This host caps a single "
        "shell command below a fan-out's runtime, so no foreground fan-out path is "
        "offered."
    ),
}

_TEMPLATES = {"subagents": _SUBAGENTS, "ask": _ASK, "detach": _DETACH}

#: The template slot each verb fills in ``hosts/<host>/shim.md.tmpl`` (D7 owns the
#: templates; the slot names are fixed here so both sides cannot drift).
VERB_SLOTS: Dict[str, str] = {
    "subagents": "{{spawn_subagents}}",
    "ask": "{{ask_user}}",
    "detach": "{{run_detached}}",
}


def verb_sentence(host_id: str, verb: str, base: Optional[str] = None) -> str:
    """The one sentence this host's shim carries for *verb*."""
    if verb not in VERBS:
        raise KeyError(
            f"unknown capability verb {verb!r}; the vocabulary is closed at "
            f"{VERBS}"
        )
    host = entry(host_id, base)
    value = host[verb]
    template = _TEMPLATES[verb][value]
    if "{tool}" not in template:
        return template
    tool = (host.get("native_tool") or {}).get(verb)
    if not tool:
        raise NeedsHuman(
            f"ERROR: host '{host_id}' resolves the *{VERB_WORDING[verb]}* verb to "
            f"'{value}', which names the host's own tool, but its entry carries no "
            f"native_tool.{verb}. A shim cannot be rendered from an incomplete "
            "profile."
        )
    return template.format(tool=tool)


def verb_sentences(host_id: str, base: Optional[str] = None) -> List[str]:
    """All three sentences, in the closed verb order."""
    return [verb_sentence(host_id, verb, base) for verb in VERBS]


def verb_block(host_id: str, base: Optional[str] = None) -> str:
    """The three sentences as the markdown list a shim embeds."""
    return "\n".join(f"- {line}" for line in verb_sentences(host_id, base))


# ---------------------------------------------------------------------------
# Recovery wording — a failed run's instruction goes to whoever can follow it
# ---------------------------------------------------------------------------
# Same fix, two addressees, chosen by ``multi_turn``. The resolution happens
# *here*, in Python, from the ``--host`` id the shim baked into the command that
# is failing — the same rule as the verb sentences, and for the same reason: the
# agent reads a finished sentence, never the matrix. Telling the agent of a host
# with no follow-up turn to "re-invoke to retry" is an instruction nobody in that
# run is able to follow, which is how a failed batch becomes a silently abandoned
# one; naming the human is the honest form of the same fix.

_RETRY_LEAD: Dict[bool, str] = {
    True: (
        "Re-invoke to retry only the failures ({failed}) — completed tasks are "
        "skipped by their result files:"
    ),
    False: (
        "This host ends its run with the command that just failed, so no agent "
        "turn follows this output — a human has to act on it. Re-invoke the "
        "phase to retry only the failures ({failed}); completed tasks are "
        "skipped by their result files:"
    ),
}

#: Printed when the caller has no resolved command to name. The retry *is* the
#: same invocation, so "the same command" is always true, on either host.
_RETRY_FALLBACK = "  (re-run the same command; completed tasks are skipped)"


def retry_instruction(
    host_id: str,
    *,
    failed_ids: str = "",
    retry_command: Optional[str] = None,
    base: Optional[str] = None,
) -> List[str]:
    """The retry block a failed batch prints, addressed to whoever can act.

    Never raises. An unknown host id or unreadable capability data is a broken
    install, and dropping the retry command over it would cost a caller more
    than the wrong addressee does: the fallback is the reference host's answer,
    which is the wording this path printed before the field had a consumer.
    """
    try:
        multi_turn = is_multi_turn(host_id, base)
    except Exception:  # pragma: no cover - unknown host / unreadable data
        multi_turn = True
    lead = _RETRY_LEAD[multi_turn].format(failed=failed_ids or "none recorded")
    return [lead, f"  {retry_command}" if retry_command else _RETRY_FALLBACK]
