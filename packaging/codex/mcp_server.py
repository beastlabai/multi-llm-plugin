#!/usr/bin/env python3
"""multi-llm's codex plugin server: one MCP tool over stdio, stdlib only.

WHAT THIS FILE IS. The whole tool route on codex. It is shipped as
``<plugin_root>/mcp/server.py`` and started by codex from the plugin's
``.mcp.json`` as ``python3 ./mcp/server.py``. It speaks newline-delimited
JSON-RPC 2.0 on stdin/stdout, lists exactly one tool, and on ``tools/call``
composes the dispatcher argv, spawns it through a login shell in the user's
workspace, supervises a detached run to a terminal result, and returns one JSON
envelope. It is a port of the OpenCode plugin's ``execute()``
(``packaging/opencode/src/index.mjs``) state for state, except at the two edges
``plans/codex-native-plugin.md`` D3c re-decides: a claimed run whose budget
expires is handed off **alive** (never aborted), and a pinned ``status`` re-attach
supervises.

RULES EVERY LINE HERE KEEPS (plan D3):

  1. **Stdlib only, one file, parses at Python 3.9.** The server must start when
     ``uv`` is absent so it can *report* that absence (rule 7); a dependency, or a
     second module, is one more thing whose absence empties the tool list.
  2. **No word of its own.** Every mode word, alias, flag, exit code, state word
     and message template is read from ``dist/cli-surface.json`` (schema v4). The
     only words held here are the JSON-RPC/MCP vocabulary, the envelope's key
     names, the anchor_source spellings (``roots``, ``turn-metadata``,
     ``env:<NAME>``), codex's turn-metadata keys and posture words, the
     degraded definition (the tool name and one sentence) and the login
     shell's constant ``-c`` string.
  3. **Fd 1 carries JSON-RPC or nothing.** At start-up fd 1 is duplicated to a
     private descriptor only the locked writer uses, and fd 1 and ``sys.stdout``
     are pointed at stderr. Nothing here prints.
  4. **Never die at start, never throw out of a call.** A missing or mismatched
     surface lists the tool under the degraded definition and refuses every call;
     every failure inside a call is an envelope with ``ok: false``.

THREADING (D3c). The main thread owns stdin and answers everything except the
work of a ``tools/call``, which runs on one worker thread; a second call while
one is in flight is refused ``-32000``. Cancellation (``notifications/cancelled``
for the in-flight id, or stdin EOF) crosses to the worker as a per-call event.
One lock serializes every write to the protocol descriptor.
"""

from __future__ import annotations

import codecs
import collections
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

# ---------------------------------------------------------------------------
# Where we are (D2): resolved from __file__, never from the process cwd, which
# is the plugin's install location rather than the user's project (R11).
# ---------------------------------------------------------------------------

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = PLUGIN_ROOT / "skills" / "multi-llm"
SURFACE_PATH = PLUGIN_ROOT / "dist" / "cli-surface.json"
MCP_JSON_PATH = PLUGIN_ROOT / ".mcp.json"
PLUGIN_MANIFEST_PATH = PLUGIN_ROOT / ".codex-plugin" / "plugin.json"

#: Bumped in lockstep with ``packaging/common/surface.py``'s
#: ``SURFACE_SCHEMA_VERSION``; ``test_surface_schema_version_matches`` compares
#: the two constants, never a literal.
SUPPORTED_SURFACE_VERSION = 4

#: The longest a cancellation may sit unobserved (D3c). It is met by the
#: worker's cancel-event watch in ``run_child`` (a 20 ms wait on the event, then
#: the process-group kill), NOT by killing poll children at this bound as the
#: plan's D3c first wrote. Changed deliberately: codex declares
#: ``timeout_ceiling_s: 0``, so the dispatcher's ``--wait`` resolves to its 90 s
#: default, and a 10 s kill would discard every poll before it returned —
#: every mid-run ``progress_lines`` relay with it. A poll child's timeout is the
#: remaining budget alone.
CANCEL_LATENCY_BOUND_S = 10

#: A2b's shipped ``tool_timeout_sec``, used only when the sibling ``.mcp.json``
#: is missing, unreadable, not JSON, or carries no cap. Never unbounded.
DEFAULT_TOOL_TIMEOUT_S = 900

#: The login shell's ``-c`` string (D3a), a constant with no interpolation of
#: any kind. The anchor travels as ``$1`` and the dispatcher argv as ``"$@"``,
#: so model text reaches the child as positional parameters bash never re-parses.
LOGIN_SHELL_SCRIPT = 'cd -- "$1" && shift && exec "$@"'

#: The shell D3a names, looked up on the server's own PATH, then the fallback.
SHELL_NAME = "bash"
FALLBACK_SHELL = "/bin/bash"

#: SIGTERM → SIGKILL grace for every kill, and the drain-thread join bound.
KILL_GRACE_S = 5
#: The signal-free abort dispatch's own timeout (rule f), never the remainder.
ABORT_DISPATCH_TIMEOUT_S = 30
#: Held back from the cap so the handoff result is written while the host is
#: still listening: result construction, the runbook fetch and one teardown.
RESERVE_MAX_S = 20
RESERVE_FRACTION = 0.1
#: How long a ``roots/list`` round trip may take before that rung is skipped.
ROOTS_TIMEOUT_S = 5
#: Mechanical continuations followed per call (the reference's cap).
DEFAULT_CONTINUATION_CAP = 8

#: Retained output budgets. What is bounded is what is *kept*, never the scan.
STDOUT_TAIL_CHARS = 64 * 1024
STDERR_TAIL_CHARS = 8 * 1024
STDOUT_LINE_BYTES = 4 * STDOUT_TAIL_CHARS

#: MCP protocol versions this server speaks; an unknown request gets the default.
PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")
DEFAULT_PROTOCOL_VERSION = "2025-06-18"
#: MCP logging's eight level words, lowest first.
LOG_LEVELS = (
    "debug", "info", "notice", "warning", "error", "critical", "alert",
    "emergency",
)

#: D3f rung 2: the per-call turn metadata codex 0.156.1 attaches to every
#: ``tools/call`` as ``params._meta["x-codex-turn-metadata"]`` (Phase 1 row 1b,
#: row 7). An undocumented upstream field, so every read of it tolerates absence
#: and any malformed shape. ``workspaces`` is an object keyed by absolute
#: workspace path — under ``codex exec -C <dir>`` it names ``<dir>``, where the
#: launching shell's ``PWD`` names the wrong checkout — and ``sandbox_mode`` is
#: the session's posture word (D3b).
TURN_METADATA_KEY = "x-codex-turn-metadata"
TURN_METADATA_WORKSPACES = "workspaces"
TURN_METADATA_SANDBOX_MODE = "sandbox_mode"

#: D3b: the one posture word under which a write-capable mode is refused, and
#: the posture words under which the session itself may write, so the tool may
#: too. Codex's own vocabulary (``--sandbox``), not a surface word. Anything
#: else — absent, not a string, a word not listed here — is unreadable, and the
#: refusal fails closed exactly as it did before the posture could be read.
POSTURE_READ_ONLY = "read-only"
POSTURES_ALLOWING_WRITES = ("workspace-write", "danger-full-access")

#: D3f rung 3: workspace names read from the environment codex handed us,
#: only when present. Each one must be in A2b's ``env_vars`` (D9) — codex
#: forwards nothing else (R12′) — so ``PWD``, the launching shell's directory,
#: is allowlisted there. Row 7: it arrives only if that shell exported it, and
#: under ``-C`` it names the shell's directory, not the session's workspace.
WORKSPACE_ENV_NAMES = ("PWD",)

#: D9 layer 2: set for the child unless the user's environment already sets it,
#: so the payload's Python writes no ``__pycache__`` into the plugin cache (row 7
#: saw ``init_config.cpython-311.pyc`` appear under ``$CODEX_HOME``).
NO_BYTECODE_ENV = "PYTHONDONTWRITEBYTECODE"

#: What this server exports for the child (D9 layer 2), beside the resolved uv.
SKILL_DIR_ENV = "MULTI_LLM_SKILL_DIR"

#: The four argument properties: the shared contract's field names (rule 1).
PROPERTIES = ("mode", "plan", "question", "extra_flags")

#: The degraded definition (rule 7) — the only words this server authors for a
#: model, used only when there is no surface to read words from. The literal
#: name is also the ``.mcp.json`` server key the cap is read under.
DEGRADED_TOOL_NAME = "multi_llm"
DEGRADED_DESCRIPTION = (
    "The multi-llm plugin's generated surface file could not be read, so every "
    "call to this tool refuses until the plugin is reinstalled."
)
#: The state word a degraded call returns. It cannot come from a surface that
#: could not be read; it is the dispatcher's exit-3 word, held for this case only.
DEGRADED_STATE = "needs-human"

#: Where a redacted ``next`` went, and where any other removed command went.
NEXT_MARKER = "<see the `next` field of this result for the tool call>"
RUNNABLE_MARKER = (
    "<a runnable command was removed; commands come back only as tool calls>"
)

# JSON-RPC error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
SERVER_BUSY = -32000


# ---------------------------------------------------------------------------
# Fd 1 guard and the one writer
# ---------------------------------------------------------------------------

def _guard_protocol_fd() -> int:
    """Move the protocol channel to a private descriptor; aim fd 1 at stderr.

    Run before the first line is read. A stray ``print()``, an unraisable thread
    traceback or anything an import writes then lands on stderr by construction,
    because neither fd 1 nor ``sys.stdout`` reaches the client any more.
    """
    private = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return private


class Wire:
    """The single locked writer: one ``json.dumps`` plus newline per write."""

    def __init__(self, fd: int):
        self._fd = fd
        self._lock = threading.Lock()
        self.closed = False

    def send(self, message: dict) -> bool:
        # json.dumps escapes control characters, so the line has no embedded
        # newline; os.write on a raw fd needs no flush.
        data = (json.dumps(message) + "\n").encode("utf-8")
        with self._lock:
            if self.closed:
                return False
            try:
                view = memoryview(data)
                while view:
                    written = os.write(self._fd, view)
                    view = view[written:]
                return True
            except OSError:
                self.closed = True
                return False


def _stderr(text: str) -> None:
    try:
        sys.stderr.write("[multi-llm] " + text + "\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        pass


# ---------------------------------------------------------------------------
# Start-up reads: the surface, the sibling .mcp.json, the manifest version
# ---------------------------------------------------------------------------

#: The surface keys this server reads, checked for presence at load so a
#: right-version wrong-shape file degrades rather than raising mid-call.
_REQUIRED_SURFACE_PATHS = (
    ("tool_name",), ("tool_description",), ("argument_descriptions",),
    ("modes",), ("positional_kinds",), ("reserved_flags",), ("baked_flags",),
    ("detach_flag",), ("uv", "env"), ("uv", "run_args_env"),
    ("uv", "default_run_args"), ("uv", "candidates"),
    ("uv", "missing_message_file"), ("dispatcher", "script"),
    ("dispatcher", "result_marker"), ("dispatcher", "exit_codes"),
    ("dispatcher", "states"), ("dispatcher", "poll"), ("dispatcher", "abort"),
    ("dispatcher", "instructions"), ("dispatcher", "host_option"),
    ("dispatcher", "run_id_option"), ("dispatcher", "banner"),
    ("needs_human_message",), ("needs_human_option",), ("write_posture_refusal",),
    ("anchor_unresolved",), ("shell_missing",), ("runbook",), ("host",),
)


def load_surface():
    """``(surface, None)`` or ``(None, problem)``. Never raises."""
    try:
        raw = SURFACE_PATH.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        return None, "cannot read {}: {}".format(SURFACE_PATH, error)
    try:
        data = json.loads(raw)
    except ValueError as error:
        return None, "{} is not valid JSON: {}".format(SURFACE_PATH, error)
    if not isinstance(data, dict):
        return None, "{} is not a JSON object".format(SURFACE_PATH)
    found = data.get("schema_version")
    if found != SUPPORTED_SURFACE_VERSION:
        return None, (
            "{} declares schema_version {}, but this server supports {}; the "
            "plugin is inconsistent and must be reinstalled".format(
                SURFACE_PATH, json.dumps(found), SUPPORTED_SURFACE_VERSION)
        )
    for keys in _REQUIRED_SURFACE_PATHS:
        node = data
        for key in keys:
            node = node.get(key) if isinstance(node, dict) else None
        if node is None:
            return None, "{} lacks the key {}".format(SURFACE_PATH, ".".join(keys))
    described = data["argument_descriptions"]
    if not isinstance(data["tool_name"], str) or not data["tool_name"]:
        return None, "{}: tool_name is not a non-empty string".format(SURFACE_PATH)
    if not isinstance(data["tool_description"], str) or not isinstance(
        described, dict
    ) or not all(isinstance(described.get(name), str) for name in PROPERTIES):
        return None, "{}: the tool definition's text is malformed".format(
            SURFACE_PATH)
    return data, None


def _server_entry(data):
    """This server's entry in ``.mcp.json``, under either spelling of the table."""
    if not isinstance(data, dict):
        return None
    for table in ("mcpServers", "mcp_servers"):
        servers = data.get(table)
        if isinstance(servers, dict):
            entry = servers.get(DEGRADED_TOOL_NAME)
            if entry is None and len(servers) == 1:
                entry = next(iter(servers.values()))
            if isinstance(entry, dict):
                return entry
    return None


def load_mcp_config():
    """``(cap_s, allowlist, problems)`` from the sibling ``.mcp.json``.

    The cap is read once at start-up and cached (D3c); a file that is missing,
    unreadable, not JSON or keyless leaves ``DEFAULT_TOOL_TIMEOUT_S``, with the
    reason reported on stderr and never on fd 1.
    """
    problems = []
    try:
        entry = _server_entry(json.loads(MCP_JSON_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError, UnicodeDecodeError) as error:
        entry = None
        problems.append("cannot read {}: {}".format(MCP_JSON_PATH, error))
    if entry is None and not problems:
        problems.append("{} names no server entry".format(MCP_JSON_PATH))
    entry = entry or {}
    cap = entry.get("tool_timeout_sec")
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or cap <= 0:
        if not problems:
            problems.append("{} carries no usable tool_timeout_sec".format(
                MCP_JSON_PATH))
        cap = DEFAULT_TOOL_TIMEOUT_S
    if problems:
        problems[-1] += "; the call budget falls back to {} s".format(
            DEFAULT_TOOL_TIMEOUT_S)
    allowlist = []
    for item in entry.get("env_vars") or []:
        name = item.get("name") if isinstance(item, dict) else item
        if isinstance(name, str) and name:
            allowlist.append(name)
    return float(cap), allowlist, problems


def _plugin_version() -> str:
    try:
        version = json.loads(PLUGIN_MANIFEST_PATH.read_text(encoding="utf-8"))
        version = version.get("version")
        return version if isinstance(version, str) and version else "0.0.0"
    except (OSError, ValueError, AttributeError, UnicodeDecodeError):
        return "0.0.0"


def fill(template: str, values: dict) -> str:
    """Substitute ``{token}`` once, in one pass. Never ``str.format``."""
    if not values:
        return template
    pattern = "\\{(" + "|".join(re.escape(key) for key in values) + ")\\}"
    return re.sub(pattern, lambda match: values[match.group(1)], template)


def _positive_int(raw, fallback):
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return fallback
    return value if value > 0 else fallback


# ---------------------------------------------------------------------------
# Reserved-flag matching — argparse's rule, not a literal-token scan
# ---------------------------------------------------------------------------

def flag_matches(head: str, name: str, exact_options) -> bool:
    """Whether option token *head* would bind to option *name* (abbreviations).

    A transliteration of ``cli_common._flag_matches``: no parser in the skill
    passes ``allow_abbrev=False``, so ``--ho`` binds to ``--host``.
    """
    if head == name:
        return True
    if not name.startswith("--"):
        return False  # short options never abbreviate
    if not head.startswith("--") or len(head) <= 2:
        return False
    if head in exact_options:
        return False  # a real option spelled in full is not an abbreviation
    return name.startswith(head)


def reserved_hit(token: str, surface: dict):
    """The reserved name *token* would bind to, or None."""
    head = token.split("=", 1)[0]
    reserved = surface["reserved_flags"]
    for prefix in reserved.get("prefixes", []):
        if head.startswith(prefix):
            return prefix + "*"
    for name in list(reserved.get("exact", [])) + list(
            reserved.get("named_exception", [])):
        if flag_matches(head, name, reserved.get("exact_options", [])):
            return name
    return None


def resolve_mode(word, surface):
    for mode in surface["modes"]:
        if mode["name"] == word:
            return mode
    for mode in surface["modes"]:
        if word in mode.get("aliases", []):
            return mode
    return None


def _type_name(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _legal_words(surface) -> str:
    words = []
    for mode in surface["modes"]:
        words.append(mode["name"])
        words.extend(mode.get("aliases", []))
    return ", ".join(words)


def validate_args(args, surface):
    """``(mode_row, None)`` or ``(None, message)``. Every rule, before any spawn."""
    if not isinstance(args, dict):
        return None, "arguments must be an object, got {}.".format(_type_name(args))
    mode = args.get("mode")
    plan, question, extra = args.get("plan"), args.get("question"), args.get(
        "extra_flags")
    if not isinstance(mode, str) or not mode:
        return None, '"mode" must be a non-empty string, got {}. Legal words: {}'.format(
            _type_name(mode), _legal_words(surface))
    row = resolve_mode(mode, surface)
    if row is None:
        return None, (
            'unknown mode "{}". Legal words: {}. Nothing is prefix-matched or '
            "guessed.".format(mode, _legal_words(surface)))
    for name, value in (("plan", plan), ("question", question)):
        if not isinstance(value, str):
            return None, '"{}" must be a string, got {}.'.format(
                name, _type_name(value))
    if not isinstance(extra, list) or not all(isinstance(t, str) for t in extra):
        return None, (
            '"extra_flags" must be an array of strings (use [] for none), got '
            "{}.".format(_type_name(extra)))
    kinds = surface["positional_kinds"]
    takes_plan = kinds["plan"] in row["positionals"]
    takes_question = kinds["question"] in row["positionals"]
    if takes_plan and not plan.strip():
        return None, 'mode "{}" takes a plan path and "plan" was empty.'.format(
            row["name"])
    if not takes_plan and plan != "":
        return None, (
            'mode "{}" declares no plan positional, so "plan" must be "".'.format(
                row["name"]))
    if takes_question and not question.strip():
        return None, 'mode "{}" takes a question and "question" was empty.'.format(
            row["name"])
    if not takes_question and question != "":
        return None, (
            'mode "{}" declares no question positional, so "question" must be '
            '"".'.format(row["name"]))
    if takes_plan and plan.startswith("-"):
        return None, (
            '"plan" looks like a flag ("{}"). The dispatcher takes leading bare '
            "words as positionals and forwards everything from the first flag "
            "on, so this would arrive with one positional missing.".format(plan))
    if takes_question and question.startswith("-"):
        return None, (
            '"question" may not begin with "-": the dispatcher would read it as '
            "a flag and the mode would arrive one positional short. A question "
            "that merely contains a flag is fine; only the first character "
            "matters.")
    reserved = surface["reserved_flags"]
    for token in extra:
        hit = reserved_hit(token, surface)
        if hit is None:
            continue
        if hit in reserved.get("named_exception", []):
            # Requirement #5: refused before the spawn, never substituted.
            return None, (
                '"{}" is refused: interactive model selection is unsupported on '
                "this route. The tool spawns the dispatcher with no terminal on "
                "its stdin, so {} would force a model picker that cannot prompt. "
                "Choose models non-interactively instead, in extra_flags or in "
                "providers.yaml, or use the file route from a real terminal."
                .format(token, hit))
        return None, (
            '"{}" is reserved: this tool composes {} itself, so a call may not '
            "duplicate, contradict or restate it. Remove it from extra_flags."
            .format(token, hit))
    return row, None


# ---------------------------------------------------------------------------
# uv preflight — unattended.find_uv(), branch for branch
# ---------------------------------------------------------------------------

def _is_file(target) -> bool:
    try:
        return os.path.isfile(target)
    except (OSError, ValueError):
        return False


def expand_home(target: str, env) -> str:
    home = env.get("HOME") or env.get("USERPROFILE") or os.path.expanduser("~")
    if target == "~":
        return home
    if target.startswith("~/") or target.startswith("~\\"):
        return os.path.join(home, target[2:])
    return target


def find_uv(surface, env):
    """``$<uv.env>`` → PATH → the surface's candidates. Absolute path or None."""
    override = env.get(surface["uv"]["env"])
    if override and _is_file(override):
        return os.path.abspath(override)
    names = ("uv.exe", "uv") if sys.platform == "win32" else ("uv",)
    for directory in (env.get("PATH") or "").split(os.pathsep):
        if not directory:
            continue
        for name in names:
            candidate = os.path.join(directory, name)
            if _is_file(candidate) and os.access(candidate, os.X_OK):
                return os.path.abspath(candidate)
    for candidate in surface["uv"]["candidates"]:
        expanded = expand_home(candidate, env)
        if _is_file(expanded):
            return expanded
    return None


#: D5's health contract, row 2: the payload files every spawn needs, beside the
#: surface's dispatcher script. Checked at call time, never at start (rule 7).
PAYLOAD_INVENTORY = ("pyproject.toml", "uv.lock")


def missing_payload(surface):
    """The absolute path of the first required payload file absent, or None."""
    for name in (surface["dispatcher"]["script"],) + PAYLOAD_INVENTORY:
        target = SKILL_DIR.joinpath(*name.split("/"))
        if not _is_file(target):
            return os.path.abspath(str(target))
    return None


def payload_missing_text(target: str) -> str:
    """A partial install, named by its absolute path (D3 rule 7)."""
    return ("This plugin install is incomplete: {} is missing, so nothing was "
            "started. Reinstall the plugin, then call the tool again.".format(target))


def uv_missing_text(surface) -> str:
    """``hosts/uv-missing.txt`` from the payload, verbatim (its single owner)."""
    target = SKILL_DIR.joinpath(*surface["uv"]["missing_message_file"].split("/"))
    try:
        return target.read_text(encoding="utf-8").rstrip("\n")
    except (OSError, UnicodeDecodeError):
        # A partial install is the only way here; name the file, author nothing.
        return "uv could not be found, and this install is missing {}.".format(
            target)


# ---------------------------------------------------------------------------
# Redaction — the port of redactCommand / commandForms / scrubValue /
# runnablePattern (D3c). Applied to next, the parsed result and the tails.
# ---------------------------------------------------------------------------

def command_forms(command):
    """Every spelling of one command a captured string may hold, escaped first."""
    if not isinstance(command, str) or not command:
        return []
    escaped = json.dumps(command, ensure_ascii=False)[1:-1]
    return [command] if escaped == command else [escaped, command]


def runnable_pattern(surface):
    """``uv … run … <script> …``, bounded at a newline, a double quote or backtick."""
    script = re.escape(surface["dispatcher"]["script"])
    return re.compile(
        r'[^\s"`]*\buv(?:\.exe)?\b\s+run\b[^\n"`]*' + script + r'[^\n"`]*')


def scrub_value(value, commands, pattern):
    if isinstance(value, str):
        for form in commands:
            value = value.replace(form, NEXT_MARKER)
        return pattern.sub(RUNNABLE_MARKER, value) if pattern else value
    if isinstance(value, list):
        return [scrub_value(item, commands, pattern) for item in value]
    if isinstance(value, dict):
        return {key: scrub_value(item, commands, pattern)
                for key, item in value.items()}
    return value


_UV_RUN = re.compile(r'[^\s"`]*\buv(?:\.exe)?\b\s+run\b')


def scrub_runbook(text: str, surface) -> str:
    """Remove every runnable dispatcher line from the runbook, whole.

    The core files quote their paths (``--project "<dir>"``), which the shape
    pattern stops at, and continue long commands with a trailing backslash. So
    the runbook is scrubbed by *logical* line: one whose ``uv run`` reaches the
    dispatcher script keeps its prefix (a list number, an env assignment) and
    loses the command and every continuation line, replaced by the pointer.
    """
    script = surface["dispatcher"]["script"]
    lines, out, index = text.split("\n"), [], 0
    while index < len(lines):
        start = index
        while lines[index].rstrip().endswith("\\") and index + 1 < len(lines):
            index += 1
        logical = "\n".join(lines[start:index + 1])
        found = _UV_RUN.search(logical)
        if found and script in logical[found.start():]:
            out.append(logical[:found.start()] + RUNNABLE_MARKER)
        else:
            out.extend(lines[start:index + 1])
        index += 1
    return "\n".join(out)


def redact(value, command, surface):
    return scrub_value(value, command_forms(command),
                       runnable_pattern(surface) if surface else None)


def split_posix(text: str):
    """POSIX word splitting, quotes only. Reads; never executes. None if unbalanced."""
    out, current, quote, any_char, index = [], [], None, False, 0
    while index < len(text):
        char = text[index]
        if quote:
            if char == quote:
                quote = None
            elif quote == '"' and char == "\\" and index + 1 < len(text):
                index += 1
                current.append(text[index])
            else:
                current.append(char)
            any_char = True
        elif char in ("'", '"'):
            quote, any_char = char, True
        elif char == "\\" and index + 1 < len(text):
            index += 1
            current.append(text[index])
            any_char = True
        elif char.isspace():
            if any_char:
                out.append("".join(current))
            current, any_char = [], False
        else:
            current.append(char)
            any_char = True
        index += 1
    if quote:
        return None
    if any_char:
        out.append("".join(current))
    return out


def parse_dispatcher_command(command, surface):
    """``uv run … <script> <mode> <positionals> <flags>`` back into tool args."""
    tokens = split_posix(command)
    if not tokens:
        return None
    script = surface["dispatcher"]["script"]
    at = next((i for i, t in enumerate(tokens) if t.endswith(script)), -1)
    if at < 0 or at + 1 >= len(tokens):
        return None
    rest = tokens[at + 1:]
    row = resolve_mode(rest[0], surface)
    if row is None:
        return None
    positionals, flags, seen_flag = [], [], False
    for token in rest[1:]:
        if token.startswith("-"):
            seen_flag = True
        (flags if seen_flag else positionals).append(token)
    kinds = surface["positional_kinds"]
    takes_plan = kinds["plan"] in row["positionals"]
    takes_question = kinds["question"] in row["positionals"]
    # Drop the flags this tool composes itself, and each one's value.
    kept, index = [], 0
    while index < len(flags):
        token = flags[index]
        if reserved_hit(token, surface) is None:
            kept.append(token)
        elif "=" not in token and index + 1 < len(flags) and not flags[
                index + 1].startswith("-"):
            index += 1
        index += 1
    args = {
        "mode": row["name"],
        "plan": (positionals[0] if positionals else "") if takes_plan else "",
        "question": (positionals[1] if len(positionals) > 1 else "")
        if takes_question else "",
        "extra_flags": kept,
    }
    if takes_plan and not args["plan"]:
        return None
    return args


def next_as_tool_call(raw, surface):
    """A ``next`` string re-spelled as tool-call args, or an unparseable note."""
    if not isinstance(raw, str) or not raw:
        return None
    parsed = parse_dispatcher_command(raw, surface)
    if parsed is None:
        return {
            "unparseable": True,
            "note": "The dispatcher returned a follow-up this tool could not "
                    "parse into a tool call. It is reported rather than "
                    "executed. Do not run it in a shell.",
        }
    return parsed


def pinned_run_id(tokens, surface):
    """The value bound to the run-id option in *tokens* (abbreviations too)."""
    name = surface["dispatcher"]["run_id_option"]
    exacts = surface["reserved_flags"].get("exact_options", [])
    for index, token in enumerate(tokens):
        head, sep, value = token.partition("=")
        if not flag_matches(head, name, exacts):
            continue
        if sep:
            return value or None
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        return following if following is not None and not following.startswith(
            "-") else None
    return None


def is_mechanical(next_args, surface) -> bool:
    if not isinstance(next_args, dict) or next_args.get("unparseable"):
        return False
    markers = surface["dispatcher"].get("mechanical_continuations") or []
    return any(token.split("=", 1)[0] in markers
               for token in next_args.get("extra_flags", []))


# ---------------------------------------------------------------------------
# One child: login shell, own process group, drained pipes, bounded
# ---------------------------------------------------------------------------

class _Tail:
    """The newest *keep* characters of a stream, without quadratic copying."""

    def __init__(self, keep: int):
        self.keep, self.parts, self.size, self.truncated = keep, collections.deque(), 0, False

    def add(self, text: str) -> None:
        self.parts.append(text)
        self.size += len(text)
        while self.size > self.keep and len(self.parts) > 1:
            self.size -= len(self.parts.popleft())
            self.truncated = True

    def text(self) -> str:
        joined = "".join(self.parts)
        if len(joined) > self.keep:
            self.truncated = True
            joined = joined[-self.keep:]
        return joined


def _signal_name(number: int) -> str:
    """`` (SIGKILL)`` for a known signal number, else empty."""
    try:
        return " ({})".format(signal.Signals(number).name)
    except ValueError:
        return ""


class ChildRun:
    """What one child left behind. Never raises; a failure is a field."""

    def __init__(self, argv):
        self.argv = argv
        self.code = None
        self.signal = None  # the signal number when the child was killed by one
        self.spawn_error = None
        self.result = None
        self.result_raw = None
        self.stdout = ""
        self.stderr = ""
        self.truncated = False
        self.banner = {}
        self.cancelled = False
        self.timed_out = False


def _drain_stdout(pipe, run: ChildRun, marker: str, banner_re, tail: _Tail):
    """Scan every line: the *last* result line wins, the first banner field wins."""
    overflow = False
    try:
        for raw in iter(lambda: pipe.readline(STDOUT_LINE_BYTES), b""):
            if overflow:
                overflow = not raw.endswith(b"\n")
                continue
            if not raw.endswith(b"\n") and len(raw) >= STDOUT_LINE_BYTES:
                overflow = True  # keep the head, where the marker is
                run.truncated = True
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            trimmed = line.strip()
            if trimmed.startswith(marker):
                run.result_raw = trimmed[len(marker):].strip()
                try:
                    parsed = json.loads(run.result_raw)
                except ValueError:
                    parsed = None
                run.result = parsed if isinstance(parsed, dict) else None
            field = banner_re.match(line)
            if field and field.group(1) not in run.banner:
                run.banner[field.group(1)] = field.group(2)
            tail.add(line + "\n")
    except (OSError, ValueError):
        pass
    finally:
        _close_quietly(pipe)


def _close_quietly(pipe) -> None:
    """The drain owns its pipe: closing it from the supervisor could block on the
    buffered reader's lock while an abandoned drain sits in readline()/read1()."""
    try:
        pipe.close()
    except (OSError, ValueError):
        pass


def _drain_stderr(pipe, tail: _Tail):
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    try:
        for chunk in iter(lambda: pipe.read1(65536), b""):
            tail.add(decoder.decode(chunk))
    except (OSError, ValueError):
        pass
    finally:
        _close_quietly(pipe)


def _signal_group(proc, which) -> None:
    try:
        os.killpg(proc.pid, which)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.send_signal(which)
        except (ProcessLookupError, OSError):
            pass


def kill_tree(proc, drains) -> None:
    """SIGTERM to the process group, a 5 s grace, then SIGKILL. No graceful drain."""
    _signal_group(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=KILL_GRACE_S)
    except subprocess.TimeoutExpired:
        pass
    if proc.poll() is None or any(t.is_alive() for t in drains):
        # The leader outlived the grace, or a grandchild still holds a pipe.
        _signal_group(proc, getattr(signal, "SIGKILL", signal.SIGTERM))
        try:
            proc.wait(timeout=KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            pass


# ---------------------------------------------------------------------------
# One tools/call
# ---------------------------------------------------------------------------

class Call:
    """Per-request state: the cancel event and everything the envelope echoes."""

    def __init__(self, request_id, args, token, meta=None):
        self.id = request_id
        self.args = args
        self.token = token
        self.meta = meta if isinstance(meta, dict) else {}
        self.cancel = threading.Event()
        self.anchor = None
        self.anchor_source = None
        self.shell_argv = None
        self.progress_n = 0
        self.progress_lines = None
        self.shown = set()
        self.relays = {"attempted": 0, "delivered": 0, "rejected": 0,
                       "unavailable": 0, "logged": 0}
        # The run in view, and whether this call claimed it (D3c).
        self.claimed = False
        self.run_id = None
        self.log = None
        self.watched = None
        self.polls = 0
        self.deadline = time.monotonic()
        self.hard_deadline = self.deadline

    def remaining(self) -> float:
        return self.deadline - time.monotonic()


class Server:
    def __init__(self, wire: Wire, environ=None):
        self.wire = wire
        # D9 layer 1: what codex handed us, snapshotted before anything is
        # written into os.environ, so `missing` and every override read reports
        # what was forwarded rather than what this server exported itself.
        self.layer1 = dict(os.environ if environ is None else environ)
        self.surface, self.surface_problem = load_surface()
        self.cap_s, self.allowlist, cap_problems = load_mcp_config()
        self.version = _plugin_version()
        self.initialized = False
        self.log_floor = LOG_LEVELS.index("info")
        self.client_roots = False
        self.roots_cache = None
        self._call_lock = threading.Lock()
        self._inflight = None
        self._pending_lock = threading.Lock()
        self._pending = {}
        self._next_request = 0
        self._deferred = []
        self.shell = None
        self.uv = None
        for problem in ([self.surface_problem] if self.surface_problem else []) + cap_problems:
            self.warn(problem)

    # -- diagnostics ---------------------------------------------------------

    def tool_name(self) -> str:
        return self.surface["tool_name"] if self.surface else DEGRADED_TOOL_NAME

    def notify_log(self, level: str, text: str) -> bool:
        """A ``notifications/message`` above the floor, once initialized."""
        if not self.initialized or LOG_LEVELS.index(level) < self.log_floor:
            return False
        return self.wire.send({
            "jsonrpc": "2.0", "method": "notifications/message",
            "params": {"level": level, "logger": self.tool_name(), "data": text},
        })

    def record_choice(self, text: str) -> None:
        """A decision the server made on the user's behalf: fd 2 and an info log."""
        _stderr(text)
        self.notify_log("info", text)

    def warn(self, text: str) -> None:
        _stderr(text)
        if self.initialized:
            self.notify_log("warning", text)
        else:
            self._deferred.append(text)

    # -- the read loop (main thread) -----------------------------------------

    def serve(self, stream) -> None:
        for raw in iter(stream.readline, b""):
            line = raw.strip()
            if line:
                self.handle_line(line)
        self.shutdown()

    def shutdown(self) -> None:
        """Stdin EOF: cancel whatever is in flight, wait for it, then exit."""
        with self._call_lock:
            call = self._inflight
        if call is not None:
            call.cancel.set()
            thread = getattr(call, "thread", None)
            if thread is not None:
                thread.join(ABORT_DISPATCH_TIMEOUT_S + 4 * KILL_GRACE_S + 10)
        with self._pending_lock:
            for slot in self._pending.values():
                slot["event"].set()

    def error(self, request_id, code: int, message: str) -> None:
        self.wire.send({"jsonrpc": "2.0", "id": request_id,
                        "error": {"code": code, "message": message}})

    def reply(self, request_id, result) -> None:
        self.wire.send({"jsonrpc": "2.0", "id": request_id, "result": result})

    def handle_line(self, line: bytes) -> None:
        try:
            message = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self.error(None, PARSE_ERROR, "Parse error")
            return
        if not isinstance(message, dict):
            self.error(None, INVALID_REQUEST, "Invalid Request")
            return
        request_id = message.get("id")
        if "method" not in message:
            if "id" in message and ("result" in message or "error" in message):
                self.route_response(message)  # a reply to our own request
                return
            self.error(request_id, INVALID_REQUEST, "Invalid Request")
            return
        method = message.get("method")
        if message.get("jsonrpc") != "2.0" or not isinstance(method, str):
            self.error(request_id, INVALID_REQUEST, "Invalid Request")
            return
        params = message.get("params")
        if "id" not in message:
            self.handle_notification(method, params)
            return
        try:
            self.handle_request(request_id, method, params, message)
        except Exception as error:  # noqa: BLE001 — rule 7: the loop survives
            _stderr("internal error handling {}: {!r}".format(method, error))
            self.error(request_id, -32603, "Internal error")

    def handle_notification(self, method, params) -> None:
        params = params if isinstance(params, dict) else {}
        if method == "notifications/initialized":
            self.initialized = True
            deferred, self._deferred = self._deferred, []
            for text in deferred:
                self.notify_log("warning", text)
        elif method == "notifications/cancelled":
            target = params.get("requestId")
            with self._call_lock:
                call = self._inflight
            # Per-request state: only the in-flight id cancels; anything else,
            # including a cancel landing after the return, is a no-op.
            if call is not None and _same_id(call.id, target):
                call.cancel.set()
        elif method == "notifications/roots/list_changed":
            self.roots_cache = None

    def handle_request(self, request_id, method, params, message) -> None:
        if method == "initialize":
            params = params if isinstance(params, dict) else {}
            requested = params.get("protocolVersion")
            capabilities = params.get("capabilities")
            self.client_roots = isinstance(capabilities, dict) and isinstance(
                capabilities.get("roots"), dict)
            self.roots_cache = None
            self.reply(request_id, {
                "protocolVersion": requested if requested in PROTOCOL_VERSIONS
                else DEFAULT_PROTOCOL_VERSION,
                "capabilities": {"tools": {}, "logging": {}},
                "serverInfo": {"name": self.tool_name(), "version": self.version},
            })
        elif method == "ping":
            self.reply(request_id, {})
        elif method == "tools/list":
            # One tool is the whole list: no nextCursor, and a cursor this server
            # never issued gets the same single-tool page.
            self.reply(request_id, {"tools": [self.tool_definition()]})
        elif method == "logging/setLevel":
            level = params.get("level") if isinstance(params, dict) else None
            if level not in LOG_LEVELS:
                self.error(request_id, INVALID_PARAMS, "Invalid params")
                return
            self.log_floor = LOG_LEVELS.index(level)
            self.reply(request_id, {})
        elif method == "tools/call":
            self.start_call(request_id, params)
        else:
            self.error(request_id, METHOD_NOT_FOUND, "Method not found")

    def tool_definition(self) -> dict:
        if self.surface is None:
            properties = {name: {"type": "string"} for name in PROPERTIES}
            properties["extra_flags"] = {"type": "array", "items": {"type": "string"}}
            return {"name": DEGRADED_TOOL_NAME, "description": DEGRADED_DESCRIPTION,
                    "inputSchema": {"type": "object", "properties": properties,
                                    "required": list(PROPERTIES)}}
        described = self.surface["argument_descriptions"]
        properties = {name: {"type": "string", "description": described[name]}
                      for name in PROPERTIES}
        properties["extra_flags"] = {"type": "array", "items": {"type": "string"},
                                     "description": described["extra_flags"]}
        return {
            "name": self.surface["tool_name"],
            "description": self.surface["tool_description"],
            "inputSchema": {"type": "object", "properties": properties,
                            "required": list(PROPERTIES)},
        }

    # -- server → client requests (roots/list) --------------------------------

    def route_response(self, message) -> None:
        with self._pending_lock:
            slot = self._pending.get(message.get("id")) if isinstance(
                message.get("id"), str) else None
        if slot is None:
            return  # a response to nothing we asked: dropped in silence
        slot["response"] = message
        slot["event"].set()

    def ask_client(self, method: str, params, timeout: float):
        with self._pending_lock:
            self._next_request += 1
            request_id = "mllm-{}".format(self._next_request)
            slot = {"event": threading.Event(), "response": None}
            self._pending[request_id] = slot
        try:
            if not self.wire.send({"jsonrpc": "2.0", "id": request_id,
                                   "method": method, "params": params}):
                return None
            slot["event"].wait(max(0.0, timeout))
            return slot["response"]
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)

    # -- tools/call: accept on the main thread, run on the worker -------------

    def start_call(self, request_id, params) -> None:
        if not isinstance(params, dict):
            self.error(request_id, INVALID_PARAMS, "Invalid params")
            return
        if params.get("name") != self.tool_name():
            self.error(request_id, INVALID_PARAMS,
                       "Unknown tool: {}".format(json.dumps(params.get("name"))))
            return
        args = params.get("arguments")
        meta = params.get("_meta")
        token = meta.get("progressToken") if isinstance(meta, dict) else None
        with self._call_lock:
            busy = self._inflight
            if busy is None:
                call = Call(request_id, args, token, meta)
                self._inflight = call
        if busy is not None:
            # Refused, never queued: a queued call spends its own budget waiting
            # and two dispatchers over one plan race on its result files.
            self.error(request_id, SERVER_BUSY,
                       "Server busy: tools/call {} is still in flight; one call "
                       "runs at a time.".format(json.dumps(busy.id)))
            return
        thread = threading.Thread(target=self.run_call, args=(call,), daemon=True)
        call.thread = thread
        thread.start()

    def run_call(self, call: Call) -> None:
        try:
            envelope = self.execute(call)
        except Exception as error:  # noqa: BLE001 — never throw out of a call
            _stderr("tools/call failed unexpectedly: {!r}".format(error))
            envelope = self.envelope(call, {
                "ok": False, "state": self.state_word("error"),
                "message": "the tool failed unexpectedly: {!r}".format(error),
            })
        # Cancellation handling is released *before* the return, so a cancel
        # landing on the teardown edge kills nothing and dispatches nothing.
        with self._call_lock:
            self._inflight = None
        text = json.dumps(envelope, ensure_ascii=False)
        self.reply(call.id, {"content": [{"type": "text", "text": text}],
                             "isError": not envelope.get("ok")})

    def state_word(self, key: str) -> str:
        if self.surface is None:
            return DEGRADED_STATE
        return self.surface["dispatcher"]["states"][key]

    def environment(self, call: Call) -> dict:
        """D3a's six keys, never partial. ``missing`` is a forwarding report."""
        return {
            "platform": sys.platform,
            "shell": list(call.shell_argv) if call.shell_argv else None,
            "login_shell": call.shell_argv is not None,
            "profile_shell": self.layer1.get("SHELL") or None,
            "interpreter": "{} ({})".format(
                sys.executable, ".".join(str(part) for part in sys.version_info[:3])),
            "missing": sorted(name for name in set(self.allowlist)
                              if name not in self.layer1),
        }

    def envelope(self, call: Call, fields: dict) -> dict:
        """Rule 5's one shape: the fields every branch carries, then the rest."""
        out = {"ok": bool(fields.get("ok")), "state": fields.get("state"),
               "message": fields.get("message"), "exit": fields.get("exit"),
               "anchor": call.anchor, "anchor_source": call.anchor_source,
               "environment": self.environment(call), "relays": dict(call.relays),
               "progress_lines": call.progress_lines}
        for key, value in fields.items():
            if key not in out and value is not None:
                out[key] = value
        return out

    def refusal(self, call: Call, state_key: str, message: str) -> dict:
        """A pre-spawn refusal: the ordinary envelope, ``exit`` null, relays zeroed."""
        return self.envelope(call, {"ok": False, "state": self.state_word(state_key),
                                    "message": message, "exit": None})

    # -- the budget ------------------------------------------------------------

    def budget_s(self) -> float:
        cap = self.cap_s
        override = _positive_int(self.layer1.get("MLLM_TEST_POLL_BUDGET_MS"), None)
        if override is not None:
            cap = min(cap, override / 1000.0)  # a test override only shortens
        return cap

    # -- the anchor ladder (D3f) -------------------------------------------------

    def _usable_dir(self, value):
        if not isinstance(value, str) or not value:
            return None, "unset"
        expanded = os.path.expanduser(value)
        if not os.path.isabs(expanded):
            return None, "not an absolute path"
        resolved = os.path.realpath(expanded)
        if not os.path.isdir(resolved):
            return None, "not a directory"
        root = os.path.realpath(str(PLUGIN_ROOT))
        if resolved == root or resolved.startswith(root.rstrip(os.sep) + os.sep):
            return None, "the plugin's own install location"
        return resolved.rstrip(os.sep) or os.sep, None

    def _roots(self, call: Call):
        if self.roots_cache is None:
            timeout = min(ROOTS_TIMEOUT_S, max(0.0, call.remaining()))
            response = self.ask_client("roots/list", {}, timeout)
            result = response.get("result") if isinstance(response, dict) else None
            roots = result.get("roots") if isinstance(result, dict) else None
            if not isinstance(roots, list):
                return None
            self.roots_cache = roots
        return self.roots_cache

    def _workspace_candidates(self, call: Call):
        """Rung 2's paths, preferred first, and a note on the choice (D3f).

        One key is the workspace. Several: the one equal to or containing the
        ``PWD`` codex handed us (the most specific when more than one does) goes
        first, else codex's own first; the rest follow in codex's order.
        """
        workspaces = turn_metadata(call.meta).get(TURN_METADATA_WORKSPACES)
        if not isinstance(workspaces, dict):
            return [], "absent" if workspaces is None else "not an object"
        # A NUL cannot be in a path and makes os.path raise, so such a key is
        # malformed metadata and dropped with the non-strings.
        keys = [key for key in workspaces
                if isinstance(key, str) and key and "\x00" not in key]
        if not keys:
            return [], "no workspaces"
        if len(keys) == 1:
            return keys, None
        pwd = self.layer1.get("PWD")
        pwd = os.path.realpath(pwd) if isinstance(pwd, str) and os.path.isabs(pwd) else None
        best = None
        for key in keys:
            if not os.path.isabs(key) or pwd is None:
                continue
            real = os.path.realpath(key).rstrip(os.sep) or os.sep
            if pwd == real or pwd.startswith(real.rstrip(os.sep) + os.sep):
                if best is None or len(real) > len(os.path.realpath(best)):
                    best = key
        chosen = best if best is not None else keys[0]
        ordered = [chosen] + [key for key in keys if key != chosen]
        note = "{} workspaces; chose {} ({})".format(
            len(keys), json.dumps(chosen),
            "it holds PWD" if best is not None else "the first; none holds PWD")
        return ordered, note

    def resolve_anchor(self, call: Call):
        """``(anchor, source, probed)``; the first green rung wins, never the process cwd."""
        probed = []
        if self.client_roots:
            roots = self._roots(call)
            note = "no answer to roots/list" if roots is None else "no roots"
            for root in roots or []:
                uri = root.get("uri") if isinstance(root, dict) else None
                parsed = urlparse(uri) if isinstance(uri, str) else None
                if parsed is None or parsed.scheme != "file":
                    note = "the first root is not a file:// URI"
                else:
                    path, note = self._usable_dir(unquote(parsed.path))
                    if path:
                        return path, "roots", probed
                break  # the first root is the anchor, or nothing is
            probed.append("roots ({})".format(note))
        else:
            probed.append("roots (not offered by the client)")
        candidates, note = self._workspace_candidates(call)
        skipped = []
        for candidate in candidates:
            path, problem = self._usable_dir(candidate)
            if path:
                if note or skipped:
                    # Several workspaces, or the preferred one unusable: the
                    # choice is recorded where the user and the host can see it.
                    parts = ([note] if note else []) + skipped
                    if skipped:
                        parts.append("took {}".format(json.dumps(candidate)))
                    self.record_choice(
                        "anchor from turn-metadata: {}".format("; ".join(parts)))
                return path, "turn-metadata", probed
            skipped.append("skipped {} ({})".format(json.dumps(candidate), problem))
        if candidates:
            probed.append("turn-metadata ({})".format("; ".join(skipped)))
        else:
            probed.append("turn-metadata ({})".format(note))
        for name in WORKSPACE_ENV_NAMES:
            path, note = self._usable_dir(self.layer1.get(name))
            if path:
                return path, "env:" + name, probed
            probed.append("env:{} ({})".format(name, note))
        return None, None, probed

    # -- execute: validate → anchor → refusals → preflight → the loop ----------

    def execute(self, call: Call) -> dict:
        start = time.monotonic()
        budget = self.budget_s()
        call.hard_deadline = start + budget
        call.deadline = call.hard_deadline - min(RESERVE_MAX_S, budget * RESERVE_FRACTION)
        call.anchor, call.anchor_source, probed = self.resolve_anchor(call)
        if self.surface is None:
            return self.refusal(call, "needs_human", "{} {}".format(
                DEGRADED_DESCRIPTION, self.surface_problem or ""))
        surface = self.surface
        row, problem = validate_args(call.args, surface)
        if problem:
            return self.refusal(call, "error", problem)
        if call.anchor is None:
            return self.refusal(call, "needs_human", fill(
                surface["anchor_unresolved"], {"probed": ", ".join(probed)}))
        args = call.args
        posture = surface["write_posture_refusal"]
        session = session_posture(call.meta)
        if row.get("writes_worktree") and session not in POSTURES_ALLOWING_WRITES \
                and self.layer1.get(posture["lever_env"]) != posture["lever_value"]:
            # D3b: the posture is read per call from codex's turn metadata. A
            # read-only session refuses; a posture that is absent or not a word
            # this server knows is treated as read-only (fail closed). Only a
            # session that may itself write lets the tool write. The lever is
            # read from layer 1 (what codex forwarded), never a login profile:
            # it must be in codex's own environment, not ~/.profile.
            return self.refusal(call, "needs_human", fill(posture["template"], {
                "mode": args["mode"],
                "posture": session or posture["posture_unreadable"]}))
        self.shell = self.find_shell()
        if self.shell is None:
            return self.refusal(call, "needs_human", fill(surface["shell_missing"], {
                "shell": SHELL_NAME, "platform": sys.platform,
                "path": self.layer1.get("PATH", "")}))
        uv = find_uv(surface, self.layer1)
        if uv is None:
            return self.refusal(call, "needs_human", uv_missing_text(surface))
        # D5 row 2 / rule 7: a partial payload is refused by name before any
        # spawn, so a missing lockfile never reaches uv (or a --no-sync override).
        missing = missing_payload(surface)
        if missing is not None:
            return self.refusal(call, "needs_human", payload_missing_text(missing))
        self.uv = uv
        # D9 layer 2: the resolved uv (the detached worker reuses it) and the
        # skill dir. The child inherits os.environ whole; no env= is passed.
        os.environ[surface["uv"]["env"]] = uv
        os.environ[SKILL_DIR_ENV] = str(SKILL_DIR)
        if NO_BYTECODE_ENV not in os.environ:
            os.environ[NO_BYTECODE_ENV] = "1"
        values = {"word": args["mode"], "plan": self.normalize_plan(args["plan"], call),
                  "question": args["question"], "extra": list(args["extra_flags"])}
        outcome = self.run_workflow(call, row, values)
        if not call.cancel.is_set() and self.unfinished(row, outcome):
            self.attach_runbook(call, row, outcome)
        return self.envelope(call, outcome)

    def find_shell(self):
        found = shutil.which(SHELL_NAME, path=self.layer1.get("PATH") or os.defpath)
        if found:
            return os.path.abspath(found)
        fallback = self.layer1.get("MLLM_TEST_FALLBACK_SHELL") or FALLBACK_SHELL
        return fallback if _is_file(fallback) and os.access(fallback, os.X_OK) else None

    def normalize_plan(self, plan: str, call: Call) -> str:
        """A relative plan resolved against the anchor, ``~`` expanded."""
        if not plan:
            return ""
        expanded = expand_home(plan, self.layer1)
        if os.path.isabs(expanded):
            return expanded
        return os.path.normpath(os.path.join(call.anchor, expanded))

    # -- argv composition (D3 rule 3) ---------------------------------------------

    def dispatcher_argv(self, row, values) -> list:
        surface = self.surface
        kinds = surface["positional_kinds"]
        argv = [values.get("word") or row["name"]]  # aliases travel unexpanded
        for kind in row["positionals"]:
            if kind == kinds["plan"] and values.get("plan"):
                argv.append(values["plan"])
            elif kind == kinds["question"] and values.get("question"):
                argv.append(values["question"])
        argv.extend(surface["baked_flags"])
        if row.get("supports_detach"):
            argv.append(surface["detach_flag"])
        argv.extend(values.get("extra") or [])
        return argv

    def uv_run_args(self) -> list:
        """``MULTI_LLM_UV_RUN_ARGS`` replaces the surface's list; ASCII-whitespace split."""
        uv = self.surface["uv"]
        override = (self.layer1.get(uv["run_args_env"]) or "").strip(" \t\n\r\f\v")
        if override:
            return re.split(r"[ \t\n\r\f\v]+", override)
        return list(uv["default_run_args"])

    def spawn_argv(self, dispatcher_argv) -> list:
        script = str(SKILL_DIR / self.surface["dispatcher"]["script"])
        return ([self.uv, "run"] + self.uv_run_args()
                + ["--project", str(SKILL_DIR), "--", "python", script]
                + list(dispatcher_argv))

    def status_argv(self, extra, plan) -> list:
        poll_word = self.surface["dispatcher"]["poll"][0]
        row = resolve_mode(poll_word, self.surface)
        return self.dispatcher_argv(row, {"word": row["name"], "plan": plan,
                                          "question": "", "extra": extra})

    # -- one child ------------------------------------------------------------------

    def run_child(self, call: Call, dispatcher_argv, timeout_s: float,
                  cancellable: bool = True) -> ChildRun:
        """Spawn ``[bash, -lc, CONST, bash, anchor, *argv]`` and supervise it.

        One shell at this one site, an argv list, ``shell=False``, its own
        session (so the kill reaches uv's python grandchild), stdin from
        /dev/null (fd 0 is the protocol's read half), both pipes drained by
        their own threads so a chatty child can never fill one.
        """
        argv = self.spawn_argv(dispatcher_argv)
        run = ChildRun(argv)
        prefix = [self.shell, "-lc", LOGIN_SHELL_SCRIPT, SHELL_NAME]
        try:
            proc = subprocess.Popen(
                prefix + [call.anchor] + argv, shell=False, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True, close_fds=True)
        except (OSError, ValueError) as error:
            run.spawn_error = str(error)
            return run
        call.shell_argv = prefix
        dispatcher = self.surface["dispatcher"]
        fields = dispatcher["banner"].get("fields") or [dispatcher["banner"]["run_id_field"]]
        banner_re = re.compile(
            r"^\s*(" + "|".join(re.escape(f) for f in fields) + r"):\s*(.+?)\s*$")
        out_tail, err_tail = _Tail(STDOUT_TAIL_CHARS), _Tail(STDERR_TAIL_CHARS)
        drains = [
            threading.Thread(target=_drain_stdout, daemon=True, args=(
                proc.stdout, run, dispatcher["result_marker"], banner_re, out_tail)),
            threading.Thread(target=_drain_stderr, daemon=True,
                             args=(proc.stderr, err_tail)),
        ]
        for thread in drains:
            thread.start()
        until = time.monotonic() + max(0.0, timeout_s)
        while True:
            if proc.poll() is not None and not any(t.is_alive() for t in drains):
                break
            if cancellable and call.cancel.is_set():
                run.cancelled = True
                break
            if time.monotonic() >= until:
                run.timed_out = True
                break
            if cancellable:
                call.cancel.wait(0.02)
            else:
                time.sleep(0.02)
        if run.cancelled or run.timed_out:
            kill_tree(proc, drains)
        for thread in drains:
            # A drain still blocked after the grace is abandoned, never waited on.
            # Each drain closes its own pipe on exit; closing it here would block
            # on the reader's lock while an abandoned drain is still inside a read.
            thread.join(KILL_GRACE_S if (run.cancelled or run.timed_out) else None)
        code = proc.poll()
        run.code = code if (code is not None and code >= 0 and not (
            run.cancelled or run.timed_out)) else None
        if code is not None and code < 0 and not (run.cancelled or run.timed_out):
            run.signal = -code  # e.g. an OOM kill: `code` stays None, the cause is kept
        run.stdout, run.stderr = out_tail.text(), err_tail.text()
        run.truncated = run.truncated or out_tail.truncated
        return run

    # -- progress (D3d) ----------------------------------------------------------------

    def relay(self, call: Call, child: ChildRun) -> None:
        """One relay point: progress notification, log notification, counters."""
        result = child.result or {}
        update = redact({"progress": result.get("progress"),
                         "lines": result.get("progress_lines"),
                         "run_id": call.run_id, "log": call.log, "polls": call.polls},
                        None, self.surface)
        supplied = update["lines"] if isinstance(update["lines"], list) else []
        fresh = []
        for line in supplied:
            if isinstance(line, str) and line and line not in call.shown:
                call.shown.add(line)
                fresh.append(line)
        if supplied:
            call.progress_lines = [line for line in supplied if isinstance(line, str)]
        text = "\n".join(fresh) if fresh else json.dumps(update)
        call.relays["attempted"] += 1
        if call.token is not None:
            call.progress_n += 1
            sent = self.wire.send({
                "jsonrpc": "2.0", "method": "notifications/progress",
                "params": {"progressToken": call.token, "progress": call.progress_n,
                           "message": text}})
            call.relays["delivered" if sent else "rejected"] += 1
        else:
            call.relays["unavailable"] += 1
        if self.notify_log("info", text):
            call.relays["logged"] += 1

    # -- S1–S6 -------------------------------------------------------------------------

    def run_workflow(self, call: Call, row, values) -> dict:
        surface = self.surface
        dispatcher = surface["dispatcher"]
        codes = dispatcher["exit_codes"]
        banner = dispatcher["banner"]
        call.watched = values

        def cycle(cycle_row, cycle_values):
            # A pinned re-attach names its run before the blocking wait, so a
            # wait that eats the whole budget still hands that run off.
            pinned = (None if cycle_row.get("supports_detach")
                      else self.pinned_reattach_id(cycle_values))
            # S1 — the launch child. Its result line is a handshake, not the answer.
            launch = self.run_child(call, self.dispatcher_argv(cycle_row, cycle_values),
                                    call.remaining())
            if launch.spawn_error:
                return "spawn-error", self.error_result(
                    call, launch, "could not start the dispatcher: {}".format(
                        launch.spawn_error))
            if cycle_row.get("supports_detach"):
                # S2 — the run id, off the banner and nothing else (rule f).
                launched = launch.banner.get(banner["run_id_field"])
                if launched:
                    call.run_id, call.claimed = launched, True
                    call.log = launch.banner.get(banner["log_field"])
                    call.watched = cycle_values
                    if not call.cancel.is_set():
                        self.relay(call, launch)
                if launch.cancelled or launch.timed_out or launch.code != codes["ok"]:
                    if launched and launch.cancelled:
                        return "return", self.stop_report(
                            call, "the call was cancelled while the launch was "
                                  "still running")
                    if launched and launch.timed_out:
                        return "return", self.handoff(call)
                    if launched:
                        return "return", self.stop_report(
                            call, "the launch exited {} after reporting a run".format(
                                launch.code))
                    return "return", self.terminal_result(call, launch, cycle_values)
                if not launched:
                    return "return", self.error_result(
                        call, launch, "the launch reported no run id on its banner, "
                        "so the run cannot be polled or aborted safely. Refusing to "
                        "adopt whatever run is newest.")
            else:
                # S0's foreground child; a pinned view of a live run re-attaches.
                if pinned is not None and launch.timed_out:
                    call.run_id, call.claimed = pinned, False
                    call.log = launch.banner.get(banner["log_field"])
                    call.watched = cycle_values
                    return "return", self.handoff(call)
                adopted = self.reattached_run_id(launch, cycle_values)
                if adopted is None:
                    return "return", self.terminal_result(call, launch, cycle_values)
                call.run_id, call.claimed = adopted, False
                call.log = launch.banner.get(banner["log_field"])
                call.watched = cycle_values
            # S3 — the poll, pinned to the run id, one blocking child per turn.
            poll_argv = self.status_argv(
                list(dispatcher["poll"][1:]) + [dispatcher["run_id_option"], call.run_id],
                call.watched.get("plan"))
            while True:
                if call.cancel.is_set():
                    return "return", self.stop_report(call, "the call was cancelled")
                remaining = call.remaining()
                if remaining <= 0:
                    return "return", self.handoff(call)
                # The remaining budget only: see CANCEL_LATENCY_BOUND_S for why
                # a poll is never cut at the cancellation bound.
                poll = self.run_child(call, poll_argv, remaining)
                call.polls += 1
                if poll.spawn_error:
                    return "return", self.error_result(
                        call, poll, "could not poll: {}".format(poll.spawn_error),
                        with_run=True)
                if poll.cancelled:
                    return "return", self.stop_report(call, "the call was cancelled")
                if poll.timed_out:
                    continue  # the budget ran out mid-poll: the loop hands off
                # S4 — the exit map is the loop condition.
                if poll.code != codes["nothing_to_do"]:
                    break
                self.relay(call, poll)  # never the terminal poll
            terminal = self.terminal_result(call, poll, call.watched)
            terminal["launch_state"] = (launch.result or {}).get("state")
            return "terminal", terminal

        kind, outcome = cycle(row, values)
        if kind != "terminal":
            return outcome
        # S5 — a mechanical continuation is re-composed and driven here; every
        # other next is returned as tool-call args, never run.
        cap = _positive_int(self.layer1.get("MLLM_TEST_NEXT_CAP"), DEFAULT_CONTINUATION_CAP)
        followed = 0
        while (outcome.get("ok") and followed < cap and not call.cancel.is_set()
               and is_mechanical(outcome.get("next"), surface)):
            args = outcome["next"]
            checked, problem = validate_args(args, surface)
            if problem:
                outcome["next"] = {
                    "unparseable": True,
                    "note": "The dispatcher returned a follow-up this tool refused "
                            "to run: {} It is reported rather than executed. Do not "
                            "run it in a shell.".format(problem)}
                break
            followed += 1
            kind, following = cycle(checked, {
                "word": args["mode"], "plan": self.normalize_plan(args["plan"], call),
                "question": args["question"], "extra": list(args["extra_flags"])})
            if kind == "spawn-error":
                break
            if kind == "return":
                return following
            outcome = following
            outcome["continuations"] = followed
        if followed >= cap and is_mechanical(outcome.get("next"), surface):
            return self.stop_report(call, "the continuation cap of {} was reached".format(cap))
        return outcome

    def pinned_reattach_id(self, values):
        """The run a pinned poll-mode view names, known before its child runs.

        Only the poll mode's own word counts (its aliases are abort spellings),
        and never alongside the abort flag: an abort that times out is not a
        watch to hand off.
        """
        surface = self.surface
        dispatcher = surface["dispatcher"]
        if values.get("word") != dispatcher["poll"][0]:
            return None
        extra = values.get("extra") or []
        exacts = surface["reserved_flags"].get("exact_options", [])
        abort_flags = list(dispatcher["abort"][1:])
        if any(flag_matches(token.partition("=")[0], flag, exacts)
               for token in extra for flag in abort_flags):
            return None
        return pinned_run_id(extra, surface)

    def reattached_run_id(self, child: ChildRun, values):
        """The run a pinned ``status`` view re-attached to, while it is live."""
        surface = self.surface
        if child.code != surface["dispatcher"]["exit_codes"]["nothing_to_do"]:
            return None
        if pinned_run_id(values.get("extra") or [], surface) is None:
            return None
        raw = (child.result or {}).get("next")
        if not isinstance(raw, str) or not raw:
            return None
        parsed = parse_dispatcher_command(raw, surface)
        poll_mode = resolve_mode(surface["dispatcher"]["poll"][0], surface)
        if parsed is None or poll_mode is None or parsed["mode"] != poll_mode["name"]:
            return None
        return pinned_run_id(parsed["extra_flags"], surface)

    def abort_args(self, call: Call) -> dict:
        """The explicit abort spelling, as tool-call args (never the alias)."""
        dispatcher = self.surface["dispatcher"]
        word, flags = dispatcher["abort"][0], list(dispatcher["abort"][1:])
        return {"mode": word, "plan": (call.watched or {}).get("plan", ""),
                "question": "", "extra_flags": flags + [dispatcher["run_id_option"],
                                                        call.run_id]}

    def reattach_args(self, call: Call) -> dict:
        dispatcher = self.surface["dispatcher"]
        return {"mode": dispatcher["poll"][0], "plan": (call.watched or {}).get("plan", ""),
                "question": "", "extra_flags": list(dispatcher["poll"][1:]) + [
                    dispatcher["run_id_option"], call.run_id]}

    def handoff(self, call: Call) -> dict:
        """The budget expired: the run is left ALIVE and named (D3c's re-decided edge).

        No abort dispatch, claimed or not. The claim was already released by
        reaching here; the re-attach ``next`` is how the run is picked up, and a
        chain of re-attaches hands off again rather than ever aborting.
        """
        return {
            # ok: the run is alive and re-attachable; an error flag would invite
            # the model to give up on it or to retry the launch.
            "ok": True, "state": self.surface["dispatcher"]["states"]["running"], "exit": None,
            "run_id": call.run_id, "log": call.log, "polls": call.polls,
            "next": self.reattach_args(call), "abort_dispatched": False,
            "abort_command": self.abort_args(call),
            "message": "This call's time budget ran out while the detached run "
                       "{} was still going, so it was left running, untouched. Send "
                       "the `next` tool call exactly as given to keep watching it; "
                       "stop it only with the `abort_command` tool call.{}".format(
                           call.run_id,
                           " Log: {}.".format(call.log) if call.log else ""),
        }

    def stop_report(self, call: Call, reason: str) -> dict:
        """Cancellation or the continuation cap: abort a claimed run, else leave it."""
        if not call.claimed:
            # A re-attach claims nothing: supervision stops, the run keeps going.
            return {
                "ok": False, "state": self.state_word("needs_human"), "exit": None,
                "reason": reason, "run_id": call.run_id, "log": call.log,
                "abort_dispatched": False, "abort_command": self.abort_args(call),
                "message": "{}. The run {} was not started by this call, so it was "
                           "left running rather than stopped.{} Watch it again by "
                           "sending this call again, or stop it with the "
                           "`abort_command` tool call in this result.".format(
                               reason, call.run_id,
                               " Log: {}.".format(call.log) if call.log else ""),
            }
        codes = self.surface["dispatcher"]["exit_codes"]
        stopped = (codes["ok"], codes["nothing_to_do"])
        # Sent after the kill ladder returned, signal-free, under its own timeout.
        out = self.run_child(call, self.status_argv(
            list(self.surface["dispatcher"]["abort"][1:])
            + [self.surface["dispatcher"]["run_id_option"], call.run_id],
            (call.watched or {}).get("plan")), ABORT_DISPATCH_TIMEOUT_S, cancellable=False)
        dispatched = (not out.spawn_error and out.code in stopped and out.result is not None
                      and out.result.get("exit") in stopped)
        return {
            "ok": False, "state": self.state_word("needs_human"), "exit": None,
            "reason": reason, "run_id": call.run_id, "log": call.log,
            "abort_dispatched": dispatched, "abort_command": self.abort_args(call),
            "message": "{}. The detached run {} was {}. {}{}If it is still alive, "
                       "stop it with the `abort_command` tool call in this result."
                       .format(reason, call.run_id,
                               "confirmed aborted" if dispatched
                               else "NOT confirmed aborted",
                               "The abort dispatch timed out and was killed. "
                               if out.timed_out else "",
                               "Log: {}. ".format(call.log) if call.log else ""),
        }

    # -- S4/S5: results -------------------------------------------------------------------

    def human_output(self, child: ChildRun, command):
        """Redacted stdout, the result line dropped only when it parsed."""
        text = redact(child.stdout, command, self.surface)
        if not text or child.result is None:
            return text.rstrip() if text else text
        marker = self.surface["dispatcher"]["result_marker"]
        return "\n".join(line for line in text.split("\n")
                         if not line.strip().startswith(marker)).rstrip()

    def tool_calls_for(self, child: ChildRun) -> list:
        raw = (child.result or {}).get("next")
        commands = [raw] if isinstance(raw, str) and raw else []
        commands.extend(runnable_pattern(self.surface).findall(
            "{}\n{}".format(child.stdout or "", child.stderr or "")))
        seen, calls = set(), []
        for command in commands:
            if command in seen:
                continue
            seen.add(command)
            call_args = next_as_tool_call(command, self.surface)
            if call_args is not None:
                calls.append(call_args)
        return calls

    def nested_exec(self, child: ChildRun, call: Call):
        haystack = "{}\n{}".format(child.stdout or "", child.stderr or "")
        for code, field in (self.surface.get("nested_exec_reasons") or {}).items():
            if code in haystack:
                return {"reason": code, "capability_field": field, "log": call.log}
        return None

    def needs_human_message(self, payload, output, stderr, values) -> str:
        text = self.surface["needs_human_message"]
        states = self.surface["dispatcher"]["states"]
        row = resolve_mode(values.get("word"), self.surface) if values and values.get(
            "word") else None
        launched = row.get("phase") if row else None
        gate = self.surface["dispatcher"].get("review_gate") or {}
        high = (payload or {}).get("high_priority_groups")
        count = high.get("count") if isinstance(high, dict) else None
        held = (row is not None and row["name"] == gate.get("mode")
                and gate.get("flag") in (values.get("extra") or [])
                and isinstance(count, int) and not isinstance(count, bool) and count > 0)
        if payload is None:
            phase = text["phase_unknown"]
        elif launched and held:
            phase = fill(text["review_completed"], {"phase": launched})
        elif launched:
            incomplete = payload.get("state") == states["needs_human"]
            phase = fill(text["phase_incomplete"] if incomplete else text["phase_known"],
                         {"phase": launched})
        else:
            phase = text["phase_unnamed"]
        if payload is not None:
            error = payload.get("error")
            line = error.strip() if isinstance(error, str) and error.strip() else None
        else:
            line = _last_line(output) or _last_line(stderr)
        error_clause = fill(text["error_known"], {"error": '"{}"'.format(line)}) \
            if line else text["error_unknown"]
        return fill(text["template"], {"phase": phase, "error": error_clause})

    def terminal_result(self, call: Call, child: ChildRun, values) -> dict:
        surface = self.surface
        codes = surface["dispatcher"]["exit_codes"]
        states = surface["dispatcher"]["states"]
        base = {"exit": child.code, "argv": child.argv, "run_id": call.run_id,
                "log": call.log, "polls": call.polls,
                "stdout_truncated": True if child.truncated else None}
        if child.code is not None and child.code == codes["needs_human"]:
            payload = redact(child.result, None, surface)
            output = self.human_output(child, None)
            stderr = redact(child.stderr, None, surface)
            base.update({
                "ok": False, "state": states["needs_human"], "result": payload,
                "error": (payload or {}).get("error"), "output": output,
                "stderr": stderr or None, "next": None,
                "options": self.tool_calls_for(child) + [
                    {"human": surface["needs_human_option"]}],
                "nested_exec": self.nested_exec(child, call),
                "message": self.needs_human_message(payload, output, stderr, values),
            })
            return base
        if child.code is not None and child.code == codes["nothing_to_do"] and child.result:
            raw = child.result.get("next")
            base.update({
                "ok": True, "state": child.result.get("state") or states["nothing_to_do"],
                "result": redact(child.result, raw, surface),
                "artifacts": redact(child.result.get("artifacts") or [], raw, surface),
                "next": next_as_tool_call(raw, surface),
                "output": self.human_output(child, raw),
                "stderr": redact(child.stderr, raw, surface) or None})
            return base
        if child.code != codes["ok"] or child.result is None:
            marker = surface["dispatcher"]["result_marker"]
            if child.cancelled:
                why = "the call was cancelled before the dispatcher finished"
            elif child.timed_out:
                why = "the call's time budget ran out before the dispatcher finished"
            elif child.signal is not None:
                why = "the dispatcher was killed by signal {}{}".format(
                    child.signal, _signal_name(child.signal))
            elif child.result is not None:
                why = "the dispatcher exited {}".format(child.code)
            elif child.result_raw is not None:
                why = "the dispatcher's {} line was not valid JSON: {}".format(
                    marker, child.result_raw)
            elif child.code is None:
                why = "the dispatcher ended without an exit code and with no {} line".format(
                    marker)
            else:
                why = "the dispatcher exited {} with no {} line".format(child.code, marker)
            return self.error_result(call, child, why, with_run=True)
        raw = child.result.get("next")
        base.update({
            "ok": True, "state": child.result.get("state") or states["ok"],
            "result": redact(child.result, raw, surface),
            "artifacts": redact(child.result.get("artifacts") or [], raw, surface),
            "next": next_as_tool_call(raw, surface),
            "output": self.human_output(child, raw),
            "stderr": redact(child.stderr, raw, surface) or None})
        return base

    def error_result(self, call: Call, child: ChildRun, message: str,
                     with_run: bool = False) -> dict:
        """A failed child is a result. With a live run, it names how to stop it."""
        options = self.tool_calls_for(child)
        out = {
            "ok": False, "state": self.state_word("error"), "exit": child.code,
            "argv": child.argv, "message": redact(message, None, self.surface),
            "stderr": redact(child.stderr, None, self.surface) or None,
            "output": self.human_output(child, None) or None,
            "options": options or None, "nested_exec": self.nested_exec(child, call),
        }
        if with_run and call.run_id:
            # `status` itself failing is never a statement about the run.
            out.update({"run_id": call.run_id, "log": call.log,
                        "abort_command": self.abort_args(call)})
        return out

    # -- the runbook (rule 5, D3's runbook paragraph) -----------------------------------

    def unfinished(self, row, outcome) -> bool:
        """Positively enumerated: running, needs-human exit, status failed, unfollowed next."""
        if not row.get("supports_detach"):
            return False
        states = self.surface["dispatcher"]["states"]
        codes = self.surface["dispatcher"]["exit_codes"]
        state = outcome.get("state")
        if state == states["running"]:
            return True
        if state == states["needs_human"] and outcome.get("exit") == codes["needs_human"]:
            return True
        if state == states["error"] and outcome.get("run_id") and outcome.get("exit") is not None:
            return True
        return outcome.get("ok") is True and outcome.get("next") is not None

    def attach_runbook(self, call: Call, row, outcome) -> None:
        """``util instructions <mode> --host <id>``: integrity-checked, banner dropped."""
        dispatcher = self.surface["dispatcher"]
        runbook = self.surface["runbook"]
        argv = list(dispatcher["instructions"]) + [
            row["name"], dispatcher["host_option"], self.surface["host"]]
        # Cut from the reserve the loop held back, not from the loop's deadline:
        # on the handoff edge that deadline has already passed.
        spare = max(1.0, call.hard_deadline - time.monotonic() - 0.5)
        child = self.run_child(call, argv, spare)
        problem = None
        if child.spawn_error or child.code != dispatcher["exit_codes"]["ok"]:
            problem = "the runbook fetch did not complete (exit {})".format(child.code)
        lines = (child.stdout or "").split("\n")
        # Drop the head banner: its character count describes the unscrubbed core.
        opening = runbook.get("banner_opening") or ""
        fence = opening.split()[0] if opening.split() else None
        kept, index = [], 0
        while index < len(lines):
            line = lines[index]
            if opening and fence and line.startswith(opening):
                index += 1
                while index < len(lines) and not lines[index].startswith(fence):
                    index += 1
                index += 1
                continue
            if not line.strip().startswith(dispatcher["result_marker"]):
                kept.append(line)
            index += 1
        sentinel = runbook["core_end_marker"] + " " + row["name"]
        if problem is None and not any(line.rstrip() == sentinel for line in kept):
            problem = ("the runbook arrived without its closing line ({}), so it is a "
                       "partial payload and was not attached".format(sentinel))
        if problem:
            # A documentation fetch never restates the run: ok/state/isError stay
            # as the run produced them (a live handoff or an exit-0 with an
            # unfollowed next is not a failure because its runbook was not fetched).
            outcome["runbook_error"] = problem
            return
        outcome["runbook"] = redact(scrub_runbook("\n".join(kept).strip("\n"), self.surface),
                                    None, self.surface)


def turn_metadata(meta) -> dict:
    """``_meta["x-codex-turn-metadata"]`` as a dict, or ``{}``; never raises.

    Seen as a JSON object on 0.156.1; a JSON-encoded string is accepted too,
    since the field is undocumented and its encoding is not a contract.
    """
    value = meta.get(TURN_METADATA_KEY) if isinstance(meta, dict) else None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def session_posture(meta):
    """D3b: the session's sandbox posture word, or ``None`` when unreadable."""
    value = turn_metadata(meta).get(TURN_METADATA_SANDBOX_MODE)
    if value == POSTURE_READ_ONLY or value in POSTURES_ALLOWING_WRITES:
        return value
    return None


def _last_line(text):
    if not isinstance(text, str):
        return None
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return lines[-1] if lines else None


def _same_id(left, right) -> bool:
    """JSON-RPC ids compare with their JSON type preserved (1 is not "1")."""
    return type(left) is type(right) and left == right


def main() -> int:
    fd = _guard_protocol_fd()
    server = Server(Wire(fd))

    def _terminate(_number, _frame):
        # Measured (Phase 1 row 10, codex 0.156.1): on an interrupted session
        # codex closes this server's stdin and sends SIGTERM in the same
        # millisecond, and sends no `notifications/cancelled`. Python's default
        # SIGTERM exit beat the EOF branch every time, so the in-flight child —
        # its own session, by design — was orphaned. SIGTERM (and SIGHUP) now
        # take the EOF branch: cancel, reap the group, then exit.
        raise KeyboardInterrupt

    for name in ("SIGTERM", "SIGHUP"):
        number = getattr(signal, name, None)
        if number is not None:
            try:
                signal.signal(number, _terminate)
            except (OSError, ValueError):
                pass
    try:
        server.serve(sys.stdin.buffer)
    except KeyboardInterrupt:
        # Ctrl-C reaches this process too when it shares the terminal's process
        # group, so SIGINT and codex's SIGTERM arrive together: the second one
        # must not cut the reap short.
        for name in ("SIGINT", "SIGTERM", "SIGHUP"):
            number = getattr(signal, name, None)
            if number is not None:
                try:
                    signal.signal(number, signal.SIG_IGN)
                except (OSError, ValueError):
                    pass
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
