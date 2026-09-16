"""LLM client wrapper for multi-provider LLM subprocess calls."""

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from . import structured_output
from .provider_registry import parse_model_spec, get_provider, get_provider_timeout
from .providers.base import (
    PROFILE_GENERATE,
    StructuredOutputUnavailable,
    availability_from_failure,
    build_child_env,
)


class LLMClientError(Exception):
    """Raised when LLM client operations fail."""
    pass


class SubagentTimeoutError(LLMClientError):
    """Raised when subagent execution times out."""
    pass


# Error codes for consistent error reporting
ERROR_TIMEOUT = "TIMEOUT"
ERROR_PARSE_ERROR = "PARSE_ERROR"
ERROR_BINARY_NOT_FOUND = "BINARY_NOT_FOUND"
ERROR_SUBPROCESS_FAILED = "SUBPROCESS_FAILED"
ERROR_FILE_NOT_FOUND = "FILE_NOT_FOUND"
ERROR_PROMPT_TOO_LONG = "PROMPT_TOO_LONG"
ERROR_PROMPT_UNSAFE = "PROMPT_UNSAFE"

# Windows-only concern: POSIX exec passes argv verbatim (no reparsing) and
# has generous per-arg limits, so prompt length is never enforced there.
_IS_WINDOWS = os.name == "nt"

# Hard Windows command-line caps, measured against the FULL rendered command
# line (subprocess renders list argv via ``list2cmdline`` before
# ``CreateProcessW``) in UTF-16 code units — the unit Windows uses, so astral
# characters count twice. cmd.exe — which executes .cmd/.bat npm shims — caps
# the whole command line at 8,191; native CreateProcess caps at 32,767
# (including the terminating NUL).
CMDLINE_CAP_UTF16_BATCH = 8191
CMDLINE_CAP_UTF16_NATIVE = 32767

# Headroom reserved below the caps for overhead that cannot be measured from
# argv alone: the terminating NUL, the implicit `%COMSPEC% /c` wrapper that
# launches batch shims, and expansion the shim itself performs ("%_prog%"
# prefixes, %* re-substitution).
CMDLINE_UTF16_HEADROOM = 256

_BATCH_SUFFIXES = frozenset({".cmd", ".bat"})

# Characters cmd.exe treats specially when it reparses a batch-shim command
# line. Deliberately conservative (interim guard until the explicit `cmd /c`
# escaping path lands — launch-strategy step (c)): `"` breaks argument
# quoting (CreateProcess-style `\"` escapes are NOT honored by cmd.exe),
# `%`/`!` trigger variable expansion even inside quotes, CR/LF split
# commands, and `^&|<>()` become live once quoting is broken or the argument
# is unquoted.
_CMD_UNSAFE_CHARS = frozenset('"%!^&|<>()\r\n')

# npm cmd-shims dispatch to `node <target.js>` via a `%dp0%`-relative (or,
# in older shims, `%~dp0`-relative) quoted path; extract that target so the
# shim can be bypassed entirely.
_NODE_SHIM_TARGET_RE = re.compile(
    r'"%(?:~dp0|dp0%)[\\/]?(?P<rel>[^"%\r\n]+?\.(?:js|cjs|mjs))"',
    re.IGNORECASE,
)


def _resolve_node_shim_target(shim_path: Path) -> Optional[List[str]]:
    """Extract the ``node <cli.js>`` launch an npm batch shim dispatches to.

    npm-installed CLIs on Windows are ``.cmd`` shims whose payload line runs
    ``node`` on a ``%dp0%``-relative script (e.g.
    ``"%_prog%" "%dp0%\\node_modules\\pkg\\bin\\cli.js" %*``). Parsing that
    target lets us launch node directly and skip cmd.exe entirely.

    Returns the ``[node, script]`` argv prefix, or None when the shim does
    not parse as a node dispatcher (missing/unreadable target, or no node
    binary available) — the caller then falls back to the flagged shim.
    """
    try:
        content = shim_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    match = _NODE_SHIM_TARGET_RE.search(content)
    if match is None:
        return None

    # Rebuild the %dp0%-relative path with host separators so the parse also
    # behaves identically under test on POSIX.
    parts = [p for p in re.split(r"[\\/]+", match.group("rel")) if p and p != "."]
    if not parts:
        return None
    script = shim_path.parent.joinpath(*parts)
    try:
        if not script.is_file():
            return None
    except OSError:
        return None

    # Mirror the shim's own dispatch: prefer a node.exe next to the shim
    # (%dp0%\node.exe), else fall back to node from PATH.
    node = shim_path.parent / "node.exe"
    try:
        if node.is_file():
            return [str(node), str(script)]
    except OSError:
        pass
    node_on_path = shutil.which("node")
    if node_on_path is None:
        return None
    return [node_on_path, str(script)]


def _resolve_executable(name: str) -> Tuple[List[str], bool]:
    """Resolve a command name to an argv launch prefix via shutil.which.

    Returns ``(launcher_argv, is_batch_shim)`` where ``launcher_argv``
    replaces ``cmd[0]`` (usually one element; two for ``node <cli.js>``).
    When ``shutil.which`` finds nothing, the bare name is returned unchanged
    so downstream error text stays meaningful. Resolution makes Windows
    ``CreateProcess`` find npm-installed provider CLIs; on POSIX the
    resolved path is exactly what PATH lookup would have executed anyway.

    Batch-shim hazard: npm installs CLIs on Windows as ``.cmd``/``.bat``
    shims, which are executed by cmd.exe. cmd.exe REPARSES the command line
    (metacharacters/newlines in argv corrupt the prompt or can escape into
    executed commands — the BatBadBut / CVE-2024-24576 class) and caps it at
    8,191 chars. To avoid the shim entirely, prefer (a1) a sibling ``.exe``
    with the same stem, then (a2) the ``node <cli.js>`` target the npm shim
    dispatches to; only when neither exists is the shim path returned with
    ``is_batch_shim=True`` so the caller enforces the stricter prompt budget
    and the cmd.exe metacharacter guard.
    """
    resolved = shutil.which(name)
    if resolved is None:
        return [name], False
    path = Path(resolved)
    if path.suffix.lower() in _BATCH_SUFFIXES:
        sibling_exe = path.with_suffix(".exe")
        try:
            if sibling_exe.is_file():
                return [str(sibling_exe)], False
        except OSError:
            pass
        node_target = _resolve_node_shim_target(path)
        if node_target is not None:
            return node_target, False
        return [resolved], True
    return [resolved], False


def expand_env_home(provider_env: Dict[str, str]) -> Dict[str, str]:
    """Shell-expand a leading ``~`` in adapter/config-supplied env values.

    Nothing expands ``~`` on the way to a child: ``CODEX_HOME='~/.codex_work'``
    reaches codex verbatim, and codex 0.146.0 either refuses ("path does not
    exist") or — if a literal ``./~/.codex_work`` happens to exist — resolves it
    CWD-relative and runs **silently against the wrong auth directory**. The
    expansion belongs at the two spawn sites rather than in each adapter, and it
    is applied only to the provider/config **deltas**: ``os.environ`` values
    reached us from a shell that already expanded them, and re-expanding an
    inherited value would change bytes nobody asked us to touch.

    ``os.path.expanduser`` is a no-op on any string that does not start with
    ``~``, and on ``~unknown-user`` where the name does not resolve, so a value
    that is not a home-relative path passes through unchanged.
    """
    return {
        key: (os.path.expanduser(value) if isinstance(value, str) else value)
        for key, value in provider_env.items()
    }


class DispatchContext:
    """What one spawn learned from the adapter's own run-scoped context.

    ``refusal`` is a human-readable reason the dispatch may not be attempted at
    all — set only by an adapter that can prove the spawn would resolve to
    nothing. ``None`` means "spawn normally", which is every adapter that
    declares no context.
    """

    __slots__ = ("refusal",)

    def __init__(self, refusal: Optional[str] = None) -> None:
        self.refusal = refusal


@contextlib.contextmanager
def provider_dispatch_context(
    provider: Any,
    model: str,
    profile: str,
    *,
    anchor: Optional[str] = None,
):
    """Enter an adapter's run-scoped dispatch context, if it declares one.

    Lives beside :func:`expand_env_home` for the same reason: it belongs to the
    **two spawn sites**, not to each adapter, so a context that scopes argv and
    env together cannot be entered on one lane and skipped on the other.

    The hook is duck-typed (``hermetic_dispatch``) rather than a base-class
    method because exactly one adapter needs it — opencode, whose config layering
    is the thing being contained. Every other adapter yields an empty context and
    renders byte-identical argv and env.
    """
    enter = getattr(provider, "hermetic_dispatch", None)
    if not callable(enter):
        yield DispatchContext()
        return
    with enter(profile, model, anchor=anchor) as refusal:
        yield DispatchContext(refusal)


def _utf16_code_units(s: str) -> int:
    """Length of ``s`` in UTF-16 code units (astral characters count as 2).

    Windows command-line caps are defined over the UTF-16 command line handed
    to ``CreateProcessW``, not over Python code points.
    """
    return len(s.encode("utf-16-le")) // 2


def _prompt_length_error(
    cmd: List[str],
    prompt: str,
    provider_name: str,
    model: str,
    is_batch_shim: bool
) -> Optional[Dict[str, Any]]:
    """Return a structured PROMPT_TOO_LONG error dict, or None if within budget.

    Guard for providers whose prompt travels on argv (see
    ``LLMProvider.prompt_transport``): exceeding the cmd.exe/CreateProcess
    command-line caps would otherwise fail with a cryptic
    ``[WinError 206]``/cmd.exe error. Callers gate on ``_IS_WINDOWS``.

    The check measures what Windows actually limits: the FULL command line
    that ``subprocess`` renders from ``cmd`` via ``list2cmdline`` (executable
    path, provider flags, quoting expansion, and the prompt), counted in
    UTF-16 code units, with headroom below the hard cap for overhead that
    cannot be measured from argv alone. Checking ``len(prompt)`` in isolation
    would pass prompts that still blow the cap once argv overhead, quote
    escaping, and astral characters are accounted for.
    """
    cap = CMDLINE_CAP_UTF16_BATCH if is_batch_shim else CMDLINE_CAP_UTF16_NATIVE
    limit = cap - CMDLINE_UTF16_HEADROOM
    cmdline_units = _utf16_code_units(subprocess.list2cmdline(cmd))
    if cmdline_units <= limit:
        return None
    prompt_units = _utf16_code_units(prompt)
    launcher = (
        "a .cmd/.bat shim run via cmd.exe (8,191-unit command-line cap)"
        if is_batch_shim
        else "a native executable (32,767-unit CreateProcess cap)"
    )
    return {
        "success": False,
        "error": (
            f"The fully rendered {provider_name} command line is "
            f"{cmdline_units} UTF-16 code units ({prompt_units} from the "
            f"prompt), but {provider_name} receives the prompt on the "
            f"command line and resolves to {launcher} on Windows; the safe "
            f"limit is {limit} units. Shorten the prompt — e.g. reference "
            f"large files by path instead of embedding their contents "
            f"inline."
        ),
        "error_code": ERROR_PROMPT_TOO_LONG,
        "details": {
            "provider": provider_name,
            "model": model,
            "prompt_chars": len(prompt),
            "prompt_utf16_units": prompt_units,
            "cmdline_utf16_units": cmdline_units,
            "cmdline_utf16_limit": limit,
        }
    }


def _batch_shim_metachar_error(
    prompt: str,
    provider_name: str,
    model: str,
    shim_path: str
) -> Optional[Dict[str, Any]]:
    """Return a structured PROMPT_UNSAFE error dict, or None if the prompt is safe.

    Interim safeguard for launch-strategy step (c): when a provider still
    resolves to a ``.cmd``/``.bat`` shim (no sibling ``.exe``, no parseable
    node target), cmd.exe reparses the command line on launch, so a prompt
    carrying cmd.exe metacharacters can be silently corrupted or escape into
    executed commands (BatBadBut / CVE-2024-24576 class). Until the explicit
    ``cmd /c`` escaping path lands, reject such prompts with a structured
    error rather than launching. Callers gate on ``_IS_WINDOWS``.
    """
    unsafe = sorted(set(prompt) & _CMD_UNSAFE_CHARS)
    if not unsafe:
        return None
    chars = ", ".join(repr(ch) for ch in unsafe)
    return {
        "success": False,
        "error": (
            f"{provider_name} resolves to a cmd.exe batch shim ({shim_path}) "
            f"and the prompt contains cmd.exe metacharacters ({chars}). "
            f"cmd.exe reparses the shim's command line, so these characters "
            f"could corrupt the prompt or escape into executed commands "
            f"(BatBadBut / CVE-2024-24576 class); refusing to launch via the "
            f"shim. Fix: install a native {provider_name} executable, or "
            f"remove these characters from the prompt."
        ),
        "error_code": ERROR_PROMPT_UNSAFE,
        "details": {
            "provider": provider_name,
            "model": model,
            "executable": shim_path,
            "unsafe_characters": unsafe,
        }
    }


MAX_LOGGED_PROMPT_LENGTH = 5000

#: §4.4's storage caps. A *successful* ``review-plan`` against a provider
#: emitting 3 MiB per stream wrote a 6,254,326 B ``log_gemini_*.txt``: this
#: function truncated the prompt and nothing else. Lane B already ``_tail``s its
#: failure captures at 64 KiB, so the log path is the one that was uncapped.
#:
#: Each stream keeps its **last** ``MAX_LOG_TAIL_LINES`` lines with every line
#: cut to ``MAX_LOG_LINE_BYTES``; the per-stream and whole-file byte budgets are
#: the backstops that make the ~100 KB total a property of the written file
#: rather than of the input. Measured against the captured 6 MB artifact,
#: 64 KiB/line + last-20 alone took it to ~17 KB.
MAX_LOG_LINE_BYTES = 64 * 1024
MAX_LOG_TAIL_LINES = 20
MAX_LOG_TOTAL_BYTES = 100 * 1024
#: Two streams plus a ≤5 KB prompt and the header stay inside the total.
MAX_LOG_STREAM_BYTES = 45 * 1024


def _redact(text: str) -> str:
    """§4.5's one redaction boundary, imported late.

    ``write_safety`` reaches this module through ``subagent_tasks``, so a
    module-level import here would close an import cycle. The call is on a
    per-invocation path, not a per-line one.
    """
    try:  # pragma: no cover - import shim mirrors the rest of utils/
        from .write_safety import redact_text
    except ImportError:  # pragma: no cover - direct-script import shim
        from utils.write_safety import redact_text
    return redact_text(text)


def _cut_bytes(text: str, limit: int, marker: str) -> str:
    """``text`` cut to ``limit`` UTF-8 bytes, head kept, ``marker`` appended.

    The marker is charged against ``limit`` rather than added on top, so a cap
    stated as a number of bytes really is one — a "~100 KB" file that comes out
    at 100 KB plus a footer is a cap nobody can assert on.
    """
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return text
    keep = max(0, limit - len(marker.encode("utf-8")))
    return encoded[:keep].decode("utf-8", errors="ignore") + marker


def _bound_stream(text: str) -> str:
    """One captured stream, bounded per §4.4 — **after** it has been redacted.

    Order is normative and is the reason L3 and L4 are one atomic change: every
    cut below slices by construction, and a cut applied first leaves a remnant no
    pattern matches. The last-20-lines rule would decapitate a PEM block's
    ``-----BEGIN`` header and keep its body; the per-line cut would keep a value
    whose key name it had just removed. Neither is recoverable downstream.
    """
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) > MAX_LOG_TAIL_LINES:
        dropped = len(lines) - MAX_LOG_TAIL_LINES
        lines = [
            f"... [{dropped} earlier line(s) truncated; last {MAX_LOG_TAIL_LINES} kept]"
        ] + lines[-MAX_LOG_TAIL_LINES:]
    bounded = [
        _cut_bytes(line, MAX_LOG_LINE_BYTES, f" ... [line truncated at {MAX_LOG_LINE_BYTES} B]")
        for line in lines
    ]
    return _cut_bytes(
        "\n".join(bounded),
        MAX_LOG_STREAM_BYTES,
        f"\n... [stream truncated at {MAX_LOG_STREAM_BYTES} B]",
    )


def _attach_stdout_excerpt(details: Dict[str, Any], stdout: Optional[str]) -> None:
    """Put the run's stdout on ``details`` as ``stdout_excerpt``, redacted then bounded.

    **Why a helper and not a third copy.** ``invoke_with_provider`` loses a run
    three ways — a non-zero exit, a timeout kill, and a zero exit whose output
    would not parse — and all three are cases where the provider may already have
    said the thing the caller needs. The rule for carrying that text out is not
    one line but four decisions (``_redact`` *before* ``_bound_stream``, per
    §4.5/§4.4; ``.strip()``; set the key only when something survives; never the
    raw stream), and this module argues elsewhere that a rule spelled three times
    is a rule that drifts twice — see ``_bound_stream``'s own note that the
    redact/bound order "is normative", and §2's forward-maintenance case against
    two hand-aligned blocks plus a comment. A cap tightened or a pattern added at
    one copy and not the others would leak from whichever site was forgotten, and
    the leak would be invisible until an artifact was read by a human. So it is
    written once, here, and the three sites call it.

    **Why the key is omitted rather than set empty.** Consumers test
    ``details.get("stdout_excerpt")`` for "did the provider say anything we can
    still use" — salvage gating at ``code_review_orchestrator.py``'s site A reads
    it ahead of ``details["stderr"]`` for exactly that. An always-present key
    holding ``""`` would make every one of those call sites carry a second
    emptiness check, and the first one written without it would treat "the
    provider printed nothing" as "the provider printed something". Absent means
    absent.

    Args:
        details: The result envelope's ``details`` dict, mutated in place.
        stdout: The raw captured stdout, or ``None`` when the path never had it
            (a ``TimeoutExpired`` that carried no partial output).
    """
    excerpt = _bound_stream(_redact(stdout or "")).strip()
    if excerpt:
        details["stdout_excerpt"] = excerpt


def _save_log(
    log_file: Union[str, Path],
    model: str,
    prompt: str,
    stdout: str,
    stderr: str,
    returncode: int,
    success: bool,
    error: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    result_source: Optional[str] = None,
    stdout_parse_error: Optional[str] = None,
) -> bool:
    """
    Save agent output to a log file for debugging.

    Args:
        log_file: Path to save the log
        model: Model identifier, ``"<provider>:<model>"`` where available —
            the header names the provider half, so a bare model string (no
            ``:``) still renders, just without a provider name to show.
        prompt: The prompt sent to the agent
        stdout: Standard output from the process
        stderr: Standard error from the process
        returncode: Process return code
        success: Whether the invocation was successful — the FINAL outcome,
            not necessarily the stdout extractor's verdict (see
            ``stdout_parse_error``/``result_source``).
        error: Error message if failed
        duration_seconds: How long the agent ran
        result_source: Where the caller's data ultimately came from —
            ``"file"``, ``"file_extracted"``, ``"stdout_fallback"``, or
            ``None`` when the caller has no secondary channel to name.
        stdout_parse_error: Set when the stdout extractor itself failed, even
            if a secondary channel (the file) went on to salvage the run —
            the two fields are independent because a failed stdout parse and
            an overall success are not a contradiction.

    Returns:
        True if log was saved successfully, False otherwise

    Every rendered field is passed through §4.5's ``redact_text`` **first** and
    bounded by §4.4's caps **second**. This is the sink where all nine planted
    secret shapes landed on a *successful* run, and the sink that wrote a
    6,254,326 B log for one ``review-plan``.
    """
    try:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # §4.5 before §4.4: redact the fully rendered text, then cap it. Reversing
        # these two would slice a secret across a cut and leave a remnant nothing
        # downstream can match.
        prompt = _redact(prompt or "")
        stdout = _bound_stream(_redact(stdout or ""))
        stderr = _bound_stream(_redact(stderr or ""))
        error = _redact(error) if error else error

        # Provider-neutral (§Fix1): every adapter shares this sink, so the
        # header names the provider the model spec actually carries rather
        # than a fixed "CURSOR-AGENT" left over from the original single-CLI
        # implementation. A bare model string (no ``provider:model`` split)
        # falls back to naming itself.
        provider_label = (model or "").split(":", 1)[0].strip().upper() or "PROVIDER"

        lines = [
            "=" * 80,
            f"{provider_label} LOG",
            "=" * 80,
            f"Timestamp: {datetime.now().isoformat()}",
            f"Model: {model or 'default'}",
            f"Success: {success}",
            f"Return Code: {returncode}",
        ]

        if duration_seconds is not None:
            lines.append(f"Duration: {duration_seconds:.1f}s")

        if result_source:
            lines.append(f"Result source: {result_source}")

        if stdout_parse_error:
            lines.append(f"Stdout parse: failed ({stdout_parse_error})")

        if error:
            lines.append(f"Error: {error}")

        truncated_prompt = prompt[:MAX_LOGGED_PROMPT_LENGTH]
        if len(prompt) > MAX_LOGGED_PROMPT_LENGTH:
            truncated_prompt += "..."

        lines.extend([
            "",
            "-" * 40,
            "PROMPT",
            "-" * 40,
            truncated_prompt,
            "",
            "-" * 40,
            "STDOUT",
            "-" * 40,
            stdout if stdout else "(empty)",
            "",
            "-" * 40,
            "STDERR",
            "-" * 40,
            stderr if stderr else "(empty)",
            "",
            "=" * 80,
            "",
        ])

        # The whole-file backstop, so "~100 KB" is a property of the bytes on
        # disk rather than an inference from the per-stream budgets. Safe to cut
        # blind: everything above was redacted before it got here.
        rendered = _cut_bytes(
            "\n".join(lines),
            MAX_LOG_TOTAL_BYTES,
            f"\n... [log truncated at {MAX_LOG_TOTAL_BYTES} B]\n",
        )

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(rendered)

        return True
    except (IOError, OSError) as e:
        print(f"WARNING: Failed to save log to {log_file}: {e}")
        return False


def _served_model_attribution(
    provider: Any, model: str, stdout: str
) -> Optional[Tuple[str, bool, Optional[str]]]:
    """§3.6's ``(served_model, is_fallback, warning)`` for one attempt, or None.

    **Gated on a configured fallback list, which ships empty.** Attribution is
    mandatory *whenever a fallback is configured* — without it a report
    attributes a haiku answer to the sonnet the user asked for. With nothing
    configured there is no substitution to disclose, so nothing is parsed,
    nothing is recorded and nothing is printed: the result dict is byte-for-byte
    what it has always been.

    Only claude-code answers here (its envelope carries ``modelUsage``); §4.6
    stays closed, so every other adapter lacks both methods and returns None.

    **Read, never depended on.** Any failure to answer degrades to None — an
    attribution field never fails an otherwise-successful attempt.
    """
    resolve = getattr(provider, "resolve_served_model", None)
    configured = getattr(provider, "fallback_models", None)
    if resolve is None or configured is None:
        return None
    try:
        fallbacks = configured(model)
        if not fallbacks:
            return None
        return resolve(stdout, model, fallbacks)
    except Exception:  # noqa: BLE001 - attribution never fails a good attempt
        return None


def invoke_with_provider(
    prompt: str,
    model_spec: str,
    timeout: Optional[int] = None,
    log_file: Optional[Union[str, Path]] = None,
    cwd: Optional[str] = None,
    prompt_transport: Optional[str] = None,
    anchor: Optional[Union[str, Path]] = None,
    schema_phase: Optional[str] = None,
    _defer_parse_failure_log: bool = False,
) -> Dict[str, Any]:
    """Invoke an LLM using the provider abstraction.

    This is the primary function for invoking LLM providers. It handles
    provider lookup, command building, subprocess execution, and output
    parsing with consistent error reporting.

    Args:
        prompt: The prompt to send to the LLM.
        model_spec: Model specification in "provider:model" or "model" format.
                   If no provider prefix is given, uses the default provider
                   from providers.yaml configuration.
        timeout: Optional timeout override in seconds. Uses provider default
                if not specified.
        log_file: Optional path to save full output log for debugging.
        prompt_transport: Optional non-argv transport override (D2d) —
            ``"stdin"`` writes the prompt to the child's stdin and omits it from
            argv, so no part of it is measured against the Windows
            command-line cap. Defaults to the adapter's ``prompt_transport``
            (``"argv"`` for every adapter today), keeping existing invocations
            byte-identical. Refused with SUBPROCESS_FAILED when the adapter does
            not declare support, rather than silently falling back to argv and
            re-introducing the cap the caller was avoiding.
        anchor: Optional discovery anchor (the run's plan-file path). Threaded
            into BOTH ``parse_model_spec`` and ``get_provider_timeout`` so a bare
            (prefix-less) spec resolves its ``default_provider`` — and its
            timeout — against the plan-derived project config rather than CWD's.
            This is the spawn site: unanchored, a run started from a foreign CWD
            launches a different binary than the project configured (§9.3.c).
        schema_phase: §4.2's phase name (``review-plan``, ``code-review``, …),
            or ``None``. **``None`` is the default and changes nothing**: no
            schema is resolved, no flag is emitted and the returned dict is
            byte-for-byte what it has always been.

            When set, this is §4.2.2's "the caller reads the flag before
            dispatch" leg: the phase's schema file is resolved through
            ``utils.structured_output``, the adapter's
            ``structured_output_support`` decides whether it is passed at all,
            codex's sidecar is written and removed in a ``finally``, and
            §4.2.1's failure policy is applied to the envelope afterwards. The
            verdict lands on ``details['structured_output']`` as ``native`` /
            ``recovered`` / ``unvalidated``; a refusal replaces the result with
            a failure carrying ``details['failure_reason']``.

            **Disposition (W1): DEFER — repository owner, 2026-08-20.** No
            production caller passes ``schema_phase`` today; the only callers
            are in ``tests/test_structured_output.py``, and that is the standing
            check. Lane A's fan-out phases are deliberately *not* wired, because
            Lane A's six shipped documents are array-rooted and claude-code
            cannot natively carry an array-rooted schema at all — wiring the
            callers first buys one WARN per dispatch and no native transport.
            Object-rooting those six files is a contract change for every
            consumer of them and is explicitly not a change a call site may
            take, so it is the repository owner's to make and it has not been
            made.

            **Re-check trigger:** claude-code accepts an array-rooted
            ``--json-schema``. **Re-read date:** 2027-02-20. Until one of those
            fires, wiring a production caller here is a regression, not
            progress. See ``plans/harness-residual-work.md`` §3, W1.
        _defer_parse_failure_log: Internal (§Fix1/F13). When ``log_file`` is
            set and the child exited 0, writing the log here — before
            :func:`invoke_with_file_output` has chosen between the file
            channel and the stdout fallback — would either freeze
            ``Success: False`` on a run the file channel goes on to resolve
            (§Fix1's original bug), or, on a clean parse, write a log with no
            ``Result source:`` line at all because this function does not yet
            know which channel will end up answering (F13). Set by that
            caller only: skips the immediate ``_save_log`` write for every
            exit-0 outcome and hands the raw capture back on
            ``details["_deferred_log"]`` instead, so the one log entry gets
            written once the final outcome — file, stdout fallback, or both
            failed — is known, and always carries `Result source:`. A hard
            failure (binary not found, timeout, non-zero exit) still logs
            immediately, unaffected by this flag — there is no channel choice
            to wait for.

    Returns:
        Dict with 'success' key and either:
        - On success: 'data' key with parsed response, plus 'details' dict
        - On failure: 'error' message, 'error_code', and 'details' dict

    Error codes:
        BINARY_NOT_FOUND: Provider CLI tool not found in PATH
        TIMEOUT: Command timed out
        SUBPROCESS_FAILED: Command exited with non-zero code (or failed to launch)
        PARSE_ERROR: Failed to parse output
        PROMPT_TOO_LONG: The fully rendered command line (executable path,
            flags, and prompt) exceeds the Windows command-line length budget
        PROMPT_UNSAFE: Prompt carries cmd.exe metacharacters and the provider
            only resolves to a Windows .cmd/.bat shim
    """
    provider_name, model = parse_model_spec(model_spec, anchor=anchor)
    provider = get_provider(provider_name)

    if provider is None:
        return {
            "success": False,
            "error": f"Unknown provider: {provider_name}",
            "error_code": ERROR_BINARY_NOT_FOUND,
            "details": {
                "provider": provider_name,
                "model": model,
            }
        }

    # M10 (§5.1) — a **bool** consumer, and it stays one. The gate is still
    # `is_available()` verbatim, so this refusal is byte-for-byte the decision it
    # always was; the typed report is read only *inside* the branch, to put the
    # state and its sentence on the error details a caller renders.
    if not provider.is_available():
        availability = provider.describe_availability()
        return {
            "success": False,
            "error": f"{provider_name} CLI not found in PATH",
            "error_code": ERROR_BINARY_NOT_FOUND,
            "details": {
                "provider": provider_name,
                "model": model,
                "availability_state": availability.state.value,
                "availability_detail": availability.detail,
            }
        }

    # Use provided timeout or provider's default
    effective_timeout = timeout or get_provider_timeout(provider_name, anchor=anchor)

    # Build command using provider, resolving cmd[0] to an absolute path
    # (and past any Windows .cmd/.bat npm shim, to a sibling .exe or the
    # shim's `node <cli.js>` target — see _resolve_executable).
    transport = prompt_transport or provider.prompt_transport
    if transport != "argv" and not provider.supports_prompt_transport(transport):
        return {
            "success": False,
            "error": (
                f"{provider_name} does not support the {transport!r} prompt "
                f"transport (supported: "
                f"{', '.join(provider.supported_prompt_transports)})"
            ),
            "error_code": ERROR_SUBPROCESS_FAILED,
            "details": {"provider": provider_name, "model": model},
        }
    # §4.2.2's pre-dispatch read, and M7's one call site. `schema_phase=None`
    # skips all of it, so every existing caller renders the same argv it always
    # did. `schema_for_dispatch` is what consults `structured_output_support`:
    # an adapter that declares `False` — or a schema this CLI provably cannot
    # carry — comes back `None` here and the phase runs its prompt-level JSON
    # path, marked `recovered`/`unvalidated` and never `native`.
    schema = None
    structured_marker = structured_output.MARKER_UNVALIDATED
    # M4/§2.3 — the adapter's run-scoped dispatch context is entered BEFORE argv
    # is built, because argv and env are rendered from the same context state
    # (opencode's `--pure` pairs with its `XDG_CONFIG_HOME` redirect and neither
    # is emitted without the other). It rides the same `ExitStack` as the schema
    # sidecar, so it is left on success, on timeout, on kill-path teardown and on
    # exception alike — the `finally` half of §2.3's lifecycle, with the
    # adapter's own `atexit` sweep as the backstop.
    # Every post-entry setup step — the schema read, the argv build, the
    # executable resolution and the env composition — runs INSIDE this `with`,
    # so a raise anywhere in it unwinds the context here rather than leaving
    # opencode's active context and its config-home reference behind for the
    # `atexit` sweep to find. Only once setup has survived is the live stack
    # handed to `schema_cleanup` (`pop_all`), whose owner from that point on is
    # the subprocess `finally` at the end of this function.
    with contextlib.ExitStack() as setup_cleanup:
        dispatch = setup_cleanup.enter_context(
            provider_dispatch_context(provider, model, PROFILE_GENERATE, anchor=anchor)
        )
        if dispatch.refusal:
            return {
                "success": False,
                "error": dispatch.refusal,
                "error_code": ERROR_SUBPROCESS_FAILED,
                "details": {
                    "provider": provider_name,
                    "model": model,
                    "retryable": False,
                },
            }
        if schema_phase:
            try:
                schema, structured_marker = structured_output.schema_for_dispatch(
                    provider, schema_phase
                )
                if schema is not None and provider.structured_output_transport == "file":
                    # One `mkdtemp` per invocation, written by the CALLER: adapters
                    # are singletons under a concurrent fan-out and cannot hold this.
                    # The stack's `close()` in the `finally` below removes it on
                    # success, on timeout, on kill-path teardown and on exception.
                    schema = setup_cleanup.enter_context(
                        structured_output.schema_sidecar(schema)
                    )
            except structured_output.SchemaConfigError as exc:
                return {
                    "success": False,
                    "error": f"structured output: {exc}",
                    "error_code": ERROR_SUBPROCESS_FAILED,
                    "details": {
                        "provider": provider_name,
                        "model": model,
                        "failure_reason": structured_output.FAILURE_CONFIG_ERROR,
                        "retryable": False,
                    },
                }

        # `timeout_s=effective_timeout` is L5's one call site (§9.6.b): the number is
        # already resolved here, so an adapter with an internal cutoff of its own
        # (agy's `--print-timeout`) can hold it under the deadline this same call
        # passes to `subprocess.run` below, instead of hardcoding a value that has to
        # be kept in sync by hand. Twelve of the thirteen adapters ignore it, and
        # dropping the keyword restores every rendered argv byte-for-byte.
        try:
            cmd, stdin_text = provider.build_command_for_transport(
                prompt, model, transport, timeout_s=effective_timeout, schema=schema
            )
        except StructuredOutputUnavailable as exc:
            # §4.2.1's sixth case: fails BEFORE spawn, so no partial argv reaches
            # the CLI. A configuration error, never retried.
            return {
                "success": False,
                "error": str(exc),
                "error_code": ERROR_SUBPROCESS_FAILED,
                "details": {
                    "provider": provider_name,
                    "model": model,
                    "failure_reason": structured_output.FAILURE_CONFIG_ERROR,
                    "retryable": False,
                },
            }
        launcher, is_batch_shim = _resolve_executable(cmd[0])
        cmd = [*launcher, *cmd[1:]]

        if _IS_WINDOWS and transport == "argv":
            # Interim safeguard until the explicit `cmd /c` escaping path lands
            # (launch-strategy step (c)): a batch shim means cmd.exe reparses
            # the command line, so refuse metacharacter-bearing prompts rather
            # than risk silent corruption or injection. Never enforced on POSIX.
            if is_batch_shim:
                metachar_error = _batch_shim_metachar_error(
                    prompt, provider_name, model, cmd[0]
                )
                if metachar_error is not None:
                    return metachar_error

            # Prompt-on-argv transport hits hard command-line length caps on
            # Windows; fail fast with an actionable error instead of a cryptic
            # CreateProcess/cmd.exe failure. Measured over the full rendered
            # command line (executable path, flags, quoting expansion), not just
            # the prompt. Never enforced on POSIX.
            length_error = _prompt_length_error(cmd, prompt, provider_name, model, is_batch_shim)
            if length_error is not None:
                return length_error

        start_time = time.time()

        # Build subprocess environment: add provider vars, remove blacklisted ones.
        # `resolve_env`, not `get_env`: the Lane A fan-out IS the `generate` profile
        # (§2.2), so it gets the global non-interactive git block plus the scoped
        # `credential.helper=""` wipe. Naming the profile rather than leaning on the
        # default keeps the scoping greppable — `agent-write` must never appear here.
        # `cwd` rides along because an adapter's delta may carry a
        # workspace-scoped side effect (cursor's trust-marker guard): the spawn
        # site is where the child's real directory is known, and `None` here
        # means the child inherits this process's.
        provider_env = expand_env_home(
            provider.resolve_env(model, PROFILE_GENERATE, anchor=anchor, cwd=cwd)
        )
        # M3/M17 (A6): the *composition* is `build_child_env`, the one builder Lane B
        # goes through too, so a slot can never strip on one lane and inherit on the
        # other — and M17's deny-by-default allowlist has exactly one place to live.
        # Lane A passes no host guards: a fan-out slot is a child of this process,
        # not of the host's shell tool, so only the adapter's own list applies.
        env = build_child_env(provider, provider_env=provider_env, anchor=anchor)
        # Setup survived: transfer the entered contexts out of the setup
        # stack so leaving this `with` releases nothing, and the subprocess
        # `finally` below owns them for the rest of the dispatch.
        schema_cleanup = setup_cleanup.pop_all()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            # Provider output is not under our control: errors="replace" is
            # lossy for non-UTF-8 bytes but guarantees a total decode — the
            # standard choice for LLM/tool output. surrogateescape rejected
            # (lone surrogates raise on downstream UTF-8 re-encode);
            # backslashreplace rejected (noisier in model-visible output).
            encoding="utf-8",
            errors="replace",
            timeout=effective_timeout,
            env=env,
            cwd=cwd,
            # Under the argv transport no provider reads stdin, and closing it
            # prevents CLIs that poll stdin for piped input (e.g. agy) from
            # blocking forever when the orchestrator's stdin is a non-EOF pipe.
            # Under the stdin transport the prompt IS the child's stdin, and it
            # is closed at EOF once written, which those CLIs treat the same way.
            **(
                {"input": stdin_text}
                if stdin_text is not None
                else {"stdin": subprocess.DEVNULL}
            ),
        )
        duration = time.time() - start_time

        if result.returncode != 0:
            error_message = f"{provider_name} exited with code {result.returncode}"
            # Some CLIs report the actual cause on stdout and leave stderr
            # empty (opencode does this for auth/quota errors), which would
            # otherwise leave the caller with a bare exit code. A provider
            # override must never mask the real failure, so it is best-effort.
            try:
                reason = provider.describe_failure(result.stdout, result.stderr)
            except Exception:  # noqa: BLE001 - diagnostics must not raise
                reason = None
            if reason:
                error_message = f"{error_message}: {reason}"
            # M10 (§5.1) — the **status**-consumer half of this site. A failure
            # whose own streams carry a login notice is an availability verdict,
            # not a parse, so name it instead of leaving the caller with a
            # generic tail. Only the named negatives speak (see
            # `availability_from_failure`), and nothing is probed here.
            unavailable = availability_from_failure(
                provider, result.stdout, result.stderr, result.returncode
            )
            if unavailable is not None:
                error_message = f"{error_message} [{unavailable.value}]"

            # Save log on failure if requested
            if log_file:
                _save_log(
                    log_file=log_file,
                    model=f"{provider_name}:{model}",
                    prompt=prompt,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    returncode=result.returncode,
                    success=False,
                    error=error_message,
                    duration_seconds=duration
                )

            details = {
                "exit_code": result.returncode,
                "stderr": result.stderr,
                "provider": provider_name,
                "model": model,
                "duration_seconds": duration
            }
            # THE ANSWER MAY ALREADY BE ON STDOUT, and this return used to drop
            # it. A non-zero exit is not only "the provider never spoke": it is
            # also the provider that wrote a truncated or fenced array and *then*
            # died — a kill after the answer, a crash on teardown — and that array
            # is on stdout, never on stderr. The salvage gate at
            # `code_review_orchestrator.py`'s site A runs under exactly this
            # branch (`if not result["success"]`), so with stderr as its only
            # channel it printed "Provider failed before producing any JSON —
            # nothing to salvage" over a run that *had* produced JSON, and a
            # recoverable review was thrown away silently.
            #
            # The JSON-erroring providers make the same point from the other
            # side: `opencode run --format json` puts its failure shape on stdout
            # with stderr empty, and a relayed `{"error": {"code": 401, ...}}`
            # body looks identical from any HTTP-backed CLI. Those reached the
            # gate as the empty string, so even the verdicts it got *right* were
            # reached for the wrong reason.
            #
            # `provider.describe_failure` and `availability_from_failure` above
            # already read `result.stdout` for their own one-line verdicts; this
            # carries the text itself, under §4.5/§4.4, so a consumer can form a
            # different judgement from the same evidence.
            _attach_stdout_excerpt(details, result.stdout)

            if schema_phase:
                # §4.2.1 cases 2 and 4 both arrive here as a non-zero exit, and
                # they are NOT the same failure: the CLI exhausting its own
                # structured-output budget is a `schema_violation` our ladder
                # must not re-run, while a malformed schema is a configuration
                # error that made **zero requests** and cost nothing. Classifying
                # here is what keeps the second from being retried as if it were
                # transient.
                verdict = structured_output.classify_attempt(
                    schema,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    exit_code=result.returncode,
                    phase=schema_phase,
                    model=f"{provider_name}:{model}",
                )
                if verdict.failure_reason:
                    details["failure_reason"] = verdict.failure_reason
                    details["retryable"] = verdict.retryable
                if verdict.warning:
                    print(f"[WARN] {verdict.warning}", file=sys.stderr)
            return {
                "success": False,
                "error": error_message,
                "error_code": ERROR_SUBPROCESS_FAILED,
                "details": details,
            }

        # Parse output using provider
        parsed = provider.parse_output(result.stdout, result.stderr)

        # Add details to successful results
        parsed["details"] = {
            "provider": provider_name,
            "model": model,
            "duration_seconds": duration
        }

        # §3.6: a fallback that fired rides out on the same details dict the
        # provider/model pair travels on, so every downstream record that names
        # the requested model can name the model that actually served the turn.
        # One WARN per slot — the adapter owns the text; there is no second one.
        attribution = _served_model_attribution(provider, model, result.stdout)
        if attribution is not None:
            served, is_fallback, warning = attribution
            parsed["details"]["served_model"] = served
            parsed["details"]["served_model_is_fallback"] = is_fallback
            if warning:
                print(f"[WARN] {warning}", file=sys.stderr)

        # §4.2.1's rule, stated as a rule and not only as a case: **no**
        # combination of exit 0, `subtype: success` and `is_error: false` is by
        # itself a success for a schema-carrying phase. The only success is
        # `structured_output` present (→ `native`) or a recovered document that
        # validates against the SAME schema file (→ `recovered`, one WARN).
        # Anything else is `schema_violation`, handed to the retry ladder like
        # any other parse failure. Applied here, on the parsed envelope, so no
        # adapter can silently upgrade prose to validated output.
        if schema_phase:
            verdict = structured_output.classify_attempt(
                schema,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                phase=schema_phase,
                model=f"{provider_name}:{model}",
            )
            if verdict.warning:
                print(f"[WARN] {verdict.warning}", file=sys.stderr)
            if verdict.failure_reason:
                parsed = {
                    "success": False,
                    "error": (
                        f"{provider_name} returned no schema-conformant document "
                        f"for {schema_phase} ({verdict.failure_reason})"
                    ),
                    "error_code": ERROR_PARSE_ERROR,
                    "data": None,
                    "details": {
                        **parsed.get("details", {}),
                        "failure_reason": verdict.failure_reason,
                        "retryable": verdict.retryable,
                        "structured_output": None,
                    },
                }
            else:
                parsed["details"]["structured_output"] = (
                    verdict.marker or structured_marker
                )
                if verdict.data is not None:
                    parsed["success"] = True
                    parsed["data"] = verdict.data

        # If parse_output indicates failure, add error_code
        if not parsed.get("success", False) and "error_code" not in parsed:
            parsed["error_code"] = ERROR_PARSE_ERROR

        # WHAT THE MODEL ACTUALLY SAID, kept on the failure. A parse failure is
        # the one class where the error message describes our disappointment
        # rather than the event: "JSON parsing failed from both file and stdout"
        # is what a cursor `--mode ask` write refusal looked like from the
        # outside, and reading the sentence that caused it ("I can't write the
        # requested file while Ask mode is active") cost three 450 s runs and a
        # hand-driven CLI invocation, because nothing downstream carried the
        # text. Redacted and bounded by the same §4.5/§4.4 boundary the log
        # writer uses — never the raw stream. This is the *original* of the three
        # `stdout_excerpt` sites; `_attach_stdout_excerpt` now spells the rule for
        # all of them, and the two hard-failure returns above and below join it,
        # so `details.get("stdout_excerpt")` means the same thing whichever way
        # the run was lost.
        if not parsed.get("success", False):
            _attach_stdout_excerpt(parsed.setdefault("details", {}), result.stdout)

        # Save log if requested. §Fix1/F13: a deferring caller (invoke_with_file_output)
        # gets the raw capture on `details` instead of an immediate write, on
        # EVERY exit-0 outcome — not only a failed stdout parse — because this
        # function does not yet know which channel (file, stdout fallback, or
        # neither) will end up answering for the run, and that is exactly what
        # `Result source:` in the log is supposed to name. Writing here
        # unconditionally on a clean parse used to freeze a log with no
        # `Result source:` line at all: the caller's file-vs-stdout decision
        # came later and had nothing left to add to. Only the deferring caller
        # actually reads `_deferred_log`, so a plain `invoke_with_provider`
        # call (no defer requested) still gets one immediate write.
        if log_file:
            if _defer_parse_failure_log:
                parsed.setdefault("details", {})["_deferred_log"] = {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "duration_seconds": duration,
                }
            else:
                _save_log(
                    log_file=log_file,
                    model=f"{provider_name}:{model}",
                    prompt=prompt,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    returncode=result.returncode,
                    success=parsed.get("success", False),
                    error=parsed.get("error"),
                    duration_seconds=duration
                )

        return parsed

    except subprocess.TimeoutExpired as e:
        duration = time.time() - start_time
        # TimeoutExpired may carry partial output as str or bytes; decode
        # bytes with the same utf-8/replace policy as the run() call above.
        if e.stdout is None:
            stdout = ""
        elif isinstance(e.stdout, str):
            stdout = e.stdout
        else:
            stdout = e.stdout.decode("utf-8", errors="replace")
        if e.stderr is None:
            stderr = ""
        elif isinstance(e.stderr, str):
            stderr = e.stderr
        else:
            stderr = e.stderr.decode("utf-8", errors="replace")

        # Save log on timeout if requested
        if log_file:
            _save_log(
                log_file=log_file,
                model=f"{provider_name}:{model}",
                prompt=prompt,
                stdout=stdout,
                stderr=stderr,
                returncode=-1,
                success=False,
                error=f"{provider_name} timed out after {effective_timeout}s",
                duration_seconds=duration
            )

        details = {
            "provider": provider_name,
            "model": model,
            "timeout": effective_timeout,
            "duration_seconds": duration
        }
        # A TIMEOUT KILL DOES RECOVER PARTIAL STDOUT, so this return no longer
        # throws it away — checked rather than assumed, because attaching a key
        # that is empty by construction would be worse than attaching nothing.
        # `subprocess.run` re-raises a `TimeoutExpired` that already carries the
        # bytes read off the pipe before the deadline: on POSIX
        # `Popen._communicate` populates the exception as it raises, and on
        # Windows `run` kills the child and calls `communicate()` again to collect
        # them. That is the `e.stdout` the decode a few lines above has just
        # turned into the `stdout` local — the same local `_save_log` is handed,
        # so the log file has always kept this text and only the returned envelope
        # did not. The envelope is the half a caller can act on.
        #
        # This is the archetype §2 of the plan names for salvage site A: a
        # provider that streamed a review array and was killed mid-emission
        # leaves a *truncated* array on stdout and nothing at all on stderr. It is
        # the most salvageable failure this skill has, and the stderr-only gate
        # was structurally unable to see it. Bounding matters more here than at
        # either sibling site, because the run killed at the deadline is the run
        # most likely to have emitted megabytes — the 6,254,326 B log `_save_log`
        # was written to survive came off a path like this one.
        #
        # When the kill caught the child before its first write, `stdout` is the
        # empty string rather than absent; the helper omits the key rather than
        # attaching `""`, so this path and the non-zero-exit path answer
        # `details.get("stdout_excerpt")` the same way.
        _attach_stdout_excerpt(details, stdout)

        return {
            "success": False,
            "error": f"{provider_name} timed out after {effective_timeout}s",
            "error_code": ERROR_TIMEOUT,
            "details": details,
        }

    except OSError as e:
        # Process creation failed (executable removed after detection, an
        # invalid resolved shim, or any other CreateProcess/exec failure):
        # return a structured error through the same result-dict path as
        # other provider errors instead of aborting the orchestrator.
        duration = time.time() - start_time
        if isinstance(e, FileNotFoundError):
            error_code = ERROR_BINARY_NOT_FOUND
            error_msg = f"{provider_name} CLI not found when launching: {e}"
        else:
            error_code = ERROR_SUBPROCESS_FAILED
            error_msg = f"Failed to launch {provider_name} CLI: {e}"

        # Save log on launch failure if requested
        if log_file:
            _save_log(
                log_file=log_file,
                model=f"{provider_name}:{model}",
                prompt=prompt,
                stdout="",
                stderr=str(e),
                returncode=-1,
                success=False,
                error=error_msg,
                duration_seconds=duration
            )

        return {
            "success": False,
            "error": error_msg,
            "error_code": error_code,
            "details": {
                "provider": provider_name,
                "model": model,
                "duration_seconds": duration
            }
        }

    finally:
        # §4.2.1: codex's per-invocation schema directory is removed in a
        # `finally` so it is cleaned up on success, on timeout, on kill-path
        # teardown and on exception alike. A no-op when `schema_phase` is unset
        # or the adapter's transport is inline.
        schema_cleanup.close()


def _is_valid_parsed_data(data: Any) -> bool:
    """Check if parsed data is valid JSON (not just a raw wrapper).

    Args:
        data: Parsed data from provider output

    Returns:
        True if data is a valid list or dict (not just a raw error wrapper)
    """
    if data is None or data == "" or data == {"raw": ""}:
        return False
    if not isinstance(data, (list, dict)):
        return False
    # Check if it's just a raw wrapper (single key "raw")
    if isinstance(data, dict) and set(data.keys()) == {"raw"}:
        return False
    return True


# ---------------------------------------------------------------------------
# The ``{output_location}`` block the three review prompts end with
# ---------------------------------------------------------------------------

#: What a file-capable adapter is asked for. Byte-identical to the text
#: ``prompts/{code,plan,task}_review.txt`` carried inline before the placeholder
#: existed, so the rendered prompt of every adapter that declares
#: ``supports_file_output`` is unchanged to the character — the golden container
#: turns under ``tests/golden/container_turns/`` are captured from those bytes.
FILE_OUTPUT_LOCATION = (
    "CRITICAL: Write your JSON output to this file: {path}\n"
    "Do NOT print JSON to stdout. Write ONLY the JSON array to the specified "
    "file path.\n"
    'After writing, confirm by printing: "JSON written to {path}"\n'
    "\n"
    "The file must contain ONLY valid JSON - no markdown, no code blocks, no "
    "explanation. Just the raw JSON array."
)

#: What an adapter that declares ``supports_file_output = False`` is asked for
#: instead. Two adapters run their CLI in a read-only chat mode on purpose
#: (cursor ``--mode ask``, aider ``/ask``), and the file instruction is one they
#: can only answer with a refusal sentence. Stdout is the channel they *do*
#: have, it is the one ``invoke_with_file_output`` already falls back to, and
#: the parsed array lands in exactly the same per-model result file either way —
#: ``save_model_result`` writes it from the returned data, not from the model's
#: own file.
STDOUT_OUTPUT_LOCATION = (
    "CRITICAL: Return your JSON output on stdout as your final message. Do NOT "
    "write it to a file — you are running in a read-only mode and any file "
    "write will be refused.\n"
    "Your final message must be ONLY the raw JSON array - no markdown, no code "
    "blocks, no explanation, no preamble and no closing remarks."
)


def render_output_location(
    output_json_path: Union[str, Path],
    model_spec: str,
    anchor: Optional[Union[str, Path]] = None,
) -> str:
    """The ``{output_location}`` block for ``model_spec``'s rendered prompt.

    One decision, taken from the adapter's declared
    :attr:`~utils.providers.base.LLMProvider.supports_file_output` rather than
    from a provider name spelled out in an orchestrator: the caller that renders
    a prompt does not need to know *which* CLIs refuse to write, only that this
    one does. An unknown provider (a config naming an adapter this build no
    longer ships) renders the file block, which is what every caller did before
    this function existed.

    ``anchor`` is the run's plan path (§9.3.c): a bare, prefix-less spec resolves
    its ``default_provider`` against the plan-derived project config, so the
    block matches the binary that will actually be spawned.
    """
    provider_name, _ = parse_model_spec(model_spec, anchor=anchor)
    provider = get_provider(provider_name)
    if provider is not None and not provider.supports_file_output:
        return STDOUT_OUTPUT_LOCATION
    return FILE_OUTPUT_LOCATION.format(path=output_json_path)


#: `result["source"]` values where the file on disk is what the model itself
#: produced. Anything else (chiefly `stdout_fallback`) means the answer came
#: over stdout and `output_file` is only where THIS PROCESS intends to persist
#: it later (`save_model_result`, run after the whole fan-out finishes) — see
#: `describe_output_file` (F14).
FILE_WRITTEN_SOURCES = frozenset({"file", "file_extracted"})


def describe_output_file(output_file: Union[str, Path], source: str) -> str:
    """The one line a fan-out prints about where a slot's answer lives (F14).

    ``[model] Output file: <path> (source: stdout_fallback)`` reads as "the
    model wrote this" — and for a provider with ``supports_file_output =
    False`` (or one that simply didn't write this time), it did not: the path
    is a promise this process is making to itself, not a fact about the
    child. Wording is picked from the *observed* ``source`` rather than the
    provider's declared capability, so a file-capable provider that happened
    to fall back to stdout this attempt still gets the accurate line.
    """
    verb = "Output file" if source in FILE_WRITTEN_SOURCES else "Result will be saved to"
    return f"{verb}: {output_file} (source: {source})"


def _finalize_file_output_log(
    log_file: Optional[Union[str, Path]],
    deferred_log: Optional[Dict[str, Any]],
    result: Dict[str, Any],
    model_spec: str,
    prompt: str,
    *,
    success: bool,
    error: Optional[str],
    result_source: Optional[str],
) -> None:
    """Write the one log entry :func:`invoke_with_file_output` owes (§Fix1/F13).

    A no-op when no log was requested, or when the run hard-failed before
    reaching the exit-0 branch that defers (there is no channel choice to
    report on those). Every exit-0 outcome defers now — a clean stdout parse
    included — so this is the single place that settles what the log
    ultimately says: the FINAL outcome — after the file channel and the
    stdout fallback both had their turn — never the stdout extractor's own
    opinion of a run it did not get the last word on, and always carrying
    ``Result source:`` (``file`` / ``file_extracted`` / ``stdout_fallback`` /
    absent on a double failure), which a clean-parse run used to lack
    entirely (F13): the immediate write inside ``invoke_with_provider`` ran
    before the file-vs-stdout choice existed to name.
    """
    if not log_file or deferred_log is None:
        return
    details = result.get("details", {})
    provider = details.get("provider")
    model = details.get("model")
    resolved_model = f"{provider}:{model}" if provider or model else model_spec
    # The stdout extractor's own verdict travels alongside the final one
    # rather than being overwritten by it — "Success: True" by itself would
    # hide that the file channel is what actually salvaged this run.
    stdout_parse_error = result.get("error") if not result.get("success") else None
    _save_log(
        log_file=log_file,
        model=resolved_model,
        prompt=prompt,
        stdout=deferred_log.get("stdout", ""),
        stderr=deferred_log.get("stderr", ""),
        returncode=deferred_log.get("returncode", -1),
        success=success,
        error=error,
        duration_seconds=deferred_log.get("duration_seconds"),
        result_source=result_source,
        stdout_parse_error=stdout_parse_error,
    )


def invoke_with_file_output(
    prompt_template: str,
    model_spec: str,
    prompt_context: Dict[str, Any],
    output_dir: Union[str, Path],
    phase: str,
    timeout: Optional[int] = None,
    log_file: Optional[Union[str, Path]] = None,
    prefer_arrays: bool = True,
    cwd: Optional[str] = None,
    anchor: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Invoke LLM with file-based JSON output.

    This function instructs the LLM to write JSON output to a specified file
    instead of relying on stdout parsing. The prompt is formatted with an
    {output_json_path} variable that the LLM should use, and an
    {output_location} block rendered from the adapter's declared
    ``supports_file_output`` — a CLI running in a read-only mode is asked for
    the array on stdout rather than for a file it is going to refuse (see
    :func:`render_output_location`). Both channels are read below either way.

    Fallback strategy:
    1. Primary: Read from output file (with extraction fallback for code blocks)
    2. Secondary: Parse stdout (current behavior)
    3. Tertiary: Return error for salvage request

    Args:
        prompt_template: Prompt template with {output_json_path} placeholder
        model_spec: Model specification (e.g., 'cursor-agent:auto')
        prompt_context: Dict of variables to substitute in the prompt template
        output_dir: Directory for output files
        phase: Operation phase for filename (e.g., 'code_review', 'plan_review')
        timeout: Optional timeout override in seconds
        log_file: Optional path to save full output log
        prefer_arrays: If True, prefer array candidates when extracting JSON
        anchor: Optional discovery anchor (the run's plan-file path), forwarded
            verbatim to :func:`invoke_with_provider` (§9.3.c)

    Returns:
        Dict with keys:
        - 'success': bool indicating overall success
        - 'data': Parsed JSON data (if successful)
        - 'source': Where JSON came from ('file', 'file_extracted', 'stdout_fallback')
        - 'output_file': Path to the output file (if generated)
        - 'error': Error message (if failed)
        - 'file_error': File-specific error (if file read failed)
        - 'details': Provider execution details, forwarded verbatim on every
          return path — including §3.6's ``served_model`` /
          ``served_model_is_fallback``, so the JSON's origin travels with it
          whichever channel (file, extracted file, stdout) supplied it
    """
    from .json_extractor import generate_output_path, read_json_from_file

    output_dir = Path(output_dir)

    # Generate output file path
    prefix = prompt_context.get("prefix", "output")
    output_json_path = generate_output_path(output_dir, prefix, phase, model_spec)

    # Inject output path into prompt context, plus the channel instruction that
    # path belongs to: an adapter whose CLI cannot write files is asked for the
    # array on stdout instead of being asked for a file it will refuse. The path
    # itself is still handed over (a template may name it outside the block, and
    # every caller-supplied template in the test suite does), and the file is
    # still read first below, so a provider that writes one anyway is unaffected.
    full_context = {
        **prompt_context,
        "output_json_path": str(output_json_path),
        "output_location": render_output_location(
            output_json_path, model_spec, anchor=anchor
        ),
    }

    # Format prompt
    try:
        prompt = prompt_template.format(**full_context)
    except KeyError as e:
        return {
            "success": False,
            "error": f"Missing prompt variable: {e}",
            "error_code": "PROMPT_FORMAT_ERROR",
            "data": None,
        }

    # Invoke provider. §Fix1: defer the log write for an exit-0 stdout parse
    # failure — this function may still salvage the run through the file
    # channel below, and the immediate write inside invoke_with_provider
    # cannot know that yet.
    result = invoke_with_provider(prompt, model_spec, timeout, log_file, cwd=cwd,
                                  anchor=anchor, _defer_parse_failure_log=True)
    deferred_log = result.get("details", {}).pop("_deferred_log", None)

    # Check for hard failures that mean the command didn't run properly
    # (as opposed to parse errors which may still have written to file)
    error_code = result.get("error_code", "")
    is_hard_failure = error_code in (
        ERROR_BINARY_NOT_FOUND,
        ERROR_TIMEOUT,
        ERROR_SUBPROCESS_FAILED,
        ERROR_PROMPT_TOO_LONG,
        ERROR_PROMPT_UNSAFE,
    )

    if not result.get("success") and is_hard_failure:
        # `deferred_log` is always None here — a hard failure never reaches
        # the exit-0 branch that sets it — so invoke_with_provider's own
        # immediate write (or lack of one) already stands as final.
        return {
            **result,
            "output_file": str(output_json_path),
        }

    # Try to read JSON from file first (even if stdout parsing failed)
    file_result = read_json_from_file(output_json_path, prefer_arrays=prefer_arrays)

    if file_result.get("success"):
        _finalize_file_output_log(
            log_file, deferred_log, result, model_spec, prompt,
            success=True, error=None,
            result_source=file_result.get("source", "file"),
        )
        return {
            "success": True,
            "data": file_result["data"],
            "source": file_result.get("source", "file"),
            "output_file": str(output_json_path),
            "details": result.get("details", {}),
        }

    # Fall back to stdout parsing (the data parsed by the provider)
    stdout_data = result.get("data")
    if _is_valid_parsed_data(stdout_data):
        _finalize_file_output_log(
            log_file, deferred_log, result, model_spec, prompt,
            success=True, error=None, result_source="stdout_fallback",
        )
        return {
            "success": True,
            "data": stdout_data,
            "source": "stdout_fallback",
            "output_file": str(output_json_path),
            "file_error": file_result.get("error"),
            "details": result.get("details", {}),
        }

    # Both failed
    _finalize_file_output_log(
        log_file, deferred_log, result, model_spec, prompt,
        success=False, error="JSON parsing failed from both file and stdout",
        result_source=None,
    )
    return {
        "success": False,
        "error": "JSON parsing failed from both file and stdout",
        "error_code": ERROR_PARSE_ERROR,
        "file_error": file_result.get("error"),
        "stdout_error": result.get("error", "No valid JSON in stdout"),
        "output_file": str(output_json_path),
        "details": result.get("details", {}),
        "data": None,
    }


def invoke_subagent(
    prompt: str,
    model: Optional[str] = None,
    context_files: Optional[List[str]] = None,
    allowed_tools: Optional[List[str]] = None,
    timeout: int = 1200,
    max_retries: int = 2,
    retry_backoff: Optional[List[int]] = None,
    log_file: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Invoke a cursor-agent subagent with the given prompt.

    This is a backward-compatible wrapper that delegates to invoke_with_provider()
    while maintaining the original interface and retry logic.

    Args:
        prompt: The prompt/task for the subagent
        model: Optional model to use (e.g., "sonnet", "opus")
        context_files: List of file paths to provide as context (included in prompt)
        allowed_tools: List of allowed tools for the subagent (included in prompt)
        timeout: Timeout in seconds (default 1200 = 20 minutes)
        max_retries: Maximum number of retry attempts
        retry_backoff: List of backoff delays in seconds (default [5, 15])
        log_file: Optional path to save full output log for debugging

    Returns:
        Dict with 'success', 'output', 'stderr', and optionally 'error' keys
    """
    if retry_backoff is None:
        retry_backoff = [5, 15]

    # Build model spec for provider abstraction
    # cursor-agent is the default provider when model doesn't have a prefix
    model_spec = f"cursor-agent:{model}" if model else "cursor-agent:auto"

    last_result = None
    last_error = None

    for attempt in range(max_retries + 1):
        # Delegate to invoke_with_provider()
        result = invoke_with_provider(
            prompt=prompt,
            model_spec=model_spec,
            timeout=timeout,
            log_file=log_file if attempt == max_retries else None  # Only log final attempt
        )

        if result["success"]:
            # Convert to legacy format for backward compatibility
            # The invoke_with_provider returns 'data', but invoke_subagent returns 'output'
            details = result.get("details", {})
            # Return raw JSON output string to match legacy behavior
            data = result.get("data", "")
            output = json.dumps(data) if data and not isinstance(data, str) else str(data) if data else ""
            return {
                "success": True,
                "output": output,
                "stderr": details.get("stderr", "")
            }

        last_result = result
        last_error = result.get("error", "Unknown error")

        # Check if it's a binary not found error (no point retrying)
        if result.get("error_code") == ERROR_BINARY_NOT_FOUND:
            raise LLMClientError("cursor-agent CLI not found. Please ensure it is installed and in PATH.")

        # Retry if we haven't exhausted attempts
        if attempt < max_retries:
            backoff_delay = retry_backoff[min(attempt, len(retry_backoff) - 1)]
            time.sleep(backoff_delay)

    # Check for timeout error
    if last_result and last_result.get("error_code") == ERROR_TIMEOUT:
        raise SubagentTimeoutError(f"Subagent timed out after {max_retries + 1} attempts: {last_error}")

    # Return error in legacy format
    details = last_result.get("details", {}) if last_result else {}
    return {
        "success": False,
        "output": "",
        "stderr": details.get("stderr", ""),
        "error": last_error
    }


def parse_subagent_response(output: str) -> Dict[str, Any]:
    """
    Parse structured output from a subagent response.

    Args:
        output: Raw output string from subagent

    Returns:
        Parsed JSON as dict/list, or {"raw": output} if no JSON found
    """
    output = output.strip()

    # First, try to parse as cursor-agent JSON wrapper format
    # cursor-agent --output-format json returns: {"type":"result","result":"...","..."}
    if output.startswith('{'):
        try:
            wrapper = json.loads(output)
            if isinstance(wrapper, dict) and "result" in wrapper:
                # Extract the inner result and parse it
                inner_result = wrapper["result"]
                if isinstance(inner_result, str):
                    inner_result = inner_result.strip()
                    if inner_result.startswith(('[', '{')):
                        try:
                            return json.loads(inner_result)
                        except json.JSONDecodeError:
                            pass
                    # Try to find JSON in the inner result
                    return _extract_json_from_text(inner_result)
                return inner_result if inner_result else {"raw": output}
            # Not a wrapper format, might be direct JSON
            return wrapper
        except json.JSONDecodeError:
            pass

    # Try direct JSON parsing for arrays
    if output.startswith('['):
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            pass

    return _extract_json_from_text(output)


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON from text that may contain other content."""
    # Try code block extraction
    code_block_match = re.search(r'```(?:json)?\s*([\[\{][\s\S]*?[\]\}])\s*```', text)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON arrays or objects
    for pattern in [r'\[[\s\S]*\]', r'\{[\s\S]*\}']:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    return {"raw": text}

def invoke_for_json(
    prompt: str,
    model: Optional[str] = None,
    context_files: Optional[List[str]] = None,
    timeout: int = 1200
) -> Dict[str, Any]:
    """
    Invoke subagent expecting JSON output.

    Args:
        prompt: The prompt (should request JSON output)
        model: Optional model to use
        context_files: Optional list of context files
        timeout: Timeout in seconds

    Returns:
        Parsed JSON response or error dict
    """
    result = invoke_subagent(
        prompt=prompt,
        model=model,
        context_files=context_files,
        timeout=timeout
    )

    if not result["success"]:
        return result

    parsed = parse_subagent_response(result["output"])
    return {
        "success": True,
        "data": parsed,
        "raw_output": result["output"]
    }
