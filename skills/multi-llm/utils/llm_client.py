"""LLM client wrapper for multi-provider LLM subprocess calls."""

import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .provider_registry import parse_model_spec, get_provider, get_provider_timeout


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


def check_cursor_agent_available() -> bool:
    """Check if cursor-agent CLI is available."""
    return shutil.which("cursor-agent") is not None


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


def _save_log(
    log_file: Union[str, Path],
    model: str,
    prompt: str,
    stdout: str,
    stderr: str,
    returncode: int,
    success: bool,
    error: Optional[str] = None,
    duration_seconds: Optional[float] = None
) -> bool:
    """
    Save agent output to a log file for debugging.

    Args:
        log_file: Path to save the log
        model: Model identifier
        prompt: The prompt sent to the agent
        stdout: Standard output from the process
        stderr: Standard error from the process
        returncode: Process return code
        success: Whether the invocation was successful
        error: Error message if failed
        duration_seconds: How long the agent ran

    Returns:
        True if log was saved successfully, False otherwise
    """
    try:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "=" * 80,
            "CURSOR-AGENT LOG",
            "=" * 80,
            f"Timestamp: {datetime.now().isoformat()}",
            f"Model: {model or 'default'}",
            f"Success: {success}",
            f"Return Code: {returncode}",
        ]

        if duration_seconds is not None:
            lines.append(f"Duration: {duration_seconds:.1f}s")

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

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return True
    except (IOError, OSError) as e:
        print(f"WARNING: Failed to save log to {log_file}: {e}")
        return False


def invoke_with_provider(
    prompt: str,
    model_spec: str,
    timeout: Optional[int] = None,
    log_file: Optional[Union[str, Path]] = None,
    cwd: Optional[str] = None
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
    provider_name, model = parse_model_spec(model_spec)
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

    if not provider.is_available():
        return {
            "success": False,
            "error": f"{provider_name} CLI not found in PATH",
            "error_code": ERROR_BINARY_NOT_FOUND,
            "details": {
                "provider": provider_name,
                "model": model,
            }
        }

    # Use provided timeout or provider's default
    effective_timeout = timeout or get_provider_timeout(provider_name)

    # Build command using provider, resolving cmd[0] to an absolute path
    # (and past any Windows .cmd/.bat npm shim, to a sibling .exe or the
    # shim's `node <cli.js>` target — see _resolve_executable).
    cmd = provider.build_command(prompt, model)
    launcher, is_batch_shim = _resolve_executable(cmd[0])
    cmd = [*launcher, *cmd[1:]]

    if _IS_WINDOWS and provider.prompt_transport == "argv":
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

    # Build subprocess environment: add provider vars, remove blacklisted ones
    provider_env = provider.get_env(model)
    remove_env = provider.get_remove_env()
    if provider_env or remove_env:
        env = {**os.environ, **provider_env}
        for key in remove_env:
            env.pop(key, None)
    else:
        env = None

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
            # No provider reads stdin (prompts are passed via argv); closing it
            # prevents CLIs that poll stdin for piped input (e.g. agy) from
            # blocking forever when the orchestrator's stdin is a non-EOF pipe.
            stdin=subprocess.DEVNULL
        )
        duration = time.time() - start_time

        if result.returncode != 0:
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
                    error=f"{provider_name} exited with code {result.returncode}",
                    duration_seconds=duration
                )

            return {
                "success": False,
                "error": f"{provider_name} exited with code {result.returncode}",
                "error_code": ERROR_SUBPROCESS_FAILED,
                "details": {
                    "exit_code": result.returncode,
                    "stderr": result.stderr,
                    "provider": provider_name,
                    "model": model,
                    "duration_seconds": duration
                }
            }

        # Parse output using provider
        parsed = provider.parse_output(result.stdout, result.stderr)

        # Add details to successful results
        parsed["details"] = {
            "provider": provider_name,
            "model": model,
            "duration_seconds": duration
        }

        # If parse_output indicates failure, add error_code
        if not parsed.get("success", False) and "error_code" not in parsed:
            parsed["error_code"] = ERROR_PARSE_ERROR

        # Save log if requested
        if log_file:
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

        return {
            "success": False,
            "error": f"{provider_name} timed out after {effective_timeout}s",
            "error_code": ERROR_TIMEOUT,
            "details": {
                "provider": provider_name,
                "model": model,
                "timeout": effective_timeout,
                "duration_seconds": duration
            }
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


def invoke_with_file_output(
    prompt_template: str,
    model_spec: str,
    prompt_context: Dict[str, Any],
    output_dir: Union[str, Path],
    phase: str,
    timeout: Optional[int] = None,
    log_file: Optional[Union[str, Path]] = None,
    prefer_arrays: bool = True,
    cwd: Optional[str] = None
) -> Dict[str, Any]:
    """Invoke LLM with file-based JSON output.

    This function instructs the LLM to write JSON output to a specified file
    instead of relying on stdout parsing. The prompt is formatted with an
    {output_json_path} variable that the LLM should use.

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

    Returns:
        Dict with keys:
        - 'success': bool indicating overall success
        - 'data': Parsed JSON data (if successful)
        - 'source': Where JSON came from ('file', 'file_extracted', 'stdout_fallback')
        - 'output_file': Path to the output file (if generated)
        - 'error': Error message (if failed)
        - 'file_error': File-specific error (if file read failed)
        - 'details': Provider execution details
    """
    from .json_extractor import generate_output_path, read_json_from_file

    output_dir = Path(output_dir)

    # Generate output file path
    prefix = prompt_context.get("prefix", "output")
    output_json_path = generate_output_path(output_dir, prefix, phase, model_spec)

    # Inject output path into prompt context
    full_context = {**prompt_context, "output_json_path": str(output_json_path)}

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

    # Invoke provider
    result = invoke_with_provider(prompt, model_spec, timeout, log_file, cwd=cwd)

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
        return {
            **result,
            "output_file": str(output_json_path),
        }

    # Try to read JSON from file first (even if stdout parsing failed)
    file_result = read_json_from_file(output_json_path, prefer_arrays=prefer_arrays)

    if file_result.get("success"):
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
        return {
            "success": True,
            "data": stdout_data,
            "source": "stdout_fallback",
            "output_file": str(output_json_path),
            "file_error": file_result.get("error"),
            "details": result.get("details", {}),
        }

    # Both failed
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
