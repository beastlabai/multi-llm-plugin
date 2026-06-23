"""LLM client wrapper for multi-provider LLM subprocess calls."""

import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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


def check_cursor_agent_available() -> bool:
    """Check if cursor-agent CLI is available."""
    return shutil.which("cursor-agent") is not None


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
        SUBPROCESS_FAILED: Command exited with non-zero code
        PARSE_ERROR: Failed to parse output
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

    # Build command using provider
    cmd = provider.build_command(prompt, model)
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
            timeout=effective_timeout,
            env=env,
            cwd=cwd
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
        # Handle both str (text=True) and bytes (text=False) modes
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
    is_hard_failure = error_code in (ERROR_BINARY_NOT_FOUND, ERROR_TIMEOUT, ERROR_SUBPROCESS_FAILED)

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
