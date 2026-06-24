"""Shared review-orchestration logic for review-plan and review-tasks phases.

This module extracts the common patterns used by review_plan_orchestrator.py
and review_tasks_orchestrator.py into reusable functions, eliminating ~1200+
lines of duplicated logic. Phase-specific orchestrators configure parameters
(prompt template, phase name, output types, consolidation toggle) and delegate
to these shared functions.

Functions cover:
- Model invocation (run_single_model, run_all_models)
- Result saving and validation (save_model_result)
- Status file management (_write_status, _update_status)
- Aggregation (aggregate_results, reaggregate_from_existing_files)
- Common helpers (derive_prefix, get_backoff_delay, extract_json_array, etc.)
"""

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .output_handler import sanitize_prefix, get_phase_dir
from .prompt_loader import load_prompt
from .interactive import resolve_models
from .provider_registry import (
    get_all_model_specs,
    parse_model_spec,
    get_provider_timeout,
    get_provider_max_concurrent,
    is_model_valid,
)
from .json_extractor import generate_output_path, read_json_from_file, sanitize_model_name, build_unsanitize_map
from .llm_client import invoke_with_provider, invoke_with_file_output
from .git_utils import get_project_root
from .html_report_generator import (
    generate_html_report,
    write_html_report,
    sort_raw_groups_by_priority,
    compute_max_importance,
    IMPORTANCE_ORDER,
)
from .state_manager import (
    get_or_create_state,
    stamp_stable_ids,
    load_groups_payload,
    save_groups_payload,
    CURRENT_FORMAT_VERSION,
)
from .validation import (
    prepare_batched_validation_tasks,
    merge_batched_validation_results,
)
from . import (
    group_similar_suggestions,
    export_groups_to_json,
    validate_groups,
    apply_validation_to_groups,
    save_validation_results,
)

# Required fields for each suggestion (shared across review phases)
REQUIRED_FIELDS = ["title", "desc", "importance", "reference", "type"]

# Valid values for certain fields
VALID_IMPORTANCE = {"high", "medium", "low"}
VALID_TYPES = {"addition", "modification", "deletion", "clarification"}

# Delay between launching provider processes to avoid concurrent launch bugs
PROVIDER_STAGGER_DELAY = 2.0  # seconds

# Maximum prompt length to log
MAX_LOGGED_PROMPT_LENGTH = 5000


# ---------------------------------------------------------------------------
# Status file helpers
# ---------------------------------------------------------------------------

def write_status(phase_dir: str, data: dict) -> None:
    """Write a .status.json checkpoint file atomically. Best-effort."""
    temp_path = None
    status_path = os.path.join(phase_dir, ".status.json")
    try:
        data["updated_at"] = datetime.now().isoformat()
        temp_fd, temp_path = tempfile.mkstemp(
            dir=phase_dir, suffix='.tmp', prefix='.status_'
        )
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, status_path)
    except Exception:
        try:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception:
            pass


def update_status(phase_dir: str, updates: dict) -> None:
    """Merge updates into existing .status.json. Best-effort."""
    status_path = os.path.join(phase_dir, ".status.json")
    try:
        existing = {}
        if os.path.exists(status_path):
            with open(status_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        if "output_files" in updates and "output_files" in existing:
            existing["output_files"].update(updates.pop("output_files"))
        existing.update(updates)
        write_status(phase_dir, existing)
    except Exception:
        try:
            write_status(phase_dir, updates)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def derive_prefix(plan_file: str) -> str:
    """Derive output prefix from plan file path."""
    basename = os.path.basename(plan_file)
    return sanitize_prefix(basename)


def read_plan(plan_path: str) -> str:
    """Read and return the plan file content."""
    with open(plan_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_backoff_delay(base_delay: float) -> float:
    """Get backoff delay, respecting test mode for fast retries.

    When MULTI_LLM_TEST_FAST_BACKOFF=1 is set, returns minimal delay (10ms).
    """
    if os.environ.get("MULTI_LLM_TEST_FAST_BACKOFF") == "1":
        return 0.01
    return base_delay


def validate_suggestion(suggestion: dict) -> Tuple[bool, Optional[str]]:
    """Validate that a suggestion has all required fields with valid values."""
    for field in REQUIRED_FIELDS:
        if field not in suggestion:
            return (False, f"Missing required field: {field}")

    importance = suggestion.get("importance", "").lower()
    if importance not in VALID_IMPORTANCE:
        return (False, f"Invalid importance: {importance}")

    stype = suggestion.get("type", "").lower()
    if stype not in VALID_TYPES:
        return (False, f"Invalid type: {stype}")

    return (True, None)


def extract_json_array(text: str) -> Optional[str]:
    """Extract JSON array from text that may contain extra content."""
    try:
        wrapper = json.loads(text)
        if isinstance(wrapper, dict) and 'result' in wrapper:
            text = wrapper['result']
    except (json.JSONDecodeError, TypeError):
        pass

    code_block_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', text)
    if code_block_match:
        return code_block_match.group(1)

    array_match = re.search(r'\[[\s\S]*\]', text)
    if array_match:
        return array_match.group(0)

    return None


def _get_tail_output(text: str, max_lines: int = 20) -> str:
    """Get the last N lines of text for error context."""
    if not text:
        return ""
    lines = text.strip().split('\n')
    if len(lines) <= max_lines:
        return text.strip()
    return '\n'.join(lines[-max_lines:])


def _format_failure_details(
    returncode: int,
    stdout_text: str,
    stderr_text: str,
    max_tail_lines: int = 20
) -> str:
    """Format failure details with clear reason and tail output."""
    error_parts = [f"Exit code {returncode}"]
    combined_output = f"{stderr_text}\n{stdout_text}".lower()

    if 'command not found' in combined_output or 'not found' in stderr_text.lower():
        error_parts.append("Reason: cursor-agent command not found (is it installed and in PATH?)")
    elif 'permission denied' in combined_output:
        error_parts.append("Reason: Permission denied")
    elif 'unauthorized' in combined_output or 'authentication' in combined_output or '401' in stderr_text:
        error_parts.append("Reason: Authentication error (check API key)")
    elif 'model not found' in combined_output or 'invalid model' in combined_output:
        error_parts.append("Reason: Model not found or invalid")
    elif '429' in stderr_text or 'rate limit' in combined_output:
        error_parts.append("Reason: Rate limited")
    elif 'quota' in combined_output or 'insufficient' in combined_output:
        error_parts.append("Reason: Quota exceeded or insufficient credits")
    elif 'api error' in combined_output or 'server error' in combined_output or '500' in stderr_text:
        error_parts.append("Reason: API/Server error")
    else:
        error_parts.append("Reason: Unknown (see tail output below)")

    tail_parts = []
    if stderr_text.strip():
        stderr_tail = _get_tail_output(stderr_text, max_tail_lines)
        tail_parts.append(f"--- stderr (last {max_tail_lines} lines) ---\n{stderr_tail}")
    if stdout_text.strip():
        stdout_tail = _get_tail_output(stdout_text, max_tail_lines)
        tail_parts.append(f"--- stdout (last {max_tail_lines} lines) ---\n{stdout_tail}")
    if tail_parts:
        error_parts.append("\n" + "\n".join(tail_parts))
    elif not stderr_text.strip() and not stdout_text.strip():
        error_parts.append("(no output captured)")

    return "\n".join(error_parts)


def _try_pretty_print_json(text: str) -> str:
    """Try to pretty-print text as JSON, return original if not valid JSON."""
    if not text or not text.strip():
        return text

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and 'result' in parsed:
            inner = parsed['result']
            if isinstance(inner, str):
                try:
                    inner_parsed = json.loads(inner)
                    pretty_inner = json.dumps(inner_parsed, indent=2, ensure_ascii=False)
                    return (
                        f"[cursor-agent wrapper]\n"
                        f"type: {parsed.get('type', 'unknown')}\n"
                        f"result (pretty-printed):\n{pretty_inner}"
                    )
                except (json.JSONDecodeError, TypeError):
                    pass
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return text


# ---------------------------------------------------------------------------
# Model invocation
# ---------------------------------------------------------------------------

async def run_single_model(
    model_spec: str,
    plan_path: str,
    prompt_template: str,
    prompt_context: Dict[str, Any],
    phase_name: str,
    invoke_phase: str,
    timeout: Optional[float] = None,
    retry_count: int = 0,
    out_dir: Optional[str] = None,
    prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single model review with file-based JSON output.

    Args:
        model_spec: Model specification (e.g., 'cursor-agent:auto')
        plan_path: Absolute path to the plan file
        prompt_template: The prompt template string
        prompt_context: Dict of variables to substitute in the prompt
        phase_name: Phase directory name (e.g., 'review-plan', 'review-tasks')
        invoke_phase: Phase name for invoke_with_file_output (e.g., 'plan_review', 'task_review')
        timeout: Optional timeout override in seconds
        retry_count: Current retry attempt (for exponential backoff)
        out_dir: Optional output directory for log files and JSON output
        prefix: Optional prefix for log file names

    Returns:
        Dict with keys: success, output, error, prompt, stdout, stderr, duration_seconds, model_spec, source
    """
    import time
    start_time = time.time()

    project_root = get_project_root(plan_path)
    provider_name, model_name = parse_model_spec(model_spec)
    display_name = f"{provider_name}:{model_name}"

    effective_prefix = prefix or sanitize_prefix(os.path.basename(plan_path))
    full_prompt_context = {**prompt_context}
    if "prefix" not in full_prompt_context:
        full_prompt_context["prefix"] = effective_prefix

    effective_timeout = timeout if timeout is not None else get_provider_timeout(provider_name)

    log_file = None
    if out_dir and prefix:
        sanitized_model = sanitize_model_name(model_spec)
        phase_dir = get_phase_dir(Path(plan_path), phase_name)
        os.makedirs(phase_dir, exist_ok=True)
        log_file = os.path.join(str(phase_dir), f"log_{sanitized_model}.txt")

    # Fall back to old method if no out_dir provided
    if not out_dir:
        prompt = prompt_template.format(**full_prompt_context, output_json_path="(stdout)")
        try:
            print(f"[{display_name}] Starting review (timeout: {effective_timeout}s)...")
            result = await asyncio.to_thread(
                invoke_with_provider,
                prompt=prompt,
                model_spec=model_spec,
                timeout=int(effective_timeout),
                log_file=log_file,
                cwd=project_root
            )
            duration = time.time() - start_time
            details = result.get("details", {})
            stderr_text = details.get("stderr", "")

            if result["success"]:
                output_data = result.get("data", "")
                if output_data and not isinstance(output_data, str):
                    output_str = json.dumps(output_data)
                else:
                    output_str = str(output_data) if output_data else ""
                return {
                    "success": True,
                    "output": output_str,
                    "error": None,
                    "prompt": prompt,
                    "stdout": output_str,
                    "stderr": stderr_text,
                    "duration_seconds": duration,
                    "model_spec": model_spec,
                    "source": "stdout"
                }
            else:
                return {
                    "success": False,
                    "output": None,
                    "error": result.get("error", "Unknown error"),
                    "prompt": prompt,
                    "stdout": "",
                    "stderr": stderr_text,
                    "duration_seconds": duration,
                    "model_spec": model_spec,
                    "source": "stdout"
                }
        except Exception as e:
            duration = time.time() - start_time
            return {
                "success": False,
                "output": None,
                "error": f"Exception: {type(e).__name__}: {e}",
                "prompt": prompt,
                "stdout": "",
                "stderr": "",
                "duration_seconds": duration,
                "model_spec": model_spec,
                "source": "stdout"
            }

    try:
        print(f"[{display_name}] Starting review with file-based output (timeout: {effective_timeout}s)...")

        result = await asyncio.to_thread(
            invoke_with_file_output,
            prompt_template=prompt_template,
            model_spec=model_spec,
            prompt_context=full_prompt_context,
            output_dir=out_dir,
            phase=invoke_phase,
            timeout=int(effective_timeout),
            log_file=log_file,
            prefer_arrays=True,
            cwd=project_root
        )

        duration = time.time() - start_time
        details = result.get("details", {})
        stderr_text = details.get("stderr", "")
        source = result.get("source", "unknown")
        output_file = result.get("output_file", "")

        prompt = prompt_template.format(**full_prompt_context, output_json_path=output_file)

        if output_file:
            print(f"[{display_name}] Output file: {output_file} (source: {source})")

        # Check for rate limiting (429)
        if not result.get("success") and '429' in str(result.get("error", "")):
            if retry_count < 3:
                base_backoff = [5, 10, 20][retry_count]
                backoff = get_backoff_delay(base_backoff)
                print(f"[{display_name}] Rate limited (429), retrying in {backoff}s...")
                await asyncio.sleep(backoff)
                return await run_single_model(
                    model_spec, plan_path, prompt_template, prompt_context,
                    phase_name, invoke_phase, timeout, retry_count + 1, out_dir, prefix
                )
            else:
                return {
                    "success": False, "output": None,
                    "error": "Rate limited (429) - max retries exceeded",
                    "prompt": prompt, "stdout": "", "stderr": stderr_text,
                    "duration_seconds": duration, "model_spec": model_spec,
                    "source": source
                }

        # Check for context exceeded
        if not result.get("success") and 'context' in str(result.get("error", "")).lower():
            return {
                "success": False, "output": None,
                "error": "Context length exceeded - consider chunking the plan",
                "prompt": prompt, "stdout": "", "stderr": stderr_text,
                "duration_seconds": duration, "model_spec": model_spec,
                "source": source
            }

        # Check for network/connection errors with retry
        error_msg = str(result.get("error", "")).lower()
        if not result.get("success") and ('connection' in error_msg or 'network' in error_msg):
            if retry_count < 1:
                backoff = get_backoff_delay(5)
                print(f"[{display_name}] Network error, retrying in {backoff}s...")
                await asyncio.sleep(backoff)
                return await run_single_model(
                    model_spec, plan_path, prompt_template, prompt_context,
                    phase_name, invoke_phase, timeout, retry_count + 1, out_dir, prefix
                )

        if result.get("success"):
            print(f"[{display_name}] Completed successfully in {duration:.1f}s (source: {source})")
            output_data = result.get("data", "")
            if output_data and not isinstance(output_data, str):
                output_str = json.dumps(output_data)
            else:
                output_str = str(output_data) if output_data else ""

            return {
                "success": True, "output": output_str,
                "error": None, "prompt": prompt,
                "stdout": output_str, "stderr": stderr_text,
                "duration_seconds": duration, "model_spec": model_spec,
                "source": source
            }
        else:
            error_details = result.get("error", "Unknown error")
            file_error = result.get("file_error", "")
            if file_error:
                print(f"[{display_name}] File error: {file_error}")
            print(f"[{display_name}] Failed: {error_details}")
            return {
                "success": False, "output": None,
                "error": error_details, "prompt": prompt,
                "stdout": "", "stderr": stderr_text,
                "duration_seconds": duration, "model_spec": model_spec,
                "source": source
            }

    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Exception: {type(e).__name__}: {e}"
        print(f"[{display_name}] {error_msg}")
        try:
            prompt_for_log = prompt
        except NameError:
            prompt_for_log = prompt_template if 'prompt_template' in dir() else "(unavailable)"
        return {
            "success": False, "output": None,
            "error": error_msg, "prompt": prompt_for_log,
            "stdout": "", "stderr": "",
            "duration_seconds": duration, "model_spec": model_spec,
            "source": "exception"
        }


async def run_models_concurrent(
    model_specs: List[str],
    per_model_coro_factory: Callable[[str, int], Any],
    timeout: Optional[float],
    max_parallel: int,
    skip_predicate: Optional[Callable[[str, int], Optional[Any]]] = None,
) -> Dict[str, Any]:
    """Run a per-model async callback across all models with shared concurrency control.

    This encapsulates the concurrency *mechanics* shared by every multi-model
    phase (review and ask), decoupled from JSON/grouping/validation concerns:

    - a global ``asyncio.Semaphore(max_parallel)``
    - per-provider semaphores derived from ``get_provider_max_concurrent`` (a
      provider's semaphore is acquired *before* the global one so capped
      providers do not hog global slots while waiting for their turn)
    - staggered launches (``index * PROVIDER_STAGGER_DELAY``, honoring
      ``MULTI_LLM_TEST_FAST_BACKOFF``)
    - an overall timeout wrapping ``asyncio.gather`` that cancels stragglers

    The caller supplies ``per_model_coro_factory(model_spec, index)`` returning
    an awaitable; whatever it produces is stored in the returned dict keyed by
    ``model_spec``. Callbacks that raise are logged and omitted from the
    results (mirroring the original gather/return_exceptions behavior).

    Args:
        model_specs: List of model specifications in provider:model format.
        per_model_coro_factory: ``(model_spec, index) -> awaitable``. The
            awaitable's result is what ends up in the results dict.
        timeout: Optional per-model timeout override (seconds); also drives the
            overall-timeout calculation.
        max_parallel: Maximum concurrent model invocations.
        skip_predicate: Optional cheap, synchronous ``(model_spec, index) ->
            Optional[result]`` checked *before* the stagger sleep and any
            semaphore acquisition. If it returns a non-``None`` value, that
            value is stored as the model's result and the model short-circuits
            immediately (no stagger delay, no semaphore serialization). This
            keeps resume of already-completed models fast. Returning ``None``
            means "not skipped" and the model proceeds normally.

    Returns:
        Dict mapping model spec to whatever the callback returned for it.
    """
    results: Dict[str, Any] = {}
    if not model_specs:
        return results

    semaphore = asyncio.Semaphore(max_parallel)

    # Per-provider semaphores for concurrency limiting
    provider_semaphores: Dict[str, asyncio.Semaphore] = {}
    for spec in model_specs:
        prov, _ = parse_model_spec(spec)
        if prov not in provider_semaphores:
            limit = get_provider_max_concurrent(prov)
            if limit is not None:
                provider_semaphores[prov] = asyncio.Semaphore(limit)

    async def run_with_semaphore(model_spec: str, index: int) -> Tuple[str, Any]:
        provider_name, model_name = parse_model_spec(model_spec)
        display_name = f"{provider_name}:{model_name}"

        # Cheap pre-stagger skip check: short-circuit already-completed models on
        # resume *before* paying the stagger delay or serializing through the
        # semaphores (otherwise a resume of N done models adds ~(N-1) stagger
        # waits + semaphore queueing it never incurred pre-refactor).
        if skip_predicate is not None:
            skipped = skip_predicate(model_spec, index)
            if skipped is not None:
                return (model_spec, skipped)

        # Stagger launches
        if index > 0:
            base_stagger = index * PROVIDER_STAGGER_DELAY
            stagger_delay = get_backoff_delay(base_stagger)
            print(f"[{display_name}] Waiting {stagger_delay:.1f}s before starting (staggered launch)...")
            await asyncio.sleep(stagger_delay)

        # Acquire provider semaphore first (if any), then global semaphore.
        prov_sem = provider_semaphores.get(provider_name)
        if prov_sem:
            async with prov_sem:
                async with semaphore:
                    return (model_spec, await per_model_coro_factory(model_spec, index))
        else:
            async with semaphore:
                return (model_spec, await per_model_coro_factory(model_spec, index))

    tasks = [asyncio.create_task(run_with_semaphore(spec, idx)) for idx, spec in enumerate(model_specs)]

    # Calculate total timeout
    if timeout is not None:
        base_timeout = timeout
    else:
        base_timeout = max(get_provider_timeout(parse_model_spec(spec)[0]) for spec in model_specs)

    stagger_overhead = len(model_specs) * PROVIDER_STAGGER_DELAY
    total_timeout = base_timeout * 2 + stagger_overhead

    try:
        completed = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=total_timeout
        )

        for item in completed:
            if isinstance(item, Exception):
                print(f"Task exception: {item}")
                continue
            model_spec, result = item
            results[model_spec] = result

    except asyncio.TimeoutError:
        print(f"Total timeout ({total_timeout}s) exceeded for all models")
        for task in tasks:
            if not task.done():
                task.cancel()

    return results


async def run_all_models(
    model_specs: List[str],
    plan_path: str,
    prompt_template: str,
    prompt_context_factory: Callable[[str], Dict[str, Any]],
    phase_name: str,
    invoke_phase: str,
    timeout: Optional[float],
    max_parallel: int,
    out_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    skip_existing: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Run all models with concurrency limit and staggered starts.

    The concurrency mechanics (semaphores, staggered starts, overall timeout)
    are delegated to :func:`run_models_concurrent`; this function supplies the
    JSON/grouping/validation-specific per-model behavior (resume skip of
    existing ``{model}.json`` files + ``run_single_model`` invocation) via the
    callback it hands to the helper.

    Args:
        model_specs: List of model specifications in provider:model format
        plan_path: Absolute path to the plan file
        prompt_template: The prompt template string
        prompt_context_factory: Callable that takes model_spec and returns prompt_context dict
            (allows per-model context variations if needed; typically returns same dict)
        phase_name: Phase directory name (e.g., 'review-plan', 'review-tasks')
        invoke_phase: Phase name for invoke_with_file_output (e.g., 'plan_review', 'task_review')
        timeout: Optional timeout override per model in seconds
        max_parallel: Maximum concurrent model invocations
        out_dir: Optional output directory for log files
        prefix: Optional prefix for log file names
        skip_existing: If True, skip models that already have result files

    Returns:
        Dict mapping model spec to result dict
    """
    def skip_existing_result(model_spec: str, index: int) -> Optional[Dict[str, Any]]:
        # Cheap, synchronous resume check (runs before stagger/semaphores):
        # skip models that already have valid results (resume after partial
        # failure). Returns None when the model must (re)run.
        if not (skip_existing and out_dir):
            return None
        provider_name, model_name = parse_model_spec(model_spec)
        display_name = f"{provider_name}:{model_name}"
        sanitized = sanitize_model_name(model_spec)
        existing_path = os.path.join(
            str(get_phase_dir(Path(plan_path), phase_name)),
            f"{sanitized}.json"
        )
        if os.path.exists(existing_path) and os.path.getsize(existing_path) > 0:
            try:
                with open(existing_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if isinstance(existing_data, list):
                    print(f"[SKIP] {display_name} - already has results ({len(existing_data)} items)")
                    return {
                        "success": True,
                        "output": json.dumps(existing_data),
                        "error": None,
                        "prompt": "(skipped - existing results)",
                        "stdout": "", "stderr": "",
                        "duration_seconds": 0.0,
                        "model_spec": model_spec,
                        "source": "existing_file"
                    }
            except (json.JSONDecodeError, IOError):
                print(f"[{display_name}] Existing result file corrupt, re-running...")
        return None

    async def per_model(model_spec: str, index: int) -> Dict[str, Any]:
        prompt_context = prompt_context_factory(model_spec)
        return await run_single_model(
            model_spec, plan_path, prompt_template, prompt_context,
            phase_name, invoke_phase, timeout,
            out_dir=out_dir, prefix=prefix,
        )

    return await run_models_concurrent(
        model_specs, per_model, timeout, max_parallel,
        skip_predicate=skip_existing_result,
    )


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_model_result(
    prefix: str,
    model: str,
    success: bool,
    output: Optional[str],
    error: Optional[str],
    out_dir: str,
    phase_dir: str,
    phase_name: str,
) -> bool:
    """Save model result to appropriate file.

    Args:
        prefix: Output file prefix
        model: Model name
        success: Whether the model run succeeded
        output: Model output (JSON string)
        error: Error message if failed
        out_dir: Output directory (base folder)
        phase_dir: Phase-specific directory for output files
        phase_name: Phase name for salvage metadata (e.g., 'review_plan', 'review_tasks')

    Returns:
        True if result was saved successfully, False otherwise
    """
    sanitized_model = sanitize_model_name(model)

    if not success or not output:
        error_path = os.path.join(phase_dir, f"error_{sanitized_model}.log")
        with open(error_path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {model}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Error: {error or 'Unknown error'}\n")
            if output:
                f.write(f"\nRaw output:\n{output}\n")
        print(f"[{model}] Error logged to: {error_path}")
        return False

    try:
        json_text = extract_json_array(output)
        if json_text is None:
            raise json.JSONDecodeError("No JSON array found", output, 0)

        suggestions = json.loads(json_text)

        if not isinstance(suggestions, list):
            raise ValueError("Expected JSON array")

        validated_suggestions = []
        for i, suggestion in enumerate(suggestions):
            is_valid, validation_error = validate_suggestion(suggestion)
            if not is_valid:
                print(f"[{model}] Warning: Suggestion {i} invalid - {validation_error}")
                continue
            suggestion["importance"] = suggestion["importance"].upper()
            suggestion["type"] = suggestion["type"].lower()
            validated_suggestions.append(suggestion)

        if len(suggestions) == 0:
            print(f"[{model}] Warning: Empty suggestions array (no issues found)")
        elif len(validated_suggestions) == 0:
            print(f"[{model}] Warning: All suggestions were invalid")

        result_path = os.path.join(phase_dir, f"{sanitized_model}.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(validated_suggestions, f, indent=2)

        print(f"[{model}] Saved {len(validated_suggestions)} suggestions to: {result_path}")
        return True

    except (json.JSONDecodeError, ValueError) as e:
        salvage_request = {
            "model": model,
            "phase": phase_name,
            "raw_output": output,
            "expected_type": "array",
            "expected_schema": {
                "fields": list(REQUIRED_FIELDS),
                "importance_values": list(VALID_IMPORTANCE),
                "type_values": list(VALID_TYPES),
            },
            "output_path": os.path.join(phase_dir, f"{sanitized_model}.json"),
            "timestamp": datetime.now().isoformat()
        }
        salvage_path = os.path.join(phase_dir, f"salvage_{sanitized_model}.json")
        with open(salvage_path, 'w', encoding='utf-8') as f:
            json.dump(salvage_request, f, indent=2)

        print(f"[SALVAGE_NEEDED] {salvage_path}")

        error_path = os.path.join(phase_dir, f"error_{sanitized_model}.log")
        with open(error_path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {model}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Parse error: {e}\n")
            f.write(f"\nRaw output:\n{output}\n")
        print(f"[{model}] JSON parse error, raw output saved to: {error_path}")
        return False


# ---------------------------------------------------------------------------
# Agent log saving
# ---------------------------------------------------------------------------

def save_agent_log(
    prefix: str,
    model: str,
    prompt: str,
    stdout: str,
    stderr: str,
    success: bool,
    out_dir: str,
    phase_dir: str,
    duration_seconds: Optional[float] = None,
    error: Optional[str] = None
) -> Optional[str]:
    """Save full agent output to a log file for debugging."""
    sanitized_model = sanitize_model_name(model)
    log_path = os.path.join(phase_dir, f"log_{sanitized_model}.txt")

    lines = [
        "=" * 80,
        "CURSOR-AGENT LOG (Plan Review)",
        "=" * 80,
        f"Timestamp: {datetime.now().isoformat()}",
        f"Model: {model}",
        f"Success: {success}",
    ]

    if duration_seconds is not None:
        lines.append(f"Duration: {duration_seconds:.1f}s")
    if error:
        lines.append(f"Error: {error}")

    truncated_prompt = prompt[:MAX_LOGGED_PROMPT_LENGTH]
    if len(prompt) > MAX_LOGGED_PROMPT_LENGTH:
        truncated_prompt += "..."

    formatted_stdout = _try_pretty_print_json(stdout) if stdout else "(empty)"

    lines.extend([
        "", "-" * 40, "PROMPT", "-" * 40,
        truncated_prompt,
        "", "-" * 40, "STDOUT", "-" * 40,
        formatted_stdout,
        "", "-" * 40, "STDERR", "-" * 40,
        stderr if stderr else "(empty)",
        "", "=" * 80, "",
    ])

    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return log_path
    except (IOError, OSError) as e:
        print(f"WARNING: Failed to save log to {log_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(
    prefix: str,
    out_dir: str,
    phase_dir: str,
    phase_name: str,
    models: List[str],
    failed_models: Dict[str, str],
    validated_groups: Optional[List[Dict]] = None,
    plan_path: Optional[str] = None,
    base_ref: Optional[str] = None,
    tasks_path: Optional[Path] = None,
    template_style: str = 'pr',
    diff_data: Optional[Dict] = None,
) -> str:
    """Aggregate all model results into a consolidated markdown report.

    Args:
        prefix: Output file prefix
        out_dir: Output directory (base folder)
        phase_dir: Phase-specific directory for output files
        phase_name: Phase type for HTML report (e.g., 'review-plan', 'review-tasks')
        models: List of all models used
        failed_models: Dict of model -> error for failed models
        validated_groups: Optional list of validated suggestion groups
        plan_path: Optional path to the plan file for HTML report
        base_ref: Optional git ref for PR-style diff context (forwarded to
            ``generate_html_report()``).
        tasks_path: Optional path to tasks JSON for task-view metadata
            (forwarded to ``generate_html_report()``).
        template_style: Template to use — ``'pr'`` or ``'flat'``
            (forwarded to ``generate_html_report()``).  Defaults to ``'pr'``.
        diff_data: Optional pre-computed diff hunk data (forwarded to
            ``generate_html_report()``).

    Returns:
        Path to the generated report
    """
    groups_to_display = validated_groups
    if not groups_to_display:
        grouped_path = os.path.join(phase_dir, "grouped.json")
        if os.path.exists(grouped_path):
            try:
                with open(grouped_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                groups_to_display = load_groups_payload(raw_data)
            except (json.JSONDecodeError, IOError, ValueError) as e:
                print(f"Warning: Could not read {grouped_path}: {e}")
                groups_to_display = []

    # Count suggestions by importance
    high_count = medium_count = low_count = total_suggestions = 0
    if groups_to_display:
        for group in groups_to_display:
            for s in group.get("suggestions", []):
                total_suggestions += 1
                importance = s.get("importance", "LOW").upper()
                if importance == "HIGH":
                    high_count += 1
                elif importance == "MEDIUM":
                    medium_count += 1
                else:
                    low_count += 1

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    successful_models = [m for m in models if m not in failed_models]

    phase_labels = {
        "review-plan": "Plan Review",
        "review-tasks": "Task Review",
        "code-review": "Code Review",
    }
    phase_label = phase_labels.get(phase_name, "Plan Review")

    report_lines = [
        f"# {phase_label} Report: {prefix}",
        "",
        f"**Original plan:** {prefix}.md",
        f"**Generated:** {timestamp}",
        f"**Models:** {', '.join(successful_models)}",
        f"**Groups:** {len(groups_to_display) if groups_to_display else 0}",
        f"**Suggestions:** {high_count} HIGH, {medium_count} MEDIUM, {low_count} LOW ({total_suggestions} total)",
    ]

    if groups_to_display:
        valid_count = sum(1 for g in groups_to_display if g.get("validation_status") == "valid")
        invalid_count = sum(1 for g in groups_to_display if g.get("validation_status") == "invalid")
        needs_human = sum(1 for g in groups_to_display if g.get("validation_status") == "needs-human-decision")
        if valid_count or invalid_count or needs_human:
            report_lines.append(f"**Validation:** {valid_count} valid, {invalid_count} invalid, {needs_human} needs human review")

    # NOTE: All phases (including review-tasks) use interactive reports with
    # skip/approve checkboxes.  The original code suppressed these controls for
    # review-tasks (readOnly=True), but the task-review workflow now supports
    # skip/override actions, so interactivity is intentionally enabled for every
    # phase.  See also html_report_generator.py where readOnly is set to False.
    report_lines.append("")
    report_lines.append("> **Note:** An interactive HTML report is also available at `report.html`.")
    report_lines.append("> If you export selections from the HTML report (`user_selections.json`),")
    report_lines.append("> those selections will take precedence over checkboxes in this file.")
    report_lines.append("")

    if groups_to_display:
        groups_to_display = sort_raw_groups_by_priority(groups_to_display)
        for group_idx, group in enumerate(groups_to_display, 1):
            report_lines.extend(format_group(group, group_idx))
    else:
        report_lines.append("_No suggestions found._")
        report_lines.append("")

    if failed_models:
        report_lines.append("## Models Failed")
        report_lines.append("")
        for model, reason in failed_models.items():
            report_lines.append(f"- **{model}**: {reason}")
        report_lines.append("")

    report_path = os.path.join(phase_dir, "report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    # Generate HTML report
    try:
        effective_plan_path = Path(plan_path) if plan_path else Path(out_dir) / f"{prefix}.md"
        html_content = generate_html_report(
            groups=groups_to_display or [],
            plan_path=effective_plan_path,
            phase_dir=Path(phase_dir),
            phase_type=phase_name,
            models=successful_models,
            failed_models=failed_models,
            base_ref=base_ref,
            tasks_path=tasks_path,
            template_style=template_style,
            diff_data=diff_data,
        )
        write_html_report(html_content, Path(phase_dir))
        print(f"HTML report written to: {os.path.join(phase_dir, 'report.html')}")
    except Exception as e:
        print(f"Warning: Failed to generate HTML report: {e}")

    return report_path


# ---------------------------------------------------------------------------
# Report formatting helpers
# ---------------------------------------------------------------------------

def format_group(group: Dict[str, Any], group_idx: int) -> List[str]:
    """Format a suggestion group for the markdown report.

    Args:
        group: Group dict with theme, suggestions, validation info, etc.
        group_idx: 1-based group index for display.
    """
    theme = group.get("theme", "Untitled Group")
    category = group.get("category", "unknown")
    validation_status = group.get("validation_status")
    validation_reason = group.get("validation_reason")
    priority_score = group.get("priority_score", 0)
    models = group.get("models", [])
    suggestions = group.get("suggestions", [])

    validation_display = {
        "valid": "Valid",
        "invalid": "Invalid",
        "needs-human-decision": "? Needs Review",
        "validation_failed": "? Validation Failed",
    }
    validation_str = validation_display.get(validation_status, "? Unknown") if validation_status else "? Unknown"

    highest_importance = compute_max_importance(suggestions) or "LOW"

    display_label = group.get("display_label", f"G{group_idx}")
    display_hash = group.get("display_hash", "")
    if display_hash:
        header = f"## {display_label} [{display_hash}]: {theme}"
    else:
        header = f"## {display_label}: {theme}"

    lines = [
        header,
        "",
    ]

    lines.append(f"- [ ] Skip this group")
    if validation_status in ("needs-human-decision", "validation_failed"):
        lines.append("- [ ] Mark valid")
        lines.append("- [ ] Mark invalid")

    lines.extend([
        f"**Validation:** {validation_str} | "
        f"**Category:** {category} | "
        f"**Priority:** {priority_score} | "
        f"**Highest Importance:** {highest_importance} | "
        f"**Models:** {', '.join(models)}",
        "",
    ])

    if validation_reason and validation_status in ("invalid", "needs-human-decision", "validation_failed"):
        lines.append(f"> **Validation Reason:** {validation_reason}")
        lines.append("")

    sorted_suggestions = sorted(
        suggestions,
        key=lambda s: IMPORTANCE_ORDER.get(s.get("importance", "LOW").upper(), 3)
    )

    for suggestion_idx, suggestion in enumerate(sorted_suggestions, 1):
        lines.extend(format_suggestion_in_group(
            suggestion, group_idx, suggestion_idx,
            group_validation_status=validation_status or "",
            group_suggestion_count=len(suggestions),
        ))

    lines.append("---")
    lines.append("")

    return lines


def format_suggestion_in_group(
    suggestion: dict,
    group_idx: int,
    suggestion_idx: int,
    group_validation_status: str = "",
    group_suggestion_count: int = 1,
) -> List[str]:
    """Format a single suggestion within a group for the markdown report.

    Args:
        suggestion: Suggestion dict.
        group_idx: 1-based group index.
        suggestion_idx: 1-based suggestion index within group.
        group_validation_status: Validation status of the parent group.
        group_suggestion_count: Number of suggestions in the parent group.
    """
    suggestion_id = f"G{group_idx}S{suggestion_idx}"
    title = suggestion.get("title", "Untitled")
    desc = suggestion.get("desc", "")
    stype = suggestion.get("type", "unknown")
    reference = suggestion.get("reference", "")
    importance = suggestion.get("importance", "LOW").upper()
    model = suggestion.get("model", suggestion.get("source_model", "unknown"))

    display_label = suggestion.get("display_label", suggestion_id)
    display_hash = suggestion.get("display_hash", "")
    if display_hash:
        header = f"### {display_label} [{display_hash}]: {title}"
    else:
        header = f"### {display_label}: {title}"

    lines = [
        header,
    ]
    lines.append(f"- [ ] Skip")
    if group_validation_status in ("needs-human-decision", "validation_failed") and group_suggestion_count > 1:
        lines.append("- [ ] Mark valid")
        lines.append("- [ ] Mark invalid")
    lines.extend([
        f"**Importance:** {importance} | "
        f"**Type:** {stype} | "
        f"**Section:** {reference} | "
        f"**Model:** {model}",
        "",
        desc,
        "",
    ])

    return lines


def format_suggestion(
    suggestion: dict,
    validation_status: Optional[str] = None,
    validation_reason: Optional[str] = None,
) -> List[str]:
    """Format a single suggestion for the markdown report.

    Args:
        suggestion: Suggestion dict with id, title, type, reference, desc, etc.
        validation_status: Optional validation status string.
        validation_reason: Optional validation reason string.
    """
    validation_display = {
        "valid": "Valid",
        "invalid": "Invalid",
        "needs-human-decision": "? Needs Review",
        "validation_failed": "? Validation Failed",
    }
    validation_str = validation_display.get(validation_status, "? Unknown") if validation_status else "? Unknown"

    lines = [
        f"### {suggestion['id']}: {suggestion['title']}",
    ]
    lines.append(f"- [ ] Skip")
    lines.extend([
        f"**Validation:** {validation_str} | "
        f"**Model:** {suggestion.get('model', 'unknown')} | "
        f"**Type:** {suggestion['type']} | "
        f"**Section:** {suggestion['reference']}",
        "",
    ])

    if validation_reason and validation_status in ("invalid", "needs-human-decision", "validation_failed"):
        lines.append(f"> **Validation Reason:** {validation_reason}")
        lines.append("")

    lines.extend([
        suggestion['desc'],
        "",
        "---",
        "",
    ])
    return lines


# ---------------------------------------------------------------------------
# Reaggregation
# ---------------------------------------------------------------------------

async def reaggregate_from_existing_files(
    prefix: str,
    out_dir: str,
    phase_dir: str,
    phase_name: str,
    plan_path: str,
    args: argparse.Namespace,
    orchestrator_script: str,
    extra_validation_metadata: Optional[Dict[str, str]] = None,
    base_ref: Optional[str] = None,
    tasks_path: Optional[Path] = None,
    template_style: str = 'pr',
    diff_data: Optional[Dict] = None,
) -> None:
    """Re-aggregate all model results after salvage operations complete.

    Args:
        prefix: Output file prefix
        out_dir: Output directory (base folder)
        phase_dir: Phase-specific directory containing model results
        phase_name: Phase name (e.g., 'review-plan', 'review-tasks')
        plan_path: Path to the plan file
        args: Command-line arguments for validation settings
        orchestrator_script: Name of the orchestrator script for reaggregate command
        extra_validation_metadata: Optional extra fields for validation_tasks.json
            (e.g., {"tasks_file": "/path/to/tasks.md"})
        base_ref: Optional git ref for PR-style diff context (forwarded to
            ``aggregate_results()``).
        tasks_path: Optional path to tasks JSON for task-view metadata
            (forwarded to ``aggregate_results()``).
        template_style: Template to use — ``'pr'`` or ``'flat'``
            (forwarded to ``aggregate_results()``).  Defaults to ``'pr'``.
        diff_data: Optional pre-computed diff hunk data (forwarded to
            ``aggregate_results()``).
    """
    import glob as glob_module

    pattern = os.path.join(phase_dir, "*.json")
    all_files = glob_module.glob(pattern)

    exclude_patterns = ['grouped', 'validation', 'salvage', 'revalidation', 'consolidated', 'consolidation', 'report_data']
    result_files = [
        f for f in all_files
        if not any(os.path.basename(f).startswith(ex) or ex in os.path.basename(f) for ex in exclude_patterns)
    ]

    if not result_files:
        print(f"No model result files found in: {phase_dir}")
        return

    # Collect all suggestions
    all_suggestions = []
    models_found = set()

    # Restore original model specs (with colons) from .status.json
    unsanitize = build_unsanitize_map(phase_dir)

    for result_path in result_files:
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                suggestions = json.load(f)
                basename = os.path.basename(result_path)
                sanitized = basename.replace(".json", "")
                model = unsanitize.get(sanitized, sanitized)
                models_found.add(model)
                for s in suggestions:
                    s["source_model"] = model
                all_suggestions.extend(suggestions)
                print(f"  Loaded {len(suggestions)} suggestions from {model}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Warning: Could not read {result_path}: {e}")

    print(f"\nRe-aggregating {len(all_suggestions)} suggestions from {len(models_found)} models")

    if not all_suggestions:
        print("No suggestions to aggregate.")
        return

    # Group similar suggestions - preserve existing groups if validation exists
    grouped_path = os.path.join(phase_dir, "grouped.json")
    validation_path_check = os.path.join(phase_dir, "validation.json")
    validation_tasks_path_check = os.path.join(phase_dir, "validation_tasks.json")

    has_existing_validation = (
        (os.path.exists(validation_path_check) and not args.skip_validation) or
        os.path.exists(validation_tasks_path_check)
    )

    if has_existing_validation and os.path.exists(grouped_path):
        print(f"Using existing grouped.json (preserves validation alignment)")
        with open(grouped_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        grouped = load_groups_payload(raw_data)
    else:
        raw_grouped = group_similar_suggestions(all_suggestions)
        grouped = [g.to_dict() if hasattr(g, 'to_dict') else g for g in raw_grouped]
        stamp_stable_ids(grouped)
        with open(grouped_path, 'w', encoding='utf-8') as f:
            json.dump(save_groups_payload(grouped), f, indent=2)
        print(f"Grouped suggestions saved to: {grouped_path}")

    plan_content = read_plan(plan_path)

    validated_groups_for_report: Optional[List[Dict]] = None
    validation_path = os.path.join(phase_dir, "validation.json")
    validation_tasks_path = os.path.join(phase_dir, "validation_tasks.json")

    # Check if validation already completed
    if os.path.exists(validation_path) and not args.skip_validation:
        print(f"\nFound existing validation.json, using it for report generation...")
        try:
            from .validation import load_validation_results as load_validation_v2
            validation_results = load_validation_v2(Path(validation_path))
            validated_groups = apply_validation_to_groups(
                [g.to_dict() if hasattr(g, 'to_dict') else g for g in grouped],
                validation_results
            )
            validated_groups_for_report = validated_groups
            print(f"Loaded {len(validation_results)} validation results")
        except Exception as e:
            print(f"Warning: Could not load validation.json: {e}")
            validated_groups_for_report = None

    # Check if batched validation completed
    elif os.path.exists(validation_tasks_path) and not args.skip_validation:
        print("\nFound validation_tasks.json, checking for completed batches...")
        try:
            with open(validation_tasks_path, 'r', encoding='utf-8') as f:
                batch_metadata = json.load(f)

            total_batches = batch_metadata.get("total_batches", 0)
            batches_found = 0
            for batch in batch_metadata.get("batches", []):
                batch_file = os.path.join(phase_dir, f"validation_batch_{batch['batch_index']}.json")
                if os.path.exists(batch_file):
                    batches_found += 1

            if batches_found == total_batches and total_batches > 0:
                print(f"All {total_batches} batch files found, merging...")
                validation_results = merge_batched_validation_results(
                    output_dir=phase_dir,
                    batch_metadata=batch_metadata,
                    total_groups=len(grouped)
                )

                save_validation_results(validation_results, Path(validation_path))
                print(f"Merged validation saved to: {validation_path}")

                validated_groups = apply_validation_to_groups(
                    [g.to_dict() if hasattr(g, 'to_dict') else g for g in grouped],
                    validation_results
                )
                validated_groups_for_report = validated_groups
            else:
                print(f"Only {batches_found}/{total_batches} batch files found.")
                print("Wait for all validation subagents to complete before reaggregating.")
                return
        except Exception as e:
            print(f"Warning: Could not process batch metadata: {e}")
            validated_groups_for_report = None

    # Run internal validation if requested
    elif not args.skip_validation:
        internal_validation = getattr(args, 'internal_validation', False)
        if internal_validation:
            print("\nRunning internal validation...")
            validation_timeout = args.timeout if args.timeout else 600
            validation_results = await validate_groups(
                [g.to_dict() if hasattr(g, 'to_dict') else g for g in grouped],
                context=plan_content,
                model=args.validation_model,
                timeout=validation_timeout
            )

            validated_groups = apply_validation_to_groups(
                [g.to_dict() if hasattr(g, 'to_dict') else g for g in grouped],
                validation_results
            )
            validated_groups_for_report = validated_groups

            save_validation_results(validation_results, Path(validation_path))
            print(f"Validation results saved to: {validation_path}")
        else:
            print("\nPreparing batched validation for Claude Code subagent...")
            extra_kwargs = {}
            if extra_validation_metadata:
                extra_kwargs = {k: v for k, v in extra_validation_metadata.items()
                                if k in ("tasks_file",)}
            batched_tasks = prepare_batched_validation_tasks(
                groups=[g.to_dict() if hasattr(g, 'to_dict') else g for g in grouped],
                context=plan_content,
                output_dir=phase_dir,
                plan_file=str(plan_path),
                model=args.validation_model,
                orchestrator=orchestrator_script,
                **extra_kwargs,
            )
            batched_tasks["format_version"] = CURRENT_FORMAT_VERSION

            with open(validation_tasks_path, 'w', encoding='utf-8') as f:
                json.dump(batched_tasks, f, indent=2)

            total_batches = batched_tasks["total_batches"]
            stats = batched_tasks.get("batching_stats", {})

            if total_batches == 1:
                print(f"\n[VALIDATION_PENDING] {validation_tasks_path}")
            else:
                print(f"\n[VALIDATION_BATCHES_PENDING] {validation_tasks_path}")
                print(f"Batches: {total_batches}")

            print(f"\nValidation prepared for Claude Code subagent.")
            print(f"Groups to validate: {stats.get('total_groups', len(grouped))}")
            print("\nAfter validation completes, run:")
            print(f"  {batched_tasks.get('reaggregate_command', '')}")
            return
    else:
        print("\nSkipping validation (--skip-validation flag set)")

    # Build list of successful models
    successful_models = list(models_found)
    failed_models_map: Dict[str, str] = {}

    # Generate report
    report_path = aggregate_results(
        prefix, out_dir, phase_dir, phase_name,
        successful_models, failed_models_map, validated_groups_for_report,
        plan_path=plan_path,
        base_ref=base_ref,
        tasks_path=tasks_path,
        template_style=template_style,
        diff_data=diff_data,
    )

    print("\nReaggregation complete!")
    print(f"Report: {report_path}")

    # Mark phase complete
    state = get_or_create_state(Path(plan_path))
    state.mark_phase_completed(phase_name)
    state.save()


# ---------------------------------------------------------------------------
# Main orchestration flow
# ---------------------------------------------------------------------------

async def run_review_orchestration(
    args: argparse.Namespace,
    phase_name: str,
    invoke_phase: str,
    salvage_phase_name: str,
    prompt_template: str,
    prompt_context_factory: Callable[[str], Dict[str, Any]],
    orchestrator_script: str,
    supports_consolidation: bool = False,
    consolidation_handlers: Optional[Dict[str, Callable]] = None,
    pre_run_hook: Optional[Callable[[], None]] = None,
    extra_validation_metadata: Optional[Dict[str, str]] = None,
    base_ref: Optional[str] = None,
    tasks_path: Optional[Path] = None,
    template_style: str = 'pr',
    diff_data: Optional[Dict] = None,
) -> None:
    """Main review orchestration flow shared across review phases.

    Args:
        args: Parsed command-line arguments (must have plan_file, models, etc.)
        phase_name: Phase directory/state name (e.g., 'review-plan', 'review-tasks')
        invoke_phase: Phase name for invoke_with_file_output (e.g., 'plan_review', 'task_review')
        salvage_phase_name: Phase name used in salvage request metadata
        prompt_template: The prompt template string
        prompt_context_factory: Factory function: model_spec -> prompt_context dict
        orchestrator_script: Script name for reaggregate command
        supports_consolidation: Whether this phase supports consolidation modes
        consolidation_handlers: Optional dict of consolidation mode handler functions
        pre_run_hook: Optional callable to execute before model invocation (e.g., validate tasks file)
        extra_validation_metadata: Optional dict of extra fields to pass to prepare_batched_validation_tasks
            and include in validation_tasks.json (e.g., {"tasks_file": "/path/to/tasks.md"})
        base_ref: Optional git ref for PR-style diff context (forwarded to
            ``aggregate_results()``).
        tasks_path: Optional path to tasks JSON for task-view metadata
            (forwarded to ``aggregate_results()``).
        template_style: Template to use — ``'pr'`` or ``'flat'``
            (forwarded to ``aggregate_results()``).  Defaults to ``'pr'``.
        diff_data: Optional pre-computed diff hunk data (forwarded to
            ``aggregate_results()``).
    """
    # Get absolute path for the plan file
    plan_path = os.path.abspath(args.plan_file)

    if not os.path.isfile(plan_path):
        print(f"ERROR: Plan file not found: {args.plan_file}")
        print(f"       Resolved path: {plan_path}")
        sys.exit(1)

    prefix = derive_prefix(plan_path)
    out_dir_arg = getattr(args, 'out_dir', None)
    base_dir = out_dir_arg or os.path.dirname(plan_path) or "."

    if os.path.basename(base_dir.rstrip(os.sep)) == prefix:
        out_dir = base_dir
    else:
        out_dir = os.path.join(base_dir, prefix)

    phase_dir = str(get_phase_dir(Path(plan_path), phase_name))

    # Handle --reaggregate mode
    if args.reaggregate:
        print("=== REAGGREGATE MODE ===")
        print(f"Plan file: {plan_path}")
        print(f"Output directory: {out_dir}")
        print(f"Phase directory: {phase_dir}")
        print("")

        if not os.path.isdir(phase_dir):
            print(f"ERROR: Phase directory does not exist: {phase_dir}")
            sys.exit(1)

        await reaggregate_from_existing_files(
            prefix, out_dir, phase_dir, phase_name, plan_path, args, orchestrator_script,
            extra_validation_metadata=extra_validation_metadata,
            base_ref=base_ref,
            tasks_path=tasks_path,
            template_style=template_style,
            diff_data=diff_data,
        )
        return

    # Handle consolidation modes (only for phases that support them)
    if supports_consolidation and consolidation_handlers:
        if getattr(args, 'consolidate_dry_run', False):
            print("=== CONSOLIDATION DRY RUN MODE ===")
            print(f"Plan file: {plan_path}")
            print(f"Phase directory: {phase_dir}")
            print("")
            if not os.path.isdir(phase_dir):
                print(f"ERROR: Phase directory does not exist: {phase_dir}")
                sys.exit(1)
            consolidation_handlers['dry_run'](phase_dir, plan_path)
            return

        if getattr(args, 'consolidate', False):
            print("=== CONSOLIDATION MODE ===")
            print(f"Plan file: {plan_path}")
            print(f"Phase directory: {phase_dir}")
            print("")
            if not os.path.isdir(phase_dir):
                print(f"ERROR: Phase directory does not exist: {phase_dir}")
                sys.exit(1)
            consolidation_handlers['consolidate'](phase_dir, plan_path)
            return

        if getattr(args, 'reaggregate_consolidation', False):
            print("=== REAGGREGATE CONSOLIDATION MODE ===")
            print(f"Plan file: {plan_path}")
            print(f"Phase directory: {phase_dir}")
            print("")
            if not os.path.isdir(phase_dir):
                print(f"ERROR: Phase directory does not exist: {phase_dir}")
                sys.exit(1)
            import glob as glob_module
            pattern_glob = os.path.join(phase_dir, "*.json")
            exclude = ['grouped', 'validation', 'salvage', 'revalidation', 'consolidated', 'consolidation', 'report_data']
            result_files = [
                f for f in glob_module.glob(pattern_glob)
                if not any(os.path.basename(f).startswith(ex) or ex in os.path.basename(f) for ex in exclude)
            ]
            models = [os.path.basename(f).replace('.json', '') for f in result_files]
            consolidation_handlers['reaggregate_consolidation'](phase_dir, plan_path, prefix, models)
            return

    # Guard against re-running already-completed phase
    state_guard = get_or_create_state(Path(plan_path))
    if state_guard.is_phase_completed(phase_name) and not args.force:
        print(f"ERROR: Phase '{phase_name}' has already been completed for this plan.")
        print(f"Output directory: {phase_dir}")
        print("Use --force to re-run, or --reaggregate to reprocess existing results.")
        sys.exit(2)
    elif args.force and state_guard.is_phase_completed(phase_name):
        state_guard.state.get("phases_completed", {}).pop(phase_name, None)
        state_guard.save()
        print("NOTE: Cleared previous phase completion (--force mode)")

    # Mutual exclusivity check
    if args.quick and args.interactive:
        print("ERROR: --quick and --interactive are mutually exclusive.")
        sys.exit(1)

    # Handle model selection
    try:
        model_specs = resolve_models(
            cli_models=args.models,
            interactive=args.interactive,
            quick=args.quick,
            mode=phase_name,
            anchor=plan_path,  # per-project config discovery follows the plan-derived root
        )
    except RuntimeError as e:
        print(f"ERROR: {e}")
        available = get_all_model_specs()
        print(f"Use --models flag. Available: {', '.join(available[:5])}...")
        sys.exit(1)

    if not model_specs:
        print("ERROR: No models selected.")
        sys.exit(1)

    # Display selection
    if args.models:
        print(f"Using models from --models flag: {', '.join(model_specs)}")
    elif args.interactive:
        print(f"Using interactively selected models: {', '.join(model_specs)}")
    elif args.quick:
        print(f"Using quick models from providers.yaml: {', '.join(model_specs)}")
    else:
        print(f"Using default models from providers.yaml: {', '.join(model_specs)}")

    # Validate model specs
    invalid_models = [m for m in model_specs if not is_model_valid(m, anchor=plan_path)]
    if invalid_models:
        print(f"WARNING: Unknown models (proceeding anyway): {', '.join(invalid_models)}")
        available = get_all_model_specs()
        print(f"Available models: {', '.join(available[:10])}...")

    # Run pre-run hook (e.g., validate tasks file existence)
    if pre_run_hook:
        pre_run_hook()

    # Read plan content
    plan_content = read_plan(plan_path)
    print(f"Loaded plan: {plan_path} ({len(plan_content)} bytes)")

    # Ensure output directories exist
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(phase_dir, exist_ok=True)

    # Create backup
    backup_path = os.path.join(phase_dir, "backup.md")
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(plan_content)
    print(f"Backup saved: {backup_path}")

    print(f"Output prefix: {prefix}")
    print(f"Output directory: {out_dir}")
    print(f"Models: {', '.join(model_specs)}")
    if args.timeout:
        print(f"Timeout override: {args.timeout}s per model")
    else:
        print("Timeout: per-provider defaults from providers.yaml")
    print(f"Max parallel: {args.max_parallel}")
    print("")

    write_status(phase_dir, {
        "phase": phase_name,
        "state": "models_running",
        "started_at": datetime.now().isoformat(),
        "models_requested": list(model_specs),
        "error": None,
    })

    # Run all models
    results = await run_all_models(
        model_specs,
        plan_path,
        prompt_template,
        prompt_context_factory,
        phase_name,
        invoke_phase,
        args.timeout,
        args.max_parallel,
        out_dir=out_dir,
        prefix=prefix,
        skip_existing=not getattr(args, "rerun_all", False),
    )

    # Save per-model results
    failed_models: Dict[str, str] = {}
    for model_spec, result in results.items():
        success = result["success"]
        output = result["output"]
        error = result["error"]

        if not save_model_result(prefix, model_spec, success, output, error, out_dir, phase_dir, salvage_phase_name):
            failed_models[model_spec] = error or "Unknown error"

    # Check if all failed
    if len(failed_models) == len(model_specs):
        print("\nERROR: All models failed")
        for model_spec, reason in failed_models.items():
            print(f"  - {model_spec}: {reason}")
        sys.exit(1)

    update_status(phase_dir, {
        "state": "models_complete",
        "models_completed": [m for m in model_specs if m not in failed_models],
        "models_failed": failed_models,
    })

    # Group similar suggestions
    all_suggestions = []
    for model_spec in model_specs:
        if model_spec in failed_models:
            continue
        sanitized_model = sanitize_model_name(model_spec)
        result_path = os.path.join(phase_dir, f"{sanitized_model}.json")
        if os.path.exists(result_path):
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    suggestions = json.load(f)
                    for s in suggestions:
                        s["source_model"] = model_spec
                    all_suggestions.extend(suggestions)
            except (json.JSONDecodeError, IOError):
                pass

    validated_groups_for_report: Optional[List[Dict]] = None

    if all_suggestions:
        raw_grouped = group_similar_suggestions(all_suggestions)
        grouped = [g.to_dict() if hasattr(g, 'to_dict') else g for g in raw_grouped]
        stamp_stable_ids(grouped)
        grouped_path = os.path.join(phase_dir, "grouped.json")
        with open(grouped_path, 'w', encoding='utf-8') as f:
            json.dump(save_groups_payload(grouped), f, indent=2)
        print(f"Grouped suggestions saved to: {grouped_path}")

        update_status(phase_dir, {
            "state": "grouping_complete",
            "total_groups": len(grouped),
            "total_suggestions": len(all_suggestions),
        })

        # Validate grouped suggestions
        if not args.skip_validation:
            internal_validation = getattr(args, 'internal_validation', False)
            if internal_validation:
                print("\nRunning internal validation...")
                validation_timeout = args.timeout if args.timeout else 600
                validation_results = await validate_groups(
                    [g.to_dict() if hasattr(g, 'to_dict') else g for g in grouped],
                    context=plan_content,
                    model=args.validation_model,
                    timeout=validation_timeout
                )

                validated_groups = apply_validation_to_groups(
                    [g.to_dict() if hasattr(g, 'to_dict') else g for g in grouped],
                    validation_results
                )
                validated_groups_for_report = validated_groups

                validation_path = os.path.join(phase_dir, "validation.json")
                save_validation_results(validation_results, Path(validation_path))
            else:
                print("\nPreparing batched validation for Claude Code subagent...")
                extra_kwargs = {}
                if extra_validation_metadata:
                    extra_kwargs = {k: v for k, v in extra_validation_metadata.items()
                                    if k in ("tasks_file",)}
                batched_tasks = prepare_batched_validation_tasks(
                    groups=[g.to_dict() if hasattr(g, 'to_dict') else g for g in grouped],
                    context=plan_content,
                    output_dir=phase_dir,
                    plan_file=str(plan_path),
                    model=args.validation_model,
                    **extra_kwargs,
                )
                batched_tasks["format_version"] = CURRENT_FORMAT_VERSION

                validation_tasks_path = os.path.join(phase_dir, "validation_tasks.json")
                with open(validation_tasks_path, 'w', encoding='utf-8') as f:
                    json.dump(batched_tasks, f, indent=2)

                total_batches = batched_tasks["total_batches"]
                stats = batched_tasks.get("batching_stats", {})

                if total_batches == 1:
                    print(f"\n[VALIDATION_PENDING] {validation_tasks_path}")
                else:
                    print(f"\n[VALIDATION_BATCHES_PENDING] {validation_tasks_path}")
                    print(f"Batches: {total_batches}")
                    if stats.get("efficiency_gain_percent", 0) > 0:
                        print(f"Efficiency gain: {stats['efficiency_gain_percent']}%")

                print(f"\nValidation prepared for Claude Code subagent.")
                print(f"Groups to validate: {stats.get('total_groups', len(grouped))}")
                print(f"HIGH priority: {stats.get('high_count', 0)}")
                print(f"Normal priority: {stats.get('normal_count', 0)}")
                print("\nAfter validation completes, run:")
                print(f"  {batched_tasks.get('reaggregate_command', '')}")

                update_status(phase_dir, {
                    "state": "validation_pending",
                    "stdout_markers": ["VALIDATION_BATCHES_PENDING"] if total_batches > 1 else ["VALIDATION_PENDING"],
                    "validation_tasks_path": validation_tasks_path,
                    "reaggregate_command": batched_tasks.get("reaggregate_command", ""),
                })

                return  # Exit early - Claude Code handles validation
        else:
            print("\nSkipping validation (--skip-validation flag set)")

    # Aggregate results
    report_path = aggregate_results(
        prefix, out_dir, phase_dir, phase_name,
        model_specs, failed_models, validated_groups_for_report,
        plan_path=plan_path,
        base_ref=base_ref,
        tasks_path=tasks_path,
        template_style=template_style,
        diff_data=diff_data,
    )

    # Handle consolidation recommendation (only for phases that support it)
    if supports_consolidation and validated_groups_for_report:
        from .consolidation import CONSOLIDATION_RECOMMENDED_THRESHOLD
        valid_group_count = sum(
            1 for g in validated_groups_for_report
            if g.get("validation_status") in ("valid", "needs-human-decision")
        )
        threshold = getattr(args, 'consolidation_threshold', None) or CONSOLIDATION_RECOMMENDED_THRESHOLD
        if valid_group_count >= threshold:
            print(f"\n[CONSOLIDATION_RECOMMENDED] {valid_group_count} valid groups >= threshold ({threshold})")
            print(f"Run: --consolidate to cluster related suggestions")

    # Mark phase as completed
    state = get_or_create_state(Path(plan_path))
    state.mark_phase_completed(phase_name)
    state.save()
