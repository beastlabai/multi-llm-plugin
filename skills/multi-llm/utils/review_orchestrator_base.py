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
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from . import detach as _detach
from . import status_file as _status_file
from .cli_common import (
    dispatcher_command,
    durable_reaggregate_fields,
    invocation,
    modes_table as _modes,
    record_next,
)
from .output_handler import sanitize_prefix, get_phase_dir
from .prompt_loader import load_prompt
from .interactive import resolve_models
from . import unattended as _unattended
from .unattended import NeedsHuman
from .provider_registry import (
    get_all_model_specs,
    parse_model_spec,
    get_provider,
    get_provider_timeout,
    get_provider_max_concurrent,
    get_subagent_retries,
    is_model_valid,
    set_run_anchor,
)
from .json_extractor import (
    build_unsanitize_map,
    generate_output_path,
    has_salvageable_json,
    read_json_from_file,
    sanitize_model_name,
)
from .llm_client import (
    STDOUT_OUTPUT_LOCATION,
    invoke_with_provider,
    invoke_with_file_output,
    render_output_location,
)
from .providers.base import (
    CATEGORY_CONNECTION_ERROR,
    CATEGORY_INVALID_MODEL,
    CATEGORY_PERMISSION_DENIED,
    CATEGORY_SERVER_ERROR,
    LLMProvider,
    classify_streams,
    context_length_message,
)
from .retry_ladder import (
    RetryLadder,
    STATUS_AUTH_FAILED,
    STATUS_CONTEXT_LENGTH_EXCEEDED,
    STATUS_OK,
    STATUS_RATE_LIMITED,
    STATUS_RUNNER_ERROR,
    STATUS_SPAWN_FAILED,
    STATUS_TIMEOUT,
    budget_exhausted_reason,
)
from .git_utils import get_project_root
from .html_report_generator import (
    generate_html_report,
    write_html_report,
    served_model_label,
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
    resolved_reaggregate_command,
)
from .subagent_tasks import (
    TaskFileError,
    build_inputs,
    read_task_file,
    write_salvage_task_file,
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
# The implementations moved to utils/status_file.py — unchanged semantics (same
# atomic temp+rename, same best-effort swallow, same special-cased merge, now
# extended to the reserved `run`/`progress` objects a detached run writes). They
# are re-exported here, and from review_plan_orchestrator as
# `_write_status`/`_update_status`, because this module is the status file's
# documented owner and every existing caller and test names it through one of
# those two paths.
#
# Why they live in their own module at all: `multi_llm.py status` must be a
# sub-second command (hosts with a 2-minute shell cap poll it in a loop), and
# importing this module pulls in the provider client, the HTML report generator
# and the validation stack.

write_status = _status_file.write_status
update_status = _status_file.update_status


def maybe_detach(
    *,
    args: argparse.Namespace,
    mode: str,
    phase_dir: str,
    plan_file: str,
    phase: Optional[str] = None,
    ask_slug: Optional[str] = None,
    model_specs: Optional[List[str]] = None,
) -> Optional[int]:
    """The one detach boundary a fan-out phase crosses.

    Returns ``None`` to keep running in this process (the ordinary foreground run,
    and also the detached worker once it owns the phase dir) or an exit code the
    caller must return immediately (the invoker, after forking; or a launch
    failure).
    """
    if getattr(args, "internal_worker", False):
        try:
            _detach.handle_phase_detach(
                args=args, mode=mode, argv=[], phase_dir=phase_dir,
                plan_file=plan_file, phase=phase, ask_slug=ask_slug,
            )
        except _detach.DetachError as exc:
            print(str(exc), file=sys.stderr)
            return exc.exit_code
        return None
    if not getattr(args, "detach", False):
        return None

    record = invocation()
    if not record or record.get("mode") != mode:
        # The child re-executes `uv run … multi_llm.py <mode> …`, rebuilt from the
        # dispatcher's normalized argv (never a remembered sys.argv). A direct
        # `python <orchestrator>.py --detach` has no such argv to rebuild from, and
        # guessing one from the parsed namespace would silently drop flags.
        print(
            "ERROR: --detach must be invoked through multi_llm.py — the detached "
            "child re-executes the dispatcher under the same `uv run --project` "
            "wrapping, which a direct orchestrator invocation cannot reconstruct."
            f"\n       Run: {dispatcher_command(mode, plan_file, '--detach')}",
            file=sys.stderr,
        )
        return 3
    try:
        return _detach.handle_phase_detach(
            args=args, mode=mode, argv=record["argv"], phase_dir=phase_dir,
            plan_file=plan_file, phase=phase, ask_slug=ask_slug,
            model_specs=model_specs,
        )
    except _detach.DetachError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code


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


def merge_validation_batches(
    phase_dir: str,
    batch_metadata: Dict[str, Any],
    grouped: List[Any],
    validation_path: str,
) -> List[Dict[str, Any]]:
    """Merge ``validation_batch_N.json`` files into ``validation.json`` + groups.

    Shared by the two reaggregation sites and by the ``--exec-subagents``
    continuation (D3b), which reaches the merge in the *same* invocation that
    dispatched the batch instead of exiting at the marker. One implementation
    means the executor path and the native path cannot drift in how a batch is
    merged — only in who dispatched it.
    """
    validation_results = merge_batched_validation_results(
        output_dir=phase_dir,
        batch_metadata=batch_metadata,
        total_groups=len(grouped),
    )
    save_validation_results(validation_results, Path(validation_path))
    print(f"Merged validation saved to: {validation_path}")
    return apply_validation_to_groups(
        [g.to_dict() if hasattr(g, 'to_dict') else g for g in grouped],
        validation_results,
    )


def exec_subagents_requested(args: Any) -> bool:
    """Whether this run drives its own subagents (D3b self-contained mode).

    A single accessor so every marker site asks the same question and the
    flag-off path stays byte-identical to what shipped before the executor.
    """
    return bool(getattr(args, "exec_subagents", False))


def dispatch_subagent_tasks(
    args: Any,
    *,
    task_file: str,
    phase_dir: str,
    phase: str,
    document: Optional[Dict[str, Any]] = None,
    plan_path: Optional[str] = None,
    label: str = "subagent",
    retry_command: Optional[str] = None,
) -> bool:
    """Run a materialized batch through the executor, or decline (D3b).

    Returns ``False`` when neither ``--exec-subagents`` nor ``--dry-run-subagents``
    was requested (the caller then prints its marker and follow-up command exactly
    as before), ``True`` when every task produced a schema-valid result file and
    the caller may continue in-process. A partial batch exits **3** here — even
    under ``--yes`` — rather than letting the phase reaggregate a short set.

    ``--dry-run-subagents`` is an executor request **in its own right** (D3l), the
    same way the write twin ``write_batch_scheduler.run_write_phase`` treats it:
    it renders the preflight report and returns ``False``, so the phase still
    prints its native marker. Gating it behind ``--exec-subagents`` would make a
    standalone dry run silently fall back to the native marker on every read,
    generation, consolidation and judging phase — no child *and* no report, which
    is the one outcome the flag must never produce.
    """
    dry_run = bool(getattr(args, "dry_run_subagents", False))
    if not exec_subagents_requested(args) and not dry_run:
        return False
    from .subagent_executor import (
        BatchFailed,
        ExecutorError,
        continue_past_marker,
        dry_run_report,
    )

    project_root = get_project_root(plan_path) if plan_path else None
    try:
        if dry_run:
            report = dry_run_report(
                task_file,
                phase_dir=phase_dir,
                phase=phase,
                document=document,
                anchor=plan_path,
                cwd=project_root,
            )
            print(
                f"[dry-run-subagents] {report} — every {label} command rendered, "
                f"no child started, nothing else changed. This phase still prints "
                f"its marker and its follow-up command."
            )
            return False
        return continue_past_marker(
            args,
            task_file=task_file,
            phase_dir=phase_dir,
            phase=phase,
            document=document,
            anchor=plan_path,
            cwd=project_root,
            label=label,
            retry_command=retry_command,
        )
    except BatchFailed as failure:
        print(failure.message, file=sys.stderr)
        update_status(phase_dir, {"state": "subagents_failed"})
        sys.exit(failure.exit_code)
    except ExecutorError as failure:
        print(f"ERROR: {failure.message}", file=sys.stderr)
        sys.exit(failure.exit_code)


def dispatch_salvage_tasks(
    args: Any,
    *,
    phase_dir: str,
    phase: str,
    plan_path: Optional[str] = None,
) -> List[str]:
    """Run every materialized ``salvage_task_*.json`` in a phase dir (D3b).

    Salvage repairs an unparseable *model review* on the fan-out path, so it is
    dispatched here — once, after the fan-out, when the requests exist — rather
    than from inside the per-model result writer, which has no flags and no
    phase-level view.

    **Deliberately non-fatal, unlike a validation batch.** A failed salvage leaves
    that model failed, which is exactly the partial-results outcome D6 row 3
    already covers (partial fan-out results are usable output); it is not the
    "reaggregating a short set" D3i forbids, because the *validation* set is still
    complete. A failed salvage is reported in ``subagent_batch_report.json`` like
    any other task.

    Returns:
        The output paths successfully repaired, in dispatch order.
    """
    if not exec_subagents_requested(args):
        return []
    if getattr(args, "dry_run_subagents", False):
        # D3l: "start no child" has no exception for salvage. Salvage is the one
        # non-fatal dispatch, which is precisely why it would be the easiest place
        # for a dry run to quietly spawn something.
        print(
            "[dry-run-subagents] skipping salvage dispatch: no child is started "
            "under this flag, so no model result is repaired."
        )
        return []
    import glob as glob_module

    from .subagent_executor import ExecutorError, run_task_file

    repaired: List[str] = []
    project_root = get_project_root(plan_path) if plan_path else None
    for task_file in sorted(
        glob_module.glob(os.path.join(phase_dir, "salvage_task_*.json"))
    ):
        print(f"\n[exec-subagents] dispatching salvage task from {task_file}")
        try:
            report = run_task_file(
                task_file,
                phase_dir=phase_dir,
                phase=phase,
                anchor=plan_path,
                cwd=project_root,
            )
        except ExecutorError as failure:
            print(f"  salvage dispatch refused: {failure.message}", file=sys.stderr)
            continue
        for outcome in report.outcomes:
            if outcome.ok and outcome.output_path:
                repaired.append(outcome.output_path)
            else:
                print(
                    f"  salvage {outcome.task_id} failed ({outcome.status}): "
                    f"{outcome.reason or ''}",
                    file=sys.stderr,
                )
    return repaired


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


#: The ``Reason:`` line, keyed by the shared classifier's category (§7.1.a).
#:
#: This replaces a seven-branch ``elif`` chain of raw substring tests that had
#: drifted from every other classifier in the tree: its first branch called
#: **every** provider's failure "cursor-agent command not found" (and swallowed
#: "model not found" with a bare ``'not found' in stderr``), its rate-limit
#: branch disagreed with the retry decision three hundred lines below it, and
#: ``'insufficient'`` matched any prose containing the word.
#:
#: A lookup rather than a chain is the point: the sentence a user acts on and
#: the retry decision now read the *same* verdict off the *same* streams, so
#: they can no longer contradict each other.
#:
#: The old "Quota exceeded or insufficient credits" branch is deliberately gone.
#: A quota notice is what the shared classifier calls ``rate_limited`` — that is
#: the rung it is retried on — and a second name for it here is exactly the
#: disagreement this table removes.
_REASON_BY_CATEGORY = {
    STATUS_SPAWN_FAILED:
        "Reason: the runner CLI could not be started (is it installed and in PATH?)",
    CATEGORY_PERMISSION_DENIED: "Reason: Permission denied",
    STATUS_AUTH_FAILED: "Reason: Authentication error (check API key)",
    CATEGORY_INVALID_MODEL: "Reason: Model not found or invalid",
    STATUS_RATE_LIMITED: "Reason: Rate limited",
    STATUS_TIMEOUT: "Reason: Timed out",
    CATEGORY_CONNECTION_ERROR: "Reason: Connection/network error",
    CATEGORY_SERVER_ERROR: "Reason: API/Server error",
    STATUS_RUNNER_ERROR: "Reason: the runner reported a failure",
}

_REASON_UNKNOWN = "Reason: Unknown (see tail output below)"


#: The one translation from §7.2's classifier vocabulary into the D3k ladder's
#: (§7.1.b). ``classify_streams`` returns four ``CATEGORY_*`` names the ladder has
#: no rung for; two of them are transport-shaped and therefore rung 4, which is
#: what makes a connection error or a 5xx spend ``driver.subagent_retries``
#: exactly once — the behaviour the deleted ``retry_count < 1`` recursion had.
#:
#: Everything **absent** from this table reaches :meth:`RetryLadder.record` under
#: its own name, and that is deliberate rather than an oversight:
#: ``rate_limited`` and ``timeout`` are already ladder statuses; ``auth_failed``
#: and ``spawn_failed`` are rung 1; and ``invalid_model``,
#: ``permission_denied``, ``context_length_exceeded`` and the unrecognized
#: ``None`` all fall through ``record``'s last branch to "never retried", which
#: is exactly what the deleted ``failure_category in
#: RETRYABLE_FAILURE_CATEGORIES`` guard decided. One table, not two.
#: ``tests/test_lane_a_ladder.py`` pins the two vocabularies in agreement.
_LADDER_STATUS_BY_CATEGORY = {
    CATEGORY_CONNECTION_ERROR: STATUS_RUNNER_ERROR,
    CATEGORY_SERVER_ERROR: STATUS_RUNNER_ERROR,
}


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
    max_tail_lines: int = 20,
    *,
    provider: Optional[LLMProvider] = None,
) -> str:
    """Format failure details with clear reason and tail output.

    The reason is the shared classifier's verdict looked up in
    :data:`_REASON_BY_CATEGORY` (§7.1.a) — there is no second rule here.

    **``provider`` is how this helper stops being the one classifier that reads
    raw stdout.** It is handed a whole ``stdout_text``, and four shipped
    adapters echo our prompt into that stream, so on the committed goose
    ``prompt-echo`` bytes this function answered ``Reason: Timed out`` about a
    run that had *succeeded*. Passing the adapter routes the verdict through
    :meth:`~utils.providers.base.LLMProvider.classify_failure`, which applies
    that adapter's :meth:`~utils.providers.base.LLMProvider.vendor_text`
    reduction first — and any caller that has an adapter in hand must pass it.

    With no adapter the behaviour is exactly what it was: the module-level
    :func:`classify_streams` against :class:`~utils.providers.base.LLMProvider`'s
    own declared pattern sets. That path is deliberately kept rather than made
    an error, because §7.2 depends on ``classify_streams`` staying callable with
    no adapter (``validation.py`` is the other such caller) — but it can only
    under-reduce, never mis-reduce, and it is not a path a production caller
    should be on.
    """
    error_parts = [f"Exit code {returncode}"]

    category = (
        provider.classify_failure(stdout_text, stderr_text, returncode)
        if provider is not None
        else classify_streams(stdout_text, stderr_text, returncode)
    )
    if category == STATUS_CONTEXT_LENGTH_EXCEEDED:
        error_parts.append(f"Reason: {context_length_message('')}")
    else:
        error_parts.append(_REASON_BY_CATEGORY.get(category, _REASON_UNKNOWN))

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

def served_model_fields(details: Dict[str, Any]) -> Dict[str, Any]:
    """§3.6's attribution keys from an invocation's ``details``, or ``{}``.

    Splatted into a per-model result record so the fan-out carries what the
    adapter read. Absent means "nothing was claimed" — every adapter but
    claude-code, and every claude-code run under the shipped empty
    ``fallback_models`` — and the record then has exactly the keys it has
    always had.
    """
    served = (details or {}).get("served_model")
    if not served:
        return {}
    return {
        "served_model": served,
        "served_model_is_fallback": bool(details.get("served_model_is_fallback")),
    }


def collect_served_models(
    results: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """``{model_spec: {"served_model", "fallback"}}`` for a fan-out's results.

    The report-side shape of the same fact, built from the per-model records
    :func:`run_single_model` returns. Empty for a default run, which is what
    keeps every downstream artifact unchanged.
    """
    served: Dict[str, Dict[str, Any]] = {}
    for model_spec, result in (results or {}).items():
        if not isinstance(result, dict):
            continue
        name = result.get("served_model")
        if name:
            served[model_spec] = {
                "served_model": name,
                "fallback": bool(result.get("served_model_is_fallback")),
            }
    return served


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

    Retries are the shared :class:`~utils.retry_ladder.RetryLadder` (§7.1.b),
    **not** the recursion this function used to be. Two self-calls — one for a
    rate limit, one for a network error — each re-entered with the *same*
    ``timeout`` argument, so a claude-code slot at ``default_timeout: 1800``
    could run 4 × 1800 s + 35 s of backoff ≈ 2 h against a configured ceiling of
    1800 s (§7.5's opening paragraph). Being a tail call inside an ``async def``
    it also kept its retry state in a place no budget could see.

    The loop below is the sketch in §7.1.b, and its only lane-specific lines are
    the ``await``ed dispatch and the ``await asyncio.sleep``. It is deliberately
    **not** Lane B's loop: this coroutine is awaited inside ``asyncio.gather``
    under two semaphores, so a blocking retry loop here would serialize the whole
    fan-out. What the two lanes share is the policy object.

    Args:
        model_spec: Model specification (e.g., 'cursor-agent:auto')
        plan_path: Absolute path to the plan file
        prompt_template: The prompt template string
        prompt_context: Dict of variables to substitute in the prompt
        phase_name: Phase directory name (e.g., 'review-plan', 'review-tasks')
        invoke_phase: Phase name for invoke_with_file_output (e.g., 'plan_review', 'task_review')
        timeout: Optional timeout override in seconds — the slot's **whole**
            wall-clock budget, taken once as a deadline and never re-issued.
        retry_count: Vestigial. Retry state lives in the ladder now, which is the
            point of §7.5 rule 1; the parameter survives only because
            ``review_plan_orchestrator``'s backward-compatible wrapper forwards
            it. Passing a non-zero value has no effect.
        out_dir: Optional output directory for log files and JSON output
        prefix: Optional prefix for log file names

    Returns:
        Dict with keys: success, output, error, prompt, stdout, stderr, duration_seconds, model_spec, source
    """
    import time
    start_time = time.time()

    project_root = get_project_root(plan_path)
    # `plan_path` IS the discovery anchor (§9.3.a): no new value is computed
    # here, the existing one simply stops being thrown away at the spawn.
    provider_name, model_name = parse_model_spec(model_spec, anchor=plan_path)
    display_name = f"{provider_name}:{model_name}"

    effective_prefix = prefix or sanitize_prefix(os.path.basename(plan_path))
    full_prompt_context = {**prompt_context}
    if "prefix" not in full_prompt_context:
        full_prompt_context["prefix"] = effective_prefix

    effective_timeout = (
        timeout if timeout is not None
        else get_provider_timeout(provider_name, anchor=plan_path)
    )

    log_file = None
    if out_dir and prefix:
        sanitized_model = sanitize_model_name(model_spec)
        phase_dir = get_phase_dir(Path(plan_path), phase_name)
        os.makedirs(phase_dir, exist_ok=True)
        log_file = os.path.join(str(phase_dir), f"log_{sanitized_model}.txt")

    # Fall back to old method if no out_dir provided
    if not out_dir:
        # There is no output file on this path at all — the result is whatever
        # `invoke_with_provider` parses off stdout — so the prompt asks for
        # stdout outright, for every provider. It used to interpolate the string
        # "(stdout)" into "Write your JSON output to this file:", which named a
        # channel as if it were a path.
        prompt = prompt_template.format(
            **full_prompt_context,
            output_json_path="(stdout)",
            output_location=STDOUT_OUTPUT_LOCATION,
        )
        try:
            print(f"[{display_name}] Starting review (timeout: {effective_timeout}s)...")
            result = await asyncio.to_thread(
                invoke_with_provider,
                prompt=prompt,
                model_spec=model_spec,
                timeout=int(effective_timeout),
                log_file=log_file,
                cwd=project_root,
                anchor=plan_path,
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
                    "source": "stdout",
                    **served_model_fields(details),
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

    result: Optional[Dict[str, Any]] = None
    details: Dict[str, Any] = {}
    stderr_text = ""
    source = "unknown"
    status: Optional[str] = None
    action = None

    try:
        # §7.5 rule 1: ONE deadline for this slot, taken once, here. Every
        # attempt below is spawned with what is LEFT of it — the whole of M9's
        # budget half. The declared vendor envelope (§7.6(b)) turns on rule 4's
        # pre-dispatch admission test and rule 6(b)'s refusal to re-issue a
        # rate-limit dispatch a CLI already exhausted internally; `getattr`
        # because the eleven unmeasured adapters declare `None`, which rule 6(c)
        # reads as "assume one request", never "assume the worst".
        provider = get_provider(provider_name)
        ladder = RetryLadder(
            timeout=effective_timeout,
            retries=get_subagent_retries(anchor=plan_path),
            envelope=getattr(provider, "internal_retries", None),
            now=time.monotonic,
        )
        # A placeholder only until the first attempt names its real output file:
        # rule 4 can refuse this slot before any dispatch, and the result dict
        # still owes its caller a `prompt`. The output-location block is already
        # this provider's real one — it depends on the adapter, not on the path.
        prompt = prompt_template.format(
            **full_prompt_context,
            output_json_path="",
            output_location=render_output_location("", model_spec, anchor=plan_path),
        )

        while (slice_s := ladder.next_attempt()) is not None:
            # Never a fresh copy of `effective_timeout`: `next_attempt()` returns
            # what is left of the one deadline, and rule 3 guarantees it is at
            # least MIN_ATTEMPT_BUDGET_S, so the floor below can never be 0.
            # `round`, not `int`: `invoke_with_file_output` takes whole seconds,
            # and truncating would silently spend up to a second of the slot's
            # budget on every attempt — including the first, whose slice is
            # `effective_timeout` minus the microseconds since the ladder was
            # constructed. The ≤ 0.5 s of overshoot the other way is bounded and
            # sits inside the fan-out's `base_timeout × 2` of slack.
            attempt_timeout = max(1, int(round(slice_s)))
            print(
                f"[{display_name}] Starting review with file-based output "
                f"(attempt {ladder.attempt_index}, timeout: {attempt_timeout}s "
                f"of a {effective_timeout}s slot)..."
            )

            result = await asyncio.to_thread(
                invoke_with_file_output,
                prompt_template=prompt_template,
                model_spec=model_spec,
                prompt_context=full_prompt_context,
                output_dir=out_dir,
                phase=invoke_phase,
                timeout=attempt_timeout,
                log_file=log_file,
                prefer_arrays=True,
                cwd=project_root,
                anchor=plan_path,
            )

            details = result.get("details", {})
            stderr_text = details.get("stderr", "")
            source = result.get("source", "unknown")
            output_file = result.get("output_file", "")

            # The prompt this attempt really sent, reconstructed for the record:
            # `invoke_with_file_output` renders the same two values from the same
            # spec, so this stays the text the child saw rather than a
            # file-flavoured approximation of it.
            prompt = prompt_template.format(
                **full_prompt_context,
                output_json_path=output_file,
                output_location=render_output_location(
                    output_file, model_spec, anchor=plan_path
                ),
            )

            if output_file:
                print(f"[{display_name}] Output file: {output_file} (source: {source})")

            # §7.1.a/§7.2: ONE classification per failed slot, read off the
            # shared classifier, replacing the three raw-substring tests that
            # used to live in this function and disagree with each other. The
            # boundary- and context-awareness L2 introduced is preserved because
            # it is the *same* implementation — a model that wrote "reviewed 429
            # files" still costs one dispatch, not four full-timeout ones plus
            # 35s of backoff.
            # The haystack is the invocation's own ``error`` string and nothing
            # else, exactly as L2 left it. That string is already what
            # ``_format_failure_details`` built from *both* streams, so adding
            # ``stderr_text`` beside it would not add information — it would
            # duplicate the same prose and let the tail of one copy supply the
            # error-context word for a bare token in the other, which is
            # precisely the false positive ``matches_any``'s 48-character window
            # exists to prevent.
            failure_category = (
                None if result.get("success")
                else classify_streams(str(result.get("error", "")), "")
            )
            status = (
                STATUS_OK if result.get("success")
                else _LADDER_STATUS_BY_CATEGORY.get(failure_category, failure_category)
            )

            # One `record` call is the whole retry decision, identical to Lane
            # B's: rung 1 is terminal, rung 3 backs off with jitter at most three
            # times without consuming the transport budget, rungs 2/4/5 spend
            # `driver.subagent_retries`, everything else is never retried — and
            # every one of those decisions is taken against the same deadline the
            # dispatches are charged to. `retry_count < 3` and `retry_count < 1`
            # are gone; they were two counters neither of which saw a clock.
            action = ladder.record(status)
            if not action.retry:
                break
            if action.delay:
                # The ladder is pure, so the test fast-backoff switch cannot live
                # there. Applying it here, at the one place this lane really
                # sleeps, keeps MULTI_LLM_TEST_FAST_BACKOFF=1 an exact 10 ms.
                delay = get_backoff_delay(action.delay)
                print(f"[{display_name}] {status}, retrying in {delay}s...")
                await asyncio.sleep(delay)

        duration = time.time() - start_time

        if result is None:
            # §7.5 rule 4: the slot was refused before anything was spawned. It
            # is recorded as its own outcome rather than dispatched to be killed
            # part-way through a retry chain the harness cannot see.
            reason = budget_exhausted_reason(
                ladder, target=display_name, timeout=effective_timeout
            )
            print(f"[{display_name}] {reason}")
            return {
                "success": False, "output": None,
                "error": reason,
                "prompt": prompt, "stdout": "", "stderr": "",
                "duration_seconds": duration, "model_spec": model_spec,
                "source": "not-dispatched",
            }

        # The ladder granted a retry and `next_attempt()` then refused it: what
        # ended this slot is the budget, not the last attempt's error. Lane B
        # appends the same sentence (`subagent_executor.py`'s STOP_DEADLINE leg),
        # because without it the recorded reason names a transport failure that
        # the harness had, and did not spend, retries for.
        budget_note = ""
        if action is not None and action.retry:
            budget_note = (
                f" (no retry: the per-slot wall-clock budget of "
                f"{effective_timeout}s is exhausted — {ladder.stop_reason})"
            )

        if status == STATUS_RATE_LIMITED:
            # Either the ≤ 3 backoffs are spent, or §7.5 rule 6(b) refused an
            # identical re-dispatch against an adapter whose CLI already ran its
            # own 429 chain. `action.reason` names which, and — for 6(b) — the
            # declared envelope that decided it.
            return {
                "success": False, "output": None,
                "error": f"Rate limited (429) - {action.reason}{budget_note}",
                "prompt": prompt, "stdout": "", "stderr": stderr_text,
                "duration_seconds": duration, "model_spec": model_spec,
                "source": source
            }

        # §7.6(a): a context overflow is TERMINAL. Splitting the prompt is open
        # question 13 and nobody has designed it, so the shipping behaviour is to
        # classify it, name it, and fail the slot with the two remedies a human
        # can actually apply — never a re-dispatch, because re-sending an
        # unchanged oversized prompt is the one retry guaranteed to fail
        # identically. The old test here was `'context' in str(error)`, which
        # also fired on `context_mode`, on "in this context", and on any prose
        # quoting the plan's own Context section.
        if status == STATUS_CONTEXT_LENGTH_EXCEEDED:
            return {
                "success": False, "output": None,
                "error": context_length_message(model_spec),
                "prompt": prompt, "stdout": "", "stderr": stderr_text,
                "duration_seconds": duration, "model_spec": model_spec,
                "source": source
            }

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
                "source": source,
                # §3.6: the served model rides out with the answer it produced.
                **served_model_fields(details),
            }
        else:
            error_details = f"{result.get('error', 'Unknown error')}{budget_note}"
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
    anchor: Optional[Union[str, Path]] = None,
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
        anchor: Optional discovery anchor (the run's plan-file path). This
            function has no plan path of its own, so without it every per-provider
            bound and every provider resolution below falls back to CWD (§9.3.c).
            ``run_review_phase`` and ``ask_orchestrator`` both hold the plan path
            and pass it.

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
        prov, _ = parse_model_spec(spec, anchor=anchor)
        if prov not in provider_semaphores:
            limit = get_provider_max_concurrent(prov, anchor=anchor)
            if limit is not None:
                provider_semaphores[prov] = asyncio.Semaphore(limit)

    async def run_with_semaphore(model_spec: str, index: int) -> Tuple[str, Any]:
        # Display-only, but still anchored: a bare spec must render the same
        # provider the router chose, not a second, different guess (§9.3.c).
        provider_name, model_name = parse_model_spec(model_spec, anchor=anchor)
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
        base_timeout = max(
            get_provider_timeout(parse_model_spec(spec, anchor=anchor)[0], anchor=anchor)
            for spec in model_specs
        )

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


async def run_tasks_concurrent(
    tasks: List[Dict[str, Any]],
    per_task_coro_factory: Callable[[Dict[str, Any], str, int], Any],
    *,
    runner_for_task: Callable[[Dict[str, Any]], str],
    timeout: Optional[float] = None,
    max_parallel: int = 4,
    skip_predicate: Optional[Callable[[Dict[str, Any], str, int], Optional[Any]]] = None,
    provider_limits: Optional[Dict[str, int]] = None,
    anchor: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """:func:`run_models_concurrent`'s mechanics, keyed by ``task_id`` (D3).

    ``run_models_concurrent`` returns a dict keyed by ``model_spec`` — correct for
    model fan-out, **wrong** for a subagent batch, where every task normally
    shares the one configured runner, so completed tasks would overwrite one
    another. This is the same machinery keyed by each task's stable ``task_id``:

    - the same global ``asyncio.Semaphore(max_parallel)``
    - the same per-provider semaphores from ``get_provider_max_concurrent``,
      acquired *before* the global one, and **shared in shape** with the fan-out
      path so a mixed-tier batch on one provider contends on ONE semaphore
      (the rate limit belongs to the provider, not the model)
    - the same stagger, bucketed by resolved provider
    - the same overall timeout wrapping ``asyncio.gather``

    ``model_spec`` is used *only* to pick the provider semaphore and stagger
    bucket; it is a column on each row, never the key. ``run_models_concurrent``
    keeps its current signature and callers.

    Args:
        tasks: The materialized task objects (not model specs — each task carries
            its own prompt and its runner is a *derived* attribute).
        per_task_coro_factory: ``(task, runner_spec, index) -> awaitable``.
        runner_for_task: Resolves a task's runner spec once, at batch start.
        timeout: Per-task timeout override, also driving the overall timeout.
        max_parallel: The global gate (``driver.max_concurrent_subagents``,
            already reduced to D3i's three-way minimum by the caller).
        skip_predicate: Cheap synchronous ``(task, runner, index) -> result|None``
            checked *before* the stagger sleep and any semaphore acquisition, so a
            resume of already-complete tasks stays fast.
        provider_limits: Optional per-provider in-flight overrides (D3i
            backpressure narrows a throttled provider's slots for the rest of the
            batch); defaults to ``get_provider_max_concurrent``.
        anchor: Optional discovery anchor (the run's plan-file path), the Lane B
            twin of ``run_models_concurrent``'s (§9.3.c). ``subagent_executor``
            already holds the phase's plan path and passes it.

    Returns:
        ``{task_id: whatever the callback returned}``.
    """
    results: Dict[str, Any] = {}
    if not tasks:
        return results

    runners = {
        str(task.get("task_id", index)): runner_for_task(task)
        for index, task in enumerate(tasks)
    }

    semaphore = asyncio.Semaphore(max_parallel)

    provider_semaphores: Dict[str, asyncio.Semaphore] = {}
    for spec in set(runners.values()):
        prov, _ = parse_model_spec(spec, anchor=anchor)
        if prov in provider_semaphores:
            continue
        limit = (provider_limits or {}).get(
            prov, get_provider_max_concurrent(prov, anchor=anchor)
        )
        if limit is not None:
            provider_semaphores[prov] = asyncio.Semaphore(limit)

    async def run_with_semaphore(task: Dict[str, Any], index: int) -> Tuple[str, Any]:
        task_id = str(task.get("task_id", index))
        runner = runners[task_id]
        # Anchored for the same reason the fan-out's display resolution is: the
        # semaphore this picks must be the one the batch was sized for (§9.3.c).
        provider_name, _ = parse_model_spec(runner, anchor=anchor)

        if skip_predicate is not None:
            skipped = skip_predicate(task, runner, index)
            if skipped is not None:
                return (task_id, skipped)

        if index > 0:
            stagger_delay = get_backoff_delay(index * PROVIDER_STAGGER_DELAY)
            await asyncio.sleep(stagger_delay)

        prov_sem = provider_semaphores.get(provider_name)
        if prov_sem:
            async with prov_sem:
                async with semaphore:
                    return (task_id, await per_task_coro_factory(task, runner, index))
        else:
            async with semaphore:
                return (task_id, await per_task_coro_factory(task, runner, index))

    coros = [
        asyncio.create_task(run_with_semaphore(task, idx))
        for idx, task in enumerate(tasks)
    ]

    if timeout is not None:
        base_timeout = timeout
    else:
        base_timeout = max(
            get_provider_timeout(parse_model_spec(spec, anchor=anchor)[0], anchor=anchor)
            for spec in runners.values()
        )
    total_timeout = base_timeout * 2 + len(tasks) * PROVIDER_STAGGER_DELAY

    try:
        completed = await asyncio.wait_for(
            asyncio.gather(*coros, return_exceptions=True), timeout=total_timeout
        )
        for item in completed:
            if isinstance(item, Exception):
                print(f"Subagent task exception: {item}")
                continue
            task_id, result = item
            results[task_id] = result
    except asyncio.TimeoutError:
        print(f"Total subagent batch timeout ({total_timeout}s) exceeded")
        for coro in coros:
            if not coro.done():
                coro.cancel()

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
        provider_name, model_name = parse_model_spec(model_spec, anchor=plan_path)
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
        anchor=plan_path,
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
        # The fan-out's sibling of code_review_orchestrator's two salvage sites,
        # and it holds the same rule: salvage only when the reply still holds an
        # answer to repair. A model that answered in prose — §40's write
        # refusals, a provider printing its auth failure on the channel it
        # exited 0 on — reaches this `except` with not one character of JSON,
        # and the salvage prompt's "if nothing recoverable is present, print an
        # empty array" would then write `[]` over `output_path`: a model that
        # produced nothing, recorded as one that reviewed the document and found
        # nothing. Unsalvageable is not fatal here either — the model stays
        # failed and D6 row 3 aggregates what the other slots returned — so the
        # error log below is written on both paths and only its wording moves.
        salvageable = has_salvageable_json(output)
        if salvageable:
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
            # The salvage prompt is materialized in Python (D2/D2i) instead of
            # being assembled from references/salvage-handling.md, so any invoker
            # can run it. Best-effort: a salvage request that cannot be paired
            # with a task file is still recoverable by hand, and must not mask
            # the parse error that produced it.
            try:
                salvage_task_path = write_salvage_task_file(
                    salvage_request=salvage_request,
                    salvage_request_path=salvage_path,
                    phase_dir=phase_dir,
                )
                print(f"[SALVAGE_TASK] {salvage_task_path}")
            except Exception as task_error:  # noqa: BLE001 - diagnostics only
                print(f"[{model}] Warning: could not materialize salvage task: {task_error}")

        error_path = os.path.join(phase_dir, f"error_{sanitized_model}.log")
        with open(error_path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {model}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"{'Parse error' if salvageable else 'No JSON in response'}: {e}\n")
            f.write(f"\nRaw output:\n{output}\n")
        if salvageable:
            print(f"[{model}] JSON parse error, raw output saved to: {error_path}")
        else:
            print(
                f"[{model}] Response carries no JSON — nothing to salvage, "
                f"raw output saved to: {error_path}"
            )
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
    served_models: Optional[Dict[str, Dict[str, Any]]] = None,
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
        served_models: Optional §3.6 attribution
            (``{model_spec: {"served_model", "fallback"}}``, from
            :func:`collect_served_models`).  A model that was served by a
            configured fallback is named as such on the ``**Models:**`` line and
            in the HTML report; with nothing configured this is empty and the
            report is byte-for-byte what it was.

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
        f"**Models:** {', '.join(served_model_label(m, served_models) for m in successful_models)}",
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
            served_models=served_models,
        )
        write_html_report(html_content, Path(phase_dir))
        print(f"HTML report written to: {os.path.join(phase_dir, 'report.html')}")
    except Exception as e:
        print(f"Warning: Failed to generate HTML report: {e}")

    # A detached run's caller never saw this stdout, so success has to hand back
    # paths rather than a search: `status` on a `done` run prints these and
    # repeats them in MULTI_LLM_RESULT.artifacts. A no-op in the foreground, which
    # is what keeps a non-detached .status.json byte-for-byte what it was.
    _detach.record_output_files(
        phase_dir,
        report=report_path,
        report_html=os.path.join(phase_dir, "report.html"),
        grouped=os.path.join(phase_dir, "grouped.json"),
    )

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
    # "Let Claude decide" routes this item to the per-item judge at apply time.
    # Scoped to needs-human-decision only (validation_failed stays valid/invalid
    # or --revalidate).
    if validation_status == "needs-human-decision":
        lines.append("- [ ] Let Claude decide")

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
        # "Let Claude decide" — needs-human-decision only. Single-suggestion
        # groups use the group-level checkbox (no per-suggestion select here).
        if group_validation_status == "needs-human-decision":
            lines.append("- [ ] Let Claude decide")
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

def resolve_result_files_by_model(
    phase_dir: str,
    result_files: List[str],
    unsanitize: Dict[str, str],
    *,
    noun: str = "suggestions",
) -> List[Tuple[str, str, Any]]:
    """Map globbed result files to logical models — one file per model (W3).

    A phase dir that straddles the L1 anchor upgrade holds BOTH ``<model>.json``
    (written before L1) and ``<provider>_<model>.json`` (written by the resume,
    which looked for the anchored name, did not find it, and **re-ran** that
    model). Every ``--reaggregate`` path globs ``*.json``, so it sees two files;
    they are two answers from ONE logical model, and reading both double-counts
    every finding. That is not cosmetic: ``source_model`` is what grouping
    attributes by, and ``SuggestionGroup._recalculate_priority`` scores
    ``len(self.models) + Σ importance``, so a finding read twice renders as two
    models agreeing when one model said it once — corroboration that did not
    happen. Collapsing the model *set* alone would not fix it; the payload of
    exactly one file per model has to be the one that reaches the aggregate.

    Identity is the **recorded canonical spec** in ``.status.json``, never the
    filename: ``sanitize_model_name`` is lossy in one direction only, so a legacy
    stem cannot be re-resolved into a spec, but it CAN be matched against the
    stems the recorded specs generate. (It is also the only workable direction
    because ``write_status`` replaces ``models_requested`` wholesale, so after the
    resume the checkpoint holds the canonical specs only.)

    Shared by both ``--reaggregate`` implementations — this module's, and
    ``code_review_orchestrator.reaggregate_from_existing_files``, which
    accumulates into ``all_issues``/``results`` and would otherwise render the
    same model twice in its report. *unsanitize* is passed in rather than built
    here so each caller keeps its own ``build_unsanitize_map`` seam.

    Args:
        phase_dir: The phase directory the files were globbed from.
        result_files: Candidate model-result paths, already filtered of
            non-model artifacts by the caller's ``exclude_patterns``.
        unsanitize: ``build_unsanitize_map(phase_dir)`` — ``sanitize(spec) -> spec``
            over the checkpoint's ``models_requested``/``models_completed``.
        noun: What the payloads are called in the log lines ("suggestions",
            "issues").

    Returns:
        ``[(model, chosen_path, payload)]`` ordered by model, exactly one entry
        per logical model. Unreadable files are warned about and dropped, as
        before.

    Raises:
        SystemExit: exit 3 when one legacy stem matches two or more recorded
            specs — the genuine ambiguity, which fails closed.
    """
    recorded_specs = sorted(set(unsanitize.values()))
    anchored_stems = {sanitize_model_name(spec): spec for spec in recorded_specs}
    legacy_stems: Dict[str, List[str]] = {}
    for spec in recorded_specs:
        if ":" not in spec:
            continue
        legacy_stems.setdefault(
            sanitize_model_name(spec.split(":", 1)[1]), []
        ).append(spec)

    def resolve_recorded_identity(stem: str) -> Tuple[str, bool]:
        """Map a result-file stem to ``(model identity, is_anchored)``."""
        if stem in anchored_stems:
            return anchored_stems[stem], True
        legacy_candidates = legacy_stems.get(stem, [])
        if len(legacy_candidates) > 1:
            # `openai:gpt-5.2` and `openrouter:gpt-5.2` both generate the legacy
            # stem `gpt-5_2`, and there is no evidence on disk for which one wrote
            # the file. Attributing it to either fabricates a `source_model`;
            # splitting it across both is the double count again. Fail closed on
            # the same needs-human code the resume reader uses on this path.
            print(
                "ERROR: "
                f"{os.path.join(phase_dir, stem + '.json')} is a pre-anchor result "
                "file whose name matches more than one model recorded in "
                f"{os.path.join(phase_dir, '.status.json')}: "
                + ", ".join(legacy_candidates)
                + ". Remove it, or rename it to the <provider>_<model>.json name of "
                "the model that wrote it, then re-run --reaggregate.",
                file=sys.stderr,
            )
            sys.exit(3)
        if legacy_candidates:
            return legacy_candidates[0], False
        # No recorded spec claims this stem — a hand-copied dir, or one with no
        # `.status.json` at all: the stem IS the identity, exactly as before.
        return unsanitize.get(stem, stem), True

    # identity -> [(is_anchored, path, payload)]
    files_by_model: Dict[str, List[Tuple[bool, str, Any]]] = {}
    for result_path in sorted(result_files):
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Warning: Could not read {result_path}: {e}")
            continue
        stem = os.path.basename(result_path)[: -len(".json")]
        model, is_anchored = resolve_recorded_identity(stem)
        files_by_model.setdefault(model, []).append(
            (is_anchored, result_path, payload)
        )

    resolved: List[Tuple[str, str, Any]] = []
    for model in sorted(files_by_model):
        # The anchored file wins unconditionally, identical payloads or not:
        # `sanitize_model_name(model_spec) + ".json"` is precisely what the normal
        # path reads, so "anchored wins" is what makes --reaggregate agree with the
        # run it is re-aggregating, and it is the newer file by construction.
        candidates = sorted(files_by_model[model], key=lambda e: (not e[0], e[1]))
        _, chosen_path, payload = candidates[0]
        for _, skipped_path, skipped in candidates[1:]:
            if skipped != payload:
                # Salvage never refuses a whole phase because a stale pre-anchor
                # file disagrees with the file the normal path would have used —
                # that strands results, and it would make salvage stricter than
                # the run it salvages. The divergence is recorded loudly and the
                # newer answer is used.
                print(
                    f"  WARN: {skipped_path} ({len(skipped)} {noun}) disagrees "
                    f"with {chosen_path} ({len(payload)} {noun}) for "
                    f"{model}; using the anchored file."
                )
            # Skipped, not merged and not content-deduplicated.
            print(
                f"  Skipping pre-anchor duplicate {skipped_path} "
                f"({model} is read from {chosen_path})"
            )
        resolved.append((model, chosen_path, payload))
    return resolved


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

    # `subagent_batch_report` is executor bookkeeping (D3i), not a model
    # result: it lives in the phase dir and MUST be excluded or the glob
    # below would parse it as a suggestion list and crash reaggregate.
    # `subagent_dry_run` is the same hazard from --dry-run-subagents (D3l), which
    # writes into THIS dir on a read phase; `write_batch_report`/`write_tasks` are
    # the write scheduler's, which normally land in an apply/implement dir but are
    # listed here so a shared or hand-copied phase dir cannot resurrect the bug.
    exclude_patterns = ['grouped', 'validation', 'salvage', 'revalidation', 'consolidated', 'consolidation', 'report_data', 'subagent_batch_report', 'subagent_dry_run', 'write_batch_report', 'write_tasks']
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

    for model, _chosen_path, suggestions in resolve_result_files_by_model(
        phase_dir, result_files, unsanitize
    ):
        models_found.add(model)
        for s in suggestions:
            s["source_model"] = model
        all_suggestions.extend(suggestions)
        print(f"  Loaded {len(suggestions)} suggestions from {model}")

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
        # A phase dir outlives the run that wrote it, and with prompts embedded a
        # task file is a snapshot: a run started under one skill version and
        # resumed under the other would silently mis-spawn, and a plan edited
        # between materialization and resume leaves prompts that still validate
        # cleanly while reviewing text nobody is looking at any more. Both fail
        # loudly here (exit 3) rather than dispatching anything. Read outside the
        # try below, whose broad handler would otherwise downgrade the refusal to
        # a warning.
        try:
            batch_metadata = read_task_file(
                validation_tasks_path,
                phase_dir=phase_dir,
                current_inputs=build_inputs(
                    plan_file=str(plan_path),
                    grouped_file=grouped_path,
                ),
            )
        except TaskFileError as task_file_error:
            print(f"ERROR: {task_file_error}", file=sys.stderr)
            sys.exit(task_file_error.exit_code)
        try:
            total_batches = batch_metadata.get("total_batches", 0)
            batches_found = 0
            for batch in batch_metadata.get("batches", []):
                batch_file = os.path.join(phase_dir, f"validation_batch_{batch['batch_index']}.json")
                if os.path.exists(batch_file):
                    batches_found += 1

            if batches_found == total_batches and total_batches > 0:
                print(f"All {total_batches} batch files found, merging...")
                validated_groups_for_report = merge_validation_batches(
                    phase_dir, batch_metadata, grouped, validation_path
                )
            elif dispatch_subagent_tasks(
                args,
                task_file=validation_tasks_path,
                phase_dir=phase_dir,
                phase=phase_name,
                document=batch_metadata,
                plan_path=str(plan_path),
                label="validation",
            ):
                # D3b on the reaggregate path: a run resumed with
                # --exec-subagents dispatches the still-incomplete batches itself
                # (completed ones are skipped by their result files, D3i) rather
                # than telling a host to go run subagents it may not have.
                validated_groups_for_report = merge_validation_batches(
                    phase_dir, batch_metadata, grouped, validation_path
                )
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

            # Resolved here, from the running skill dir, because the task file
            # stores only the durable mode + plan pair (D1c).
            reaggregate_cmd = resolved_reaggregate_command(batched_tasks)

            # D3b: with --exec-subagents this process continues past the marker
            # instead of exiting — dispatch the batch, then fall through to the
            # merge below and finish the phase in one invocation. The marker is
            # still printed above (logs and .status.json), but it is
            # informational on this path.
            if dispatch_subagent_tasks(
                args,
                task_file=validation_tasks_path,
                phase_dir=phase_dir,
                phase=phase_name,
                document=batched_tasks,
                plan_path=str(plan_path),
                label="validation",
                retry_command=reaggregate_cmd,
            ):
                validated_groups_for_report = merge_validation_batches(
                    phase_dir, batched_tasks, grouped, validation_path
                )
            else:
                print("\nAfter validation completes, run:")
                print(f"  {reaggregate_cmd or ''}")
                # The single follow-up a native-subagent host is owed, also
                # exposed on the dispatcher's MULTI_LLM_RESULT line so a host
                # that cannot parse prose does not have to regex-mine stdout.
                record_next(reaggregate_cmd or None)
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

    # §9.3.a/b: the anchor capture point. Absolute, taken before anything can
    # chdir and before any provider read that decides routing, so a call site
    # that forgets the keyword still falls back to THIS plan, not to CWD.
    set_run_anchor(plan_path)

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
            # `args` is threaded so the handler can honor --exec-subagents
            # (D3b): with the flag on it dispatches the consolidation
            # batch and reaggregates in this same invocation.
            consolidation_handlers['consolidate'](phase_dir, plan_path, args)
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
            exclude = ['grouped', 'validation', 'salvage', 'revalidation', 'consolidated', 'consolidation', 'report_data', 'subagent_batch_report']
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

    # Handle model selection (D6 row 2: auto-select `defaults.models` when
    # nothing was named; fail closed with exit 3 when nothing resolves).
    unattended = bool(getattr(args, "yes", False))
    try:
        model_specs = resolve_models(
            cli_models=args.models,
            interactive=args.interactive,
            quick=args.quick,
            mode=phase_name,
            anchor=plan_path,  # per-project config discovery follows the plan-derived root
            unattended=unattended,
        )
    except NeedsHuman as e:
        sys.exit(_unattended.fail_closed(e.message, site_id=e.site))
    except RuntimeError as e:
        print(f"ERROR: {e}")
        available = get_all_model_specs(anchor=plan_path)
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
        # Anchored to match the rejection above: an anchored `is_model_valid`
        # paired with a CWD-discovered suggestion list is its own bug (§9.3.c).
        available = get_all_model_specs(anchor=plan_path)
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

    # --- the detach boundary -------------------------------------------------
    # Everything above ran in the foreground on purpose: arguments and flag
    # compatibility, the plan file and prior-phase inputs, provider/model
    # resolution, the completed-phase guard, output-dir creation and the write
    # probe. Detaching before them would bury an ordinary startup failure (a
    # typo'd flag, an unconfigured provider) inside orchestrator-run.log,
    # reachable only by polling `status` and finding `failed` seconds later. So
    # the fork happens here, once nothing knowable is left unchecked — and
    # nothing below this line runs in the invoker.
    outcome = maybe_detach(
        args=args,
        mode=_modes().mode_for_script(orchestrator_script) or phase_name,
        phase_dir=phase_dir,
        plan_file=plan_path,
        phase=phase_name,
        model_specs=model_specs,
    )
    if outcome is not None:
        sys.exit(outcome)

    write_status(phase_dir, {
        "phase": phase_name,
        "state": "models_running",
        "started_at": datetime.now().isoformat(),
        "models_requested": list(model_specs),
        "error": None,
    })
    # D6 row 2, recorded rather than merely applied: an auto-selection nobody can
    # see afterwards is indistinguishable from a silent degradation. Only under
    # `--yes`, where it replaced a question — with `--yes` unset nothing about a
    # run changes.
    if unattended and not args.models and not args.quick:
        _unattended.record_resolution(
            phase_dir,
            _unattended.row(2).site_id,
            "defaults.models: " + ", ".join(model_specs),
        )
    _detach.report_progress(
        phase_dir, step="fanout", models_total=len(model_specs), models_done=0
    )

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

    # §3.6: which model actually served each slot. The per-model `{model}.json`
    # is a bare suggestion list reaggregation re-reads, so the attribution is
    # recorded once here and carried to the report rather than stamped into an
    # artifact whose shape another phase depends on. Empty for a default run.
    served_models = collect_served_models(results)

    # D3b: a salvage task exists for every model whose output would not parse.
    # With --exec-subagents this run repairs them itself; a model whose result
    # file now exists is no longer failed, so the partial-results accounting
    # below sees the post-salvage truth rather than the pre-salvage one.
    if failed_models and exec_subagents_requested(args):
        dispatch_salvage_tasks(
            args, phase_dir=phase_dir, phase=phase_name, plan_path=plan_path
        )
        for model_spec in list(failed_models):
            recovered = os.path.join(
                phase_dir, f"{sanitize_model_name(model_spec)}.json"
            )
            if os.path.exists(recovered) and os.path.getsize(recovered) > 0:
                print(f"[{model_spec}] recovered by salvage: {recovered}")
                del failed_models[model_spec]

    # D6 row 3: partial results are usable output, so a run that lost *some*
    # models resumes-then-reaggregates (never a question, never `--rerun-all`)
    # and exits 0 with `state: "partial"`; a run where *no* model produced a
    # result has nothing to aggregate and is exit 3 naming the retry command.
    completed_models = [m for m in model_specs if m not in failed_models]
    if not completed_models:
        print("\nERROR: All models failed")
        for model_spec, reason in failed_models.items():
            print(f"  - {model_spec}: {reason}")
        sys.exit(_unattended.report_partial_results(
            completed=completed_models,
            failed=list(failed_models),
            phase_dir=phase_dir,
            retry_command=dispatcher_command(
                _modes().mode_for_script(orchestrator_script) or phase_name,
                plan_path,
                "--force",
            ),
        ))
    _unattended.report_partial_results(
        completed=completed_models,
        failed=list(failed_models),
        phase_dir=phase_dir,
        unattended=unattended,
    )

    update_status(phase_dir, {
        "state": "models_complete",
        "models_completed": completed_models,
        "models_failed": failed_models,
        # Added only when something was attributed, so a default run's
        # `.status.json` gains no key.
        **({"models_served": served_models} if served_models else {}),
    })
    _detach.report_progress(
        phase_dir,
        step="grouping",
        models_total=len(model_specs),
        models_done=len(model_specs) - len(failed_models),
    )

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

                # Resolved here, from the running skill dir, because the task
                # file stores only the durable mode + plan pair (D1c).
                reaggregate_cmd = resolved_reaggregate_command(batched_tasks)

                # D3b: self-contained mode continues past the marker in this same
                # process — dispatch, merge, and finish the phase in one
                # invocation. Host-driven mode (the default) is unchanged below.
                if dispatch_subagent_tasks(
                    args,
                    task_file=validation_tasks_path,
                    phase_dir=phase_dir,
                    phase=phase_name,
                    document=batched_tasks,
                    plan_path=str(plan_path),
                    label="validation",
                    retry_command=reaggregate_cmd,
                ):
                    validation_path = os.path.join(phase_dir, "validation.json")
                    validated_groups_for_report = merge_validation_batches(
                        phase_dir, batched_tasks, grouped, validation_path
                    )
                else:
                    print("\nAfter validation completes, run:")
                    print(f"  {reaggregate_cmd or ''}")
                    record_next(reaggregate_cmd or None)

                    # The follow-up command is persisted in its **durable,
                    # rebindable** form (mode word + plan path) — in the task
                    # file above and here alike — never as the
                    # resolved skill-dir-absolute string the same run prints: the
                    # Claude plugin cache is keyed by commit hash, so a stored
                    # absolute command points at a directory that no longer
                    # exists after any plugin update. `status` (and
                    # utils/detach.resolved_reaggregate_command) rebuild it at
                    # read time from the *running* skill dir.
                    update_status(phase_dir, {
                        "state": "validation_pending",
                        "stdout_markers": ["VALIDATION_BATCHES_PENDING"] if total_batches > 1 else ["VALIDATION_PENDING"],
                        "validation_tasks_path": validation_tasks_path,
                        **durable_reaggregate_fields(orchestrator_script, plan_path),
                    })
                    _detach.report_progress(phase_dir, step="validation_pending")

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
        served_models=served_models or None,
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
