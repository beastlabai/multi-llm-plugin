#!/usr/bin/env python3
"""
Shared validation utility for plan review and code review orchestrators.

Validates grouped suggestions/issues using an LLM to filter false positives
and identify items needing human review.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import the invoke_for_json utility
try:
    from .llm_client import invoke_for_json
    from .providers.base import (
        CATEGORY_CONNECTION_ERROR,
        CATEGORY_INVALID_MODEL,
        CATEGORY_PERMISSION_DENIED,
        CATEGORY_SERVER_ERROR,
        classify_streams,
    )
    from .retry_ladder import (
        STATUS_AUTH_FAILED,
        STATUS_CONTEXT_LENGTH_EXCEEDED,
        STATUS_RATE_LIMITED,
        STATUS_RUNNER_ERROR,
        STATUS_SPAWN_FAILED,
        STATUS_TIMEOUT,
    )
    from .state_manager import generate_group_id, CURRENT_FORMAT_VERSION
    from .cli_common import (
        current_replay_flags, durable_reaggregate_fields, retry_command,
    )
    from .subagent_prompts import build_validation_prompt
    from .subagent_tasks import (
        PromptBudget,
        TaskFileError,
        build_inputs,
        file_content_hash,
        make_task,
        make_task_file,
        split_indices,
    )
except ImportError:
    from utils.llm_client import invoke_for_json
    from utils.providers.base import (
        CATEGORY_CONNECTION_ERROR,
        CATEGORY_INVALID_MODEL,
        CATEGORY_PERMISSION_DENIED,
        CATEGORY_SERVER_ERROR,
        classify_streams,
    )
    from utils.retry_ladder import (
        STATUS_AUTH_FAILED,
        STATUS_CONTEXT_LENGTH_EXCEEDED,
        STATUS_RATE_LIMITED,
        STATUS_RUNNER_ERROR,
        STATUS_SPAWN_FAILED,
        STATUS_TIMEOUT,
    )
    from utils.state_manager import generate_group_id, CURRENT_FORMAT_VERSION
    from utils.cli_common import (
        current_replay_flags, durable_reaggregate_fields, retry_command,
    )
    from utils.subagent_prompts import build_validation_prompt
    from utils.subagent_tasks import (
        PromptBudget,
        TaskFileError,
        build_inputs,
        file_content_hash,
        make_task,
        make_task_file,
        split_indices,
    )


#: The shared-context ceiling for a validation prompt. Excerpted **once per
#: phase** and the identical excerpt reused across every batch: per-batch
#: re-truncation would both bloat the file and break prompt determinism.
VALIDATION_CONTEXT_LIMIT = 30000

#: The documented per-file ceiling on a phase's **materialized total** (D2f).
#: Embedding prompts reverses the reference-based design at a stated cost: the
#: one shared excerpt is duplicated into every batch, so the file grows with
#: batch count (D2f's own worked example is ~300–400 KB for ten batches). The
#: lever that shrinks it is a *larger* ``max_per_batch`` — fewer batches means
#: fewer copies of the excerpt — bounded above by the D2d per-task budget.
#: Crossing this ceiling is **reported, not resolved**: D2f's remedy is a
#: downgrade to ``context_mode: workspace-read``, which no generator in this
#: codebase emits (recorded as a deviation in
#: ``plans/host-harness-portability.md``), and failing a phase outright would
#: throw away a completed fan-out over a scratch file being large.
VALIDATION_TASKS_TOTAL_CHARS_CEILING = 2_000_000


def excerpt_context(context: str, limit: int = VALIDATION_CONTEXT_LIMIT) -> str:
    """Return the one shared context excerpt a phase's batches all embed."""
    if len(context) <= limit:
        return context
    return context[:limit] + "\n\n[... truncated ...]"


def materialized_total_chars(tasks: List[Dict[str, Any]]) -> int:
    """Prompt characters a phase's task file will carry — the D2f file total."""
    return sum(len(task.get("prompt", "")) for task in tasks)


# Error type constants - ALWAYS use these constants, never raw strings
ERROR_TYPE_PARSING = "parsing_error"
ERROR_TYPE_TIMEOUT = "timeout"
ERROR_TYPE_RATE_LIMITED = "rate_limited"
ERROR_TYPE_AMBIGUOUS = "real_ambiguity"  # For genuine LLM-determined ambiguity
ERROR_TYPE_MODEL_FAILURE = "model_failure"
ERROR_TYPE_UNKNOWN = "unknown"

# Define which error types are recoverable (can be retried/auto-approved)
RECOVERABLE_ERROR_TYPES = frozenset({
    ERROR_TYPE_PARSING,
    ERROR_TYPE_TIMEOUT,
    ERROR_TYPE_RATE_LIMITED,
})


#: Our *own* extraction failing, which is not a transport category at all — the
#: provider answered, and this codebase could not turn the answer into JSON.
#: Lane B's equivalent is D3k rung 5 (``unparseable_result``), not anything
#: ``classify_failure`` returns, which is why this half deliberately stays local
#: while the four transport categories delegate (§7.1.a).
#:
#: The bare ``"json"`` this list used to open with is gone: it classified *any*
#: error merely **naming** a ``.json`` path — "output file …/plan_review.json was
#: never written" — as a parsing error, hence ``recoverable``, hence worth a
#: whole revalidation dispatch on a condition no re-ask can fix. Every entry now
#: names a parse *event*.
_PARSING_PATTERNS = (
    "json parse",
    "parse json",
    "json decode",
    "jsondecodeerror",
    "invalid json",
    "no valid json",
    "no json found",
    "malformed json",
    "parse error",
    "parsing error",
    "decode error",
    "decodeerror",
    "syntax error",
    "unexpected token",
    "expecting value",
    "expecting property",
)

#: The shared classifier's categories, mapped onto the vocabulary this module
#: **persists** (§7.1.a). The mapping is deliberately lossy in one direction
#: only: several distinct infrastructure verdicts collapse onto
#: ``ERROR_TYPE_MODEL_FAILURE``, which is correctly **not** in
#: :data:`RECOVERABLE_ERROR_TYPES` — none of them is fixed by re-asking.
_CATEGORY_TO_ERROR_TYPE = {
    STATUS_RATE_LIMITED: ERROR_TYPE_RATE_LIMITED,
    STATUS_TIMEOUT: ERROR_TYPE_TIMEOUT,
    CATEGORY_CONNECTION_ERROR: ERROR_TYPE_MODEL_FAILURE,
    CATEGORY_SERVER_ERROR: ERROR_TYPE_MODEL_FAILURE,
    STATUS_AUTH_FAILED: ERROR_TYPE_MODEL_FAILURE,
    STATUS_SPAWN_FAILED: ERROR_TYPE_MODEL_FAILURE,
    CATEGORY_PERMISSION_DENIED: ERROR_TYPE_MODEL_FAILURE,
    CATEGORY_INVALID_MODEL: ERROR_TYPE_MODEL_FAILURE,
    # A broken OS sandbox (D6) is a property of the machine, never of the turn.
    "sandbox_unavailable": ERROR_TYPE_MODEL_FAILURE,
    STATUS_RUNNER_ERROR: ERROR_TYPE_MODEL_FAILURE,
    # A prompt that does not fit is terminal (§7.6(a)) and must never be
    # recorded recoverable: a revalidation pass would resend it unchanged.
    STATUS_CONTEXT_LENGTH_EXCEEDED: ERROR_TYPE_MODEL_FAILURE,
}


def _classify_validation_error(error: str, http_status: Optional[int] = None) -> str:
    """
    Classify a validation error by type.

    Note: This function is for FAILURES during validation (parsing, timeout, etc.).
    For genuine ambiguity where the LLM returns needs-human-decision, use
    ERROR_TYPE_AMBIGUOUS directly in the validation result handler.

    **Signature, return vocabulary and the ``http_status`` fast path are frozen**
    (§7.1.a): callers branch on the ``ERROR_TYPE_*`` constants, and
    ``error_type``/``recoverable`` are written into every persisted group result
    and read downstream by ``filtering.RECOVERABLE_ERROR_TYPES``. What changed is
    the *body*: this used to be a second, complete classifier — five ordered
    substring lists including a bare ``"429"`` with the same boundary flaw Lane A
    had, and a bare ``"json"`` — living three modules away from the one in
    ``providers/base.py`` and disagreeing with it. The four transport categories
    now delegate, and only the parse half stays local, because that classifies
    *our own* extraction failure rather than the provider's transport.

    Args:
        error: The error message string
        http_status: Optional HTTP status code if available (e.g., 429 for rate limit)

    Returns:
        One of the ERROR_TYPE_* constants
    """
    error_lower = (error or "").lower()

    # Check HTTP status codes first (most reliable). It stays *first* and stays
    # local: it is the one input that needs no patterns at all.
    if http_status:
        if http_status == 429:
            return ERROR_TYPE_RATE_LIMITED
        elif http_status in (408, 504):
            return ERROR_TYPE_TIMEOUT
        elif http_status in (500, 502, 503):
            return ERROR_TYPE_MODEL_FAILURE

    if any(pattern in error_lower for pattern in _PARSING_PATTERNS):
        return ERROR_TYPE_PARSING

    # Also classify non-success-but-parsable responses (raw/malformed data) as parsing errors
    if "raw" in error_lower and ("response" in error_lower or "output" in error_lower):
        return ERROR_TYPE_PARSING

    category = classify_streams(error or "", "")
    return _CATEGORY_TO_ERROR_TYPE.get(category, ERROR_TYPE_UNKNOWN)


def _reported_group_index(result: Dict[str, Any]) -> int:
    """The ``group_index`` a verdict echoed, as an int, or ``-1`` when it is not one.

    The echo is advisory — ``group_hash`` is what results are joined on — so an
    echo that is not an index must degrade to "matches nothing", never to a
    ``TypeError`` inside the index-based fallbacks that compare it against real
    indices. The reply contract now shows ``group_index`` as an unfillable
    placeholder (``<the group's group_index value>``) precisely so a model cannot
    copy a literal ``0``; a model that copies the *placeholder* instead sends a
    string, which is exactly the input this function absorbs. A numeric string
    (``"3"``) is a transcription of a real index and is accepted as one.

    Args:
        result: One verdict record as the subagent returned it.

    Returns:
        The echoed index, or ``-1`` for an absent, null or unparseable one.
    """
    raw = result.get("group_index", result.get("index", -1))
    if isinstance(raw, bool):  # bools are ints in Python; an index they are not
        return -1
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw) if raw.is_integer() else -1
    if isinstance(raw, str):
        try:
            return int(raw.strip())
        except ValueError:
            return -1
    return -1


VALIDATION_PROMPT_TEMPLATE = '''You are validating review suggestions/issues for accuracy.

## Context
{context}

## Suggestions to Validate
{suggestions_json}

## Your Task
For each suggestion, determine if it is:
- "valid": The issue is real and should be addressed
- "invalid": False positive, the suggestion is incorrect or not applicable
- "needs-human-decision": Cannot be determined automatically, requires human judgment

## Output Format
Return ONLY a valid JSON array with one entry per suggestion group:
[
  {{
    "group_index": 0,
    "status": "valid" | "invalid" | "needs-human-decision",
    "reason": "Brief explanation for the classification",
    "confidence": 0.0-1.0
  }}
]
'''


async def validate_groups(
    groups: List[Dict[str, Any]],
    context: str,
    model: str = "auto",
    timeout: int = 120
) -> List[Dict[str, Any]]:
    """
    Validate grouped suggestions/issues using an LLM.

    Args:
        groups: List of suggestion groups (from group_similar_suggestions or similar)
                Each group should have 'theme', 'suggestions', etc.
        context: Context for validation (plan content or diff)
        model: Model to use for validation (default: "auto")
        timeout: Timeout in seconds

    Returns:
        List of validation results:
        [
            {
                "group_index": int,
                "status": "valid" | "invalid" | "needs-human-decision",
                "reason": str,
                "confidence": float
            }
        ]
    """
    if not groups:
        return []

    # Pre-compute group_ids for stable matching after reaggregation
    group_ids = []
    for group in groups:
        if hasattr(group, 'to_dict'):
            group_dict = group.to_dict()
        else:
            group_dict = group
        group_ids.append(generate_group_id(group_dict))

    # Prepare suggestions JSON for the prompt
    # Extract key info from each group for validation
    validation_input = []
    for i, group in enumerate(groups):
        # Handle both dict and SuggestionGroup objects
        if hasattr(group, 'to_dict'):
            group_dict = group.to_dict()
        else:
            group_dict = group

        validation_input.append({
            "index": i,
            "group_id": group_ids[i],
            "theme": group_dict.get("theme", "Unknown"),
            "category": group_dict.get("category", "unknown"),
            "models": group_dict.get("models", []),
            "suggestions": [
                {
                    "title": s.get("title", ""),
                    "desc": s.get("desc", ""),
                    "importance": s.get("importance", "medium"),
                    "type": s.get("type", "unknown")
                }
                for s in group_dict.get("suggestions", [])
            ]
        })

    suggestions_json = json.dumps(validation_input, indent=2)

    # Truncate context if too long
    max_context_len = 30000
    if len(context) > max_context_len:
        context = context[:max_context_len] + "\n\n[... truncated ...]"

    prompt = VALIDATION_PROMPT_TEMPLATE.format(
        context=context,
        suggestions_json=suggestions_json
    )

    print(f"[validation] Starting validation of {len(groups)} groups with model '{model}'...")

    result = invoke_for_json(
        prompt=prompt,
        model=model,
        timeout=timeout
    )

    if not result.get("success"):
        error = result.get("error", "Unknown error")
        error_type = _classify_validation_error(error)
        print(f"[validation] Failed: {error} (type: {error_type})")

        # Use validation_failed status for recoverable errors
        status = "validation_failed" if error_type in RECOVERABLE_ERROR_TYPES else "needs-human-decision"

        # Return all as validation_failed or needs-human-decision on failure WITH error type
        return [
            {
                "group_index": i,
                "group_id": group_ids[i],
                "status": status,
                "reason": f"Validation failed: {error}",
                "confidence": 0.0,
                "error_type": error_type,
                "recoverable": error_type in RECOVERABLE_ERROR_TYPES
            }
            for i in range(len(groups))
        ]

    data = result.get("data", [])

    # Handle case where data is wrapped or malformed
    if isinstance(data, dict) and "raw" in data:
        print("[validation] Warning: Could not parse validation response")
        return [
            {
                "group_index": i,
                "group_id": group_ids[i],
                "status": "validation_failed",
                "reason": "Could not parse validation response",
                "confidence": 0.0,
                "error_type": ERROR_TYPE_PARSING,
                "recoverable": True
            }
            for i in range(len(groups))
        ]

    if not isinstance(data, list):
        print("[validation] Warning: Unexpected response format")
        return [
            {
                "group_index": i,
                "group_id": group_ids[i],
                "status": "validation_failed",
                "reason": "Unexpected validation response format",
                "confidence": 0.0,
                "error_type": ERROR_TYPE_PARSING,
                "recoverable": True
            }
            for i in range(len(groups))
        ]

    # Validate and normalize results
    validation_results = []
    valid_statuses = {"valid", "invalid", "needs-human-decision"}

    for item in data:
        # Coerced, not read raw: an echo that is not an index (a copied reply
        # placeholder, a null) must not reach the `0 <= group_index` comparison
        # below or the sort at the end of this function as a string.
        group_index = _reported_group_index(item)
        status = item.get("status", "needs-human-decision")
        reason = item.get("reason", "No reason provided")
        # `.get(..., 0.5)`'s default only fires when the key is absent; an LLM
        # that replies with an explicit `"confidence": null` would otherwise
        # reach the clamp below as None and crash `float(None)`.
        confidence = item.get("confidence", 0.5)
        if confidence is None:
            confidence = 0.5

        # Normalize status
        if status not in valid_statuses:
            status = "needs-human-decision"

        # Clamp confidence
        confidence = max(0.0, min(1.0, float(confidence)))

        result_entry = {
            "group_index": group_index,
            "status": status,
            "reason": reason,
            "confidence": confidence
        }
        # Add group_id - prefer from LLM response, else compute from index
        if item.get("group_id"):
            result_entry["group_id"] = item["group_id"]
        elif 0 <= group_index < len(group_ids):
            result_entry["group_id"] = group_ids[group_index]
        validation_results.append(result_entry)

    # Fill in any missing groups
    validated_indices = {r["group_index"] for r in validation_results}
    for i in range(len(groups)):
        if i not in validated_indices:
            validation_results.append({
                "group_index": i,
                "group_id": group_ids[i],
                "status": "needs-human-decision",
                "reason": "Not included in validation response",
                "confidence": 0.0
            })

    # Sort by group index
    validation_results.sort(key=lambda x: x["group_index"])

    # Summary
    valid_count = sum(1 for r in validation_results if r["status"] == "valid")
    invalid_count = sum(1 for r in validation_results if r["status"] == "invalid")
    needs_human = sum(1 for r in validation_results if r["status"] == "needs-human-decision")
    validation_failed = sum(1 for r in validation_results if r["status"] == "validation_failed")
    summary_parts = [f"{valid_count} valid", f"{invalid_count} invalid", f"{needs_human} needs-human-decision"]
    if validation_failed > 0:
        summary_parts.append(f"{validation_failed} validation_failed")
    print(f"[validation] Complete: {', '.join(summary_parts)}")

    return validation_results


def apply_validation_to_groups(
    groups: List[Dict[str, Any]],
    validation_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Apply validation results to groups by adding validation status.

    Uses group_id (stable content hash) for matching when available. This
    ensures validation results remain correctly associated with groups even
    after reaggregation changes the sort order.

    The ``group_index`` fallback exists for *hash-less* records only — the
    pre-hash ``validation.json`` format, where the index was the identity. A
    hashed record that matches no group is a verdict about a group that no
    longer exists (the grouping changed underneath it, D7); handing it to
    whichever group now sits at its index attributes another group's status,
    reason and confidence to it, so such a group is left at the default
    "No validation result" and the count of those groups is printed.

    Args:
        groups: Original groups (list of dicts or SuggestionGroup objects)
        validation_results: Results from validate_groups()

    Returns:
        Groups with validation_status and validation_reason added
    """
    # Build lookups - group_id for hashed records, group_index for the rest
    validation_by_id = {
        r.get("group_id"): r
        for r in validation_results
        if r.get("group_id")
    }
    validation_by_index = {
        r["group_index"]: r
        for r in validation_results
        if not r.get("group_id")
    }

    default_validation = {
        "status": "needs-human-decision",
        "reason": "No validation result",
        "confidence": 0.0
    }

    updated_groups = []
    unmatched = 0
    for i, group in enumerate(groups):
        # Handle both dict and SuggestionGroup objects
        if hasattr(group, 'to_dict'):
            group_dict = group.to_dict()
        else:
            group_dict = dict(group)

        # Compute group_id for matching
        gid = generate_group_id(group_dict)

        # group_id first; the index only reaches records that carry no hash
        validation = (
            validation_by_id.get(gid) or
            validation_by_index.get(i)
        )
        if validation is None:
            validation = default_validation
            unmatched += 1

        group_dict["validation_status"] = validation.get("status", "needs-human-decision")
        group_dict["validation_reason"] = validation.get("reason", "")
        # `default_validation` above sets an explicit 0.0 for "no result at all"
        # (a real, meaningful zero); omitting the `.get()` default here only
        # changes the other case — a real record that omitted `confidence` —
        # which now comes through as None instead of a manufactured 0.0.
        group_dict["validation_confidence"] = validation.get("confidence")
        # Copy error_type and recoverable fields if present
        if "error_type" in validation:
            group_dict["validation_error_type"] = validation.get("error_type")
        if "recoverable" in validation:
            group_dict["validation_recoverable"] = validation.get("recoverable")

        updated_groups.append(group_dict)

    if unmatched:
        print(
            f"[validation] {unmatched} of {len(updated_groups)} groups have no "
            f"validation result (left at needs-human-decision)"
        )

    return updated_groups


def stale_validation_reason(
    validation_path: Path,
    grouped_path: Path,
) -> Optional[str]:
    """Why a ``validation.json`` must not be trusted over the batch files, or ``None``.

    A ``validation.json`` belongs to the grouping it was produced for. Since D7
    it records that grouping as ``metadata.grouped_sha256`` — the same value
    ``validation_tasks.json`` binds under ``inputs`` — and is honoured only when
    it matches the ``grouped.json`` sitting beside it now. A file without the
    key is legacy: its grouping is unknowable, so it is honoured only if no
    ``validation_batch_*.json`` in the same dir is newer than it (mtime), because
    a batch written after it is evidence of a validation round it did not come
    from.

    Returns:
        ``None`` when the file may be used, else a one-line reason for the log.
    """
    try:
        with open(validation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        return f"could not read {validation_path.name}: {exc}"

    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    bound_hash = metadata.get("grouped_sha256")
    if bound_hash:
        current_hash = file_content_hash(str(grouped_path))
        if bound_hash == current_hash:
            return None
        return (
            f"{validation_path.name} belongs to grouping {bound_hash}, current "
            f"{grouped_path.name} is {current_hash or 'missing'}"
        )

    # Legacy file: no binding to check, so fall back to what the phase dir says
    # happened after it was written.
    try:
        validation_mtime = validation_path.stat().st_mtime
    except OSError as exc:
        return f"could not stat {validation_path.name}: {exc}"
    newer = sorted(
        batch.name
        for batch in validation_path.parent.glob("validation_batch_*.json")
        if batch.stat().st_mtime > validation_mtime
    )
    if newer:
        return (
            f"{validation_path.name} carries no grouped_sha256 and "
            f"{len(newer)} batch file(s) are newer than it "
            f"({', '.join(newer[:3])}{', ...' if len(newer) > 3 else ''})"
        )
    return None


def _derive_model_from_results(validation_results: List[Dict[str, Any]]) -> Optional[str]:
    """Best-effort ``metadata.model`` for a caller that passed none.

    ``merge_batched_validation_results`` tags every merged result with the
    ``_source_model`` its own batch file's ``metadata.model`` reported (every
    materialized validation batch carries one — the model that answered it).
    A single value wins outright; two or more distinct values join as a
    sorted, comma-separated list rather than silently picking one, since the
    merge output is honestly mixed-provenance when batches ran on different
    models. Returns ``None`` when no result carries the tag at all (a legacy
    caller, or results that never went through the batched merge).
    """
    models = {r["_source_model"] for r in validation_results if r.get("_source_model")}
    if not models:
        return None
    # A subagent reports its own model name free-hand, and the same runner
    # answers as both ``nemotron-3-ultra-free`` and
    # ``opencode/nemotron-3-ultra-free`` across batches. A bare name that is
    # the model half of a prefixed sibling is that sibling, not a second model.
    bare = {m for m in models if "/" not in m}
    for prefixed in {m for m in models if "/" in m}:
        bare.discard(prefixed.rsplit("/", 1)[1])
    models = bare | {m for m in models if "/" in m}
    return ",".join(sorted(models))


def save_validation_results(
    validation_results: List[Dict[str, Any]],
    output_path: Path,
    model: Optional[str] = None,
    groups: Optional[List[Dict[str, Any]]] = None,
    grouped_file: Optional[str] = None,
    runner: Optional[str] = None,
) -> None:
    """
    Save validation results to a JSON file with metadata.

    When groups are provided, computes and stores group_id for each result
    to enable stable matching after reaggregation.

    ``metadata.grouped_sha256`` binds the file to the ``grouped.json`` it was
    produced for (D7) — the same ``file_content_hash`` value
    ``validation_tasks.json`` records under ``inputs.grouped_sha256`` — so
    ``reaggregate_from_existing_files`` can tell a file left behind by an
    earlier grouping from one that belongs to the current one. Defaults to the
    ``grouped.json`` beside ``output_path``; the key is omitted when that file
    does not exist.

    Args:
        validation_results: List of validation results
        output_path: Path to save the JSON file
        model: Model used for validation. When omitted (``None``), derived
            from the batch files via :func:`_derive_model_from_results`
            (``_source_model``, stamped by
            :func:`merge_batched_validation_results`) and falls back to
            ``"unknown"`` only when no batch tagged a model at all — several
            call sites merge then save without ever naming a model, which used
            to write the literal string ``"unknown"`` even though every batch
            file on disk carried its own ``metadata.model``.
        groups: Optional list of groups to compute group_ids from
        grouped_file: The ``grouped.json`` this validation belongs to
        runner: The executor's own dispatch record for this batch (F3) —
            e.g. ``opencode:openrouter/~deepseek/deepseek-flash-latest``, read
            back from ``subagent_batch_report.json`` rather than trusted from
            a subagent's free-hand self-report. Stored verbatim under
            ``metadata.runner`` when the caller has it (an ``--exec-subagents``
            dispatch); omitted otherwise — the host-native (Claude Code Task
            tool) path has no executor record to attribute. Does not change
            how ``model`` is derived; a caller that wants the authoritative
            spec to also win ``metadata.model`` passes it as both.
    """
    if not model:
        model = _derive_model_from_results(validation_results) or "unknown"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if grouped_file is None:
        grouped_file = str(output_path.parent / "grouped.json")
    grouped_sha256 = file_content_hash(grouped_file)

    # Build a mapping from index to group_id if groups provided
    group_id_by_index = {}
    if groups:
        for i, group in enumerate(groups):
            if hasattr(group, 'to_dict'):
                group_dict = group.to_dict()
            else:
                group_dict = group
            group_id_by_index[i] = generate_group_id(group_dict)

    # Build validation data with metadata
    saved_groups = []
    for v in validation_results:
        group_index = v["group_index"]
        group_entry = {
            "group_index": group_index,
            "status": v["status"],
            "reason": v.get("reason", ""),
            # Absent, not 0.0: a batch that omitted `confidence` (a subagent
            # replied without one) is genuinely unknown, not a real zero. Only
            # `v.get("confidence")` preserves an explicit 0.0 the subagent did
            # report while letting a missing key come through as None.
            "confidence": v.get("confidence"),
            "error_type": v.get("error_type", ERROR_TYPE_UNKNOWN),
            "recoverable": v.get("recoverable", False),
            "revalidated": v.get("revalidated", False),
        }
        # Add group_id - compute from groups if available, else preserve existing
        if group_index in group_id_by_index:
            group_entry["group_id"] = group_id_by_index[group_index]
        elif v.get("group_id"):
            group_entry["group_id"] = v["group_id"]
        saved_groups.append(group_entry)

    validation_data = {
        "format_version": CURRENT_FORMAT_VERSION,
        "groups": saved_groups,
        "metadata": {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "schema_version": "2.1",
            **({"runner": runner} if runner else {}),
            **({"grouped_sha256": grouped_sha256} if grouped_sha256 else {}),
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(validation_data, f, indent=2)
    print(f"[validation] Saved validation results to: {output_path}")


def build_validation_subagent_prompt(
    context: str,
    suggestions_json: str,
    output_path: str
) -> str:
    """
    Build the prompt for the validation subagent.

    Args:
        context: The plan content or diff context for validation
        suggestions_json: JSON string of suggestions to validate
        output_path: Path where the subagent should write results

    Returns:
        Full prompt string for the validation subagent
    """
    return f'''You are validating review suggestions for accuracy.

## Context (Plan Content)
{context}

## Suggestions to Validate
{suggestions_json}

## Task
For each suggestion group (indexed 0 to N-1), determine:
- "valid": Issue is real, should be addressed
- "invalid": False positive, not applicable
- "needs-human-decision": Requires human judgment

## Output
Write a JSON file to: {output_path}

Format:
{{
  "groups": [
    {{"group_index": 0, "group_hash": "<copy from group's group_hash field>", "status": "valid|invalid|needs-human-decision", "reason": "...", "confidence": 0.0-1.0}}
  ],
  "metadata": {{"model": "claude", "timestamp": "ISO", "schema_version": "2.1"}}
}}

IMPORTANT:
1. Read each suggestion group carefully
2. Compare against the plan context to verify accuracy
3. Be conservative - if unsure, use "needs-human-decision"
4. Write ONLY valid JSON to the output file
5. Include a reason for each classification
6. Copy the exact group_hash value from each input group to the corresponding output result
'''


def prepare_validation_task(
    groups: List[Dict[str, Any]],
    context: str,
    output_path: str,
    model: str = "auto"
) -> Dict[str, Any]:
    """
    Prepare validation input for Claude Code subagent.

    Instead of running validation internally, this prepares all the data
    needed for Claude Code to spawn a validation subagent.

    Args:
        groups: List of suggestion groups to validate
        context: Context for validation (plan content or diff)
        output_path: Path where subagent should write validation.json
        model: Model hint (not used by Claude Code, but preserved for reference)

    Returns:
        Dict with:
        - prompt: Full validation prompt for subagent
        - output_path: Where subagent should write results
        - groups_count: Number of groups to validate
        - suggestions_json: The JSON representation of suggestions
    """
    # Prepare suggestions JSON for the prompt
    validation_input = []
    for i, group in enumerate(groups):
        # Handle both dict and SuggestionGroup objects
        if hasattr(group, 'to_dict'):
            group_dict = group.to_dict()
        else:
            group_dict = group

        # Compute group_id for stable matching after reaggregation
        gid = generate_group_id(group_dict)

        validation_input.append({
            "index": i,
            "group_id": gid,
            "theme": group_dict.get("theme", "Unknown"),
            "category": group_dict.get("category", "unknown"),
            "models": group_dict.get("models", []),
            "suggestions": [
                {
                    "title": s.get("title", ""),
                    "desc": s.get("desc", ""),
                    "importance": s.get("importance", "medium"),
                    "type": s.get("type", "unknown")
                }
                for s in group_dict.get("suggestions", [])
            ]
        })

    suggestions_json = json.dumps(validation_input, indent=2)

    # Truncate context if too long
    max_context_len = 30000
    if len(context) > max_context_len:
        context = context[:max_context_len] + "\n\n[... truncated ...]"

    prompt = build_validation_subagent_prompt(
        context=context,
        suggestions_json=suggestions_json,
        output_path=output_path
    )

    return {
        "prompt": prompt,
        "output_path": output_path,
        "groups_count": len(groups),
        "suggestions_json": suggestions_json,
        "model_hint": model,
    }


def prepare_revalidation_task(
    groups: List[Dict[str, Any]],
    validation_results: List[Dict[str, Any]],
    context: str,
    output_path: str,
    include_all_human: bool = False,
    model: str = "auto"
) -> Dict[str, Any]:
    """
    Prepare revalidation input for Claude Code subagent.

    Similar to prepare_validation_task, but only includes items that need
    revalidation (validation_failed status, optionally needs-human-decision).

    Args:
        groups: Original grouped suggestions/issues
        validation_results: Previous validation results
        context: Context for validation (plan content or diff)
        output_path: Path where subagent should write results
        include_all_human: If True, also revalidate needs-human-decision items
        model: Model hint for reference

    Returns:
        Dict with:
        - prompt: Full validation prompt for subagent (or None if nothing to revalidate)
        - output_path: Where subagent should write results
        - items_to_revalidate: Count of items needing revalidation
        - item_indices: List of original indices being revalidated
        - original_validation: The original validation results (for merging later)
    """
    # Identify items needing revalidation
    items_to_revalidate = []
    item_indices = []

    for i, val in enumerate(validation_results):
        status = val.get("status", "needs-human-decision")
        error_type = val.get("error_type", ERROR_TYPE_UNKNOWN)

        should_revalidate = (
            status == "validation_failed" or
            (include_all_human and status == "needs-human-decision" and
             error_type != ERROR_TYPE_AMBIGUOUS)
        )

        if should_revalidate and i < len(groups):
            items_to_revalidate.append(groups[i])
            item_indices.append(i)

    if not items_to_revalidate:
        return {
            "prompt": None,
            "output_path": output_path,
            "items_to_revalidate": 0,
            "item_indices": [],
            "original_validation": validation_results,
            "model_hint": model,
        }

    # Prepare the validation task for the subset
    validation_task = prepare_validation_task(
        groups=items_to_revalidate,
        context=context,
        output_path=output_path,
        model=model
    )

    return {
        "prompt": validation_task["prompt"],
        "output_path": output_path,
        "items_to_revalidate": len(items_to_revalidate),
        "item_indices": item_indices,
        "original_validation": validation_results,
        "suggestions_json": validation_task["suggestions_json"],
        "model_hint": model,
    }


def load_validation_results(validation_path: Path) -> List[Dict[str, Any]]:
    """
    Load validation results with migration for older format files.

    Older validation.json files may not have error_type, recoverable, or
    revalidated fields. This function applies sensible defaults to ensure
    compatibility.

    Args:
        validation_path: Path to the validation.json file

    Returns:
        List of validation results with all fields populated
    """
    with open(validation_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle both old format (list) and new format (dict with groups/metadata)
    if isinstance(data, list):
        # Old format: direct list of validation results
        groups = data
        schema_version = "1.0"
    else:
        # New format: dict with groups and metadata
        schema_version = data.get("metadata", {}).get("schema_version", "1.0")
        groups = data.get("groups", [])

    if schema_version < "2.0":
        # Migrate v1 format to v2
        for group in groups:
            # Default error_type based on status
            if "error_type" not in group:
                if group.get("status") == "needs-human-decision":
                    # Assume older needs-human-decision items were from parsing errors
                    # unless the reason suggests genuine ambiguity
                    reason = group.get("reason", "").lower()
                    if any(word in reason for word in ["ambiguous", "unclear", "judgment"]):
                        group["error_type"] = ERROR_TYPE_AMBIGUOUS
                    else:
                        group["error_type"] = ERROR_TYPE_PARSING
                elif group.get("status") == "validation_failed":
                    group["error_type"] = ERROR_TYPE_PARSING
                else:
                    group["error_type"] = ERROR_TYPE_UNKNOWN

            # Default recoverable based on error_type
            if "recoverable" not in group:
                group["recoverable"] = group.get("error_type") in RECOVERABLE_ERROR_TYPES

            # Default revalidated to False
            if "revalidated" not in group:
                group["revalidated"] = False

        print(f"[validation] Migrated {len(groups)} items from schema v1 to v2", file=sys.stderr)

    return groups


async def revalidate_failed_items(
    groups: List[Dict[str, Any]],
    validation_results: List[Dict[str, Any]],
    context: str,
    model: str = "auto",
    include_all_human: bool = False,
    timeout: int = 180
) -> List[Dict[str, Any]]:
    """
    Re-run validation only on items that previously failed.

    Args:
        groups: Original grouped suggestions/issues
        validation_results: Previous validation results
        context: Context for validation (plan content or diff)
        model: Model to use for revalidation
        include_all_human: If True, also revalidate needs-human-decision items
        timeout: Timeout in seconds (default higher for retry)

    Returns:
        Updated validation results
    """
    # Identify items needing revalidation
    items_to_revalidate = []
    item_indices = []

    for i, val in enumerate(validation_results):
        status = val.get("status", "needs-human-decision")
        error_type = val.get("error_type", ERROR_TYPE_UNKNOWN)

        should_revalidate = (
            status == "validation_failed" or
            (include_all_human and status == "needs-human-decision" and
             error_type != ERROR_TYPE_AMBIGUOUS)
        )

        if should_revalidate and i < len(groups):
            items_to_revalidate.append(groups[i])
            item_indices.append(i)

    if not items_to_revalidate:
        print("[revalidate] No items need revalidation")
        return validation_results

    print(f"[revalidate] Re-validating {len(items_to_revalidate)} items with model '{model}'...")

    # Run validation on subset with error handling
    try:
        new_results = await validate_groups(
            groups=items_to_revalidate,
            context=context,
            model=model,
            timeout=timeout
        )
    except Exception as e:
        # Revalidation itself failed - apply fallback strategy
        print(f"[revalidate] ERROR: Revalidation failed: {e}", file=sys.stderr)
        error_type = _classify_validation_error(str(e))

        # Fallback strategies based on error type
        if error_type == ERROR_TYPE_RATE_LIMITED:
            print("[revalidate] Fallback: Rate limited. Try again later or use --revalidate-model with a different provider.", file=sys.stderr)
        elif error_type == ERROR_TYPE_TIMEOUT:
            print("[revalidate] Fallback: Timeout. Try --revalidate-model with a faster model.", file=sys.stderr)
        else:
            print(f"[revalidate] Fallback: Keeping original validation status for {len(items_to_revalidate)} items.", file=sys.stderr)

        # Return original results unchanged on failure
        return validation_results

    # Check if revalidation produced more failures
    revalidation_failures = sum(1 for r in new_results if r.get("status") == "validation_failed")
    if revalidation_failures > 0:
        print(f"[revalidate] Warning: {revalidation_failures} items still failed validation after retry.", file=sys.stderr)
        print("[revalidate] Consider using --approve-validation-failed or trying a different model.", file=sys.stderr)

    # Merge results back
    updated_results = list(validation_results)
    for j, original_idx in enumerate(item_indices):
        new_result = new_results[j] if j < len(new_results) else validation_results[original_idx]
        new_result["group_index"] = original_idx
        new_result["revalidated"] = True
        new_result["revalidation_model"] = model
        updated_results[original_idx] = new_result

    # Summary of changes
    improved = sum(
        1 for j, idx in enumerate(item_indices)
        if j < len(new_results) and new_results[j].get("status") == "valid"
    )
    print(f"[revalidate] Complete: {improved} items now valid after revalidation")

    return updated_results


# --- Batched Validation Functions ---

try:
    from .validation_batcher import (
        batch_validation_groups,
        estimate_validation_batching_stats,
        ValidationBatch,
    )
except ImportError:
    from utils.validation_batcher import (
        batch_validation_groups,
        estimate_validation_batching_stats,
        ValidationBatch,
    )


def build_validation_batch_records(
    groups: List[Dict[str, Any]],
    group_indices: List[str],
    group_ids: List[str],
) -> List[Dict[str, Any]]:
    """Build the suggestion records one validation batch embeds in its prompt.

    The records travel as *values* inside the prompt (``context_mode: inline``),
    never as a path to ``grouped.json``, because a headless runner is not
    guaranteed a workspace-read tool.

    Args:
        groups: All groups for the phase.
        group_indices: This batch's indices into ``groups``.
        group_ids: ``group_ids[i]`` is the stable id of ``groups[i]``.

    Returns:
        One record per group in the batch, carrying the global index under
        ``group_index`` — the same key the reply contract asks the subagent to
        echo back, rather than a differently-named ``index`` the subagent has
        to translate — and the ``group_hash`` the subagent must also echo.
    """
    records: List[Dict[str, Any]] = []
    for idx in group_indices:
        group = groups[idx]
        group_dict = group.to_dict() if hasattr(group, 'to_dict') else group
        records.append({
            "group_index": idx,
            # Both spellings carry the same value: the prompt tells the subagent
            # to copy `group_hash`, while `merge_batched_validation_results`
            # accepts either, so no reader can miss it.
            "group_hash": group_ids[idx],
            "group_id": group_ids[idx],
            "theme": group_dict.get("theme", "Unknown"),
            "category": group_dict.get("category", "unknown"),
            "models": group_dict.get("models", []),
            "suggestions": [
                {
                    "title": s.get("title", ""),
                    "desc": s.get("desc", ""),
                    "importance": s.get("importance", "medium"),
                    "type": s.get("type", "unknown"),
                }
                for s in group_dict.get("suggestions", [])
            ],
        })
    return records


def resolved_reaggregate_command(document: Dict[str, Any]) -> Optional[str]:
    """Rebuild a task document's follow-up command from the *running* skill dir.

    ``validation_tasks.json`` stores the durable, rebindable pair
    (``reaggregate_mode`` + ``plan_file``) and never the resolved
    skill-dir-absolute command (D1c). Resolution therefore happens here, at
    print time, exactly as :func:`utils.detach.resolved_reaggregate_command`
    does for ``.status.json``: printing an absolute path is safe because it is
    session-scoped, persisting one is not.

    Returns ``None`` when the document carries neither the durable pair nor a
    legacy resolved command (an empty phase, which owes no follow-up).
    """
    mode = document.get("reaggregate_mode")
    plan = document.get("plan_file")
    if mode and plan:
        # The live invocation of this mode wins over the document's stored
        # `replay_flags` (D2's precedence rule); the stored list is the fallback
        # for a reader in a later process, and is never sanitized.
        return retry_command(
            str(mode), str(plan), "--reaggregate",
            flags=current_replay_flags(
                document.get("replay_flags"), mode=str(mode)
            ),
        )
    # Written before the durable pair existed. Its stored command is only as
    # good as the skill dir it was written under — precisely the staleness this
    # key stopped persisting — but honouring it is never worse than the build
    # that wrote it, and it keeps an in-flight phase resumable across the change.
    legacy = document.get("reaggregate_command")
    return str(legacy) if legacy else None


def prepare_batched_validation_tasks(
    groups: List[Dict[str, Any]],
    context: str,
    output_dir: str,
    plan_file: str = "",
    model: str = "auto",
    max_per_batch: int = 4,
    orchestrator: str = "review_plan_orchestrator.py",
    base_ref: str = "",
    tasks_file: str = "",
    kind: str = "validation",
    context_label: str = "Plan content",
    secondary_context: str = "",
    secondary_context_label: str = "",
    extra_instructions: str = "",
    prompt_budget: Optional[PromptBudget] = None,
) -> Dict[str, Any]:
    """
    Prepare batched validation tasks for sequential subagent processing.

    Instead of creating one task with all groups, this creates multiple batch
    tasks that can be processed sequentially by subagents with fresh context.

    Each batch carries a **complete prompt** (D2/D2f). This deliberately
    reverses this function's original reference-based design, in which the file
    stored only references to ``grouped.json`` and the plan and a Claude reading
    the instruction markdown assembled the prompt itself. Prompt-in-JSON is what
    makes one task file executable by a native-subagent host *and* by the Python
    executor with byte-identical prompt text; resolving the references at
    dispatch time would only fix the executor path, since a native host has no
    Python at dispatch time. The reference-based approach survives as
    ``context_mode: workspace-read`` — declared in the task and capability-gated,
    instead of implicit in who reads the file.

    The shared context is excerpted **once** and the identical excerpt reused
    across batches: per-batch re-truncation would break prompt determinism.

    Args:
        groups: List of suggestion groups to validate
        context: Context for validation (plan content or diff); excerpted once
            and embedded in every batch prompt
        output_dir: Directory where subagents should write batch results
        plan_file: Path to the plan file (provenance and the ``inputs`` hash;
            never interpolated into prompt text)
        model: Model hint, preserved verbatim (free-form model string or ``auto``)
        max_per_batch: Maximum groups per batch for normal items
        orchestrator: Name of the orchestrator script for reaggregate command
        base_ref: Git ref for code diff context (empty string for plan reviews);
            recorded as ``inputs.diff_base``
        tasks_file: Path to the tasks file for review-tasks validation context;
            recorded as provenance *and* hashed into ``inputs.tasks_sha256``
        kind: ``"validation"`` or ``"revalidation"`` — the task ``kind``
        context_label: What the embedded context is (``"Plan content"``, …)
        secondary_context: An optional second document to embed beside
            ``context`` (review-tasks: the task list its suggestions are about).
            Excerpted **separately**, against the same per-context limit, so a
            long plan cannot squeeze the artifact under review out of the prompt
        secondary_context_label: What that second document is; required with it
        extra_instructions: Optional capability-neutral phase guidance appended
            to every batch prompt (e.g. the code-review issue-location guidance)
        prompt_budget: Optional D2d budget. Over-budget batches are split; a
            single group that still does not fit fails loudly rather than being
            truncated.

    Returns:
        Dict with:
        - tasks / batches: the materialized tasks (same objects under both keys)
        - tasks_schema_version, kind, inputs: the D2g file envelope
        - total_batches: Number of batches
        - batching_stats: Statistics about the batching
        - grouped_file, plan_file, phase_dir, base_ref
        - reaggregate_mode: the durable half of the follow-up command; the
          resolved string is rebuilt at print time by
          :func:`resolved_reaggregate_command`, never persisted (D1c)
    """
    grouped_file = str(Path(output_dir) / "grouped.json")
    inputs = build_inputs(
        plan_file=plan_file,
        grouped_file=grouped_file,
        tasks_file=tasks_file,
        diff_base=base_ref,
    )

    if not groups:
        empty = make_task_file(kind=kind, tasks=[], inputs=inputs, task_list_key="batches", extra={
            "total_batches": 0,
            "batching_stats": estimate_validation_batching_stats([]),
            "grouped_file": grouped_file,
            "plan_file": plan_file,
            "phase_dir": output_dir,
            "model_hint": model,
            "base_ref": base_ref,
        })
        return empty

    # Create batches
    validation_batches = batch_validation_groups(
        groups=groups,
        max_per_batch=max_per_batch,
        isolate_high=True
    )

    # Pre-compute group_ids for all groups
    group_ids = []
    for group in groups:
        if hasattr(group, 'to_dict'):
            group_dict = group.to_dict()
        else:
            group_dict = group
        group_ids.append(generate_group_id(group_dict))

    # The one shared excerpt, reused verbatim by every batch below. A second
    # document is excerpted on its own budget rather than appended to the first:
    # concatenating would let a plan at the limit truncate the artifact under
    # review down to nothing, which is the failure two contexts exist to fix.
    shared_context = excerpt_context(context)
    shared_secondary = excerpt_context(secondary_context) if secondary_context else ""
    budget = prompt_budget or PromptBudget(None)

    # Work list of (group_indices, is_high_priority). A batch whose materialized
    # prompt exceeds the D2d budget is split in half and re-queued; a single
    # group that still does not fit raises (never truncated).
    pending: List[Tuple[List[int], bool]] = [
        (list(batch.group_indices), batch.is_high_priority)
        for batch in validation_batches
    ]

    batch_tasks: List[Dict[str, Any]] = []
    next_index = 0
    while pending:
        indices, is_high = pending.pop(0)
        records = build_validation_batch_records(groups, indices, group_ids)
        prompt = build_validation_prompt(
            plan_excerpt=shared_context,
            suggestion_records=records,
            kind=kind,
            context_label=context_label,
            secondary_excerpt=shared_secondary,
            secondary_label=secondary_context_label,
            extra_instructions=extra_instructions,
        )
        if not budget.fits(prompt):
            first, second = split_indices(indices)
            if second:
                pending.insert(0, (second, is_high))
                pending.insert(0, (first, is_high))
                continue
            # Unsplittable and over budget: fail loudly with the actionable fix.
            # `resolution` raises for a runner with no workspace-read capability,
            # but *returns* "workspace-read" for one that has it — and this kind
            # cannot take that downgrade: its tasks pin `context_mode: inline`
            # (D2c), so materializing one any other way is schema-invalid.
            # Acting on the return value is what keeps that from falling through
            # into a silently over-budget inline prompt.
            resolution = budget.resolution(prompt, splittable=False)
            raise TaskFileError(
                f"A {kind} batch holding a single group materializes a prompt of "
                f"{budget.units(prompt)} UTF-16 code units against the runner's "
                f"{budget.max_units}-unit {budget.transport} budget, so it can be "
                f"neither split nor sent. The '{resolution}' resolution is not "
                f"available to this kind — a materialized {kind} task pins "
                f"context_mode: inline (D2c) and the schema rejects any other "
                f"value — so configure a runner with a stdin/file prompt "
                f"transport (driver.subagent_runner). The prompt is never "
                f"truncated."
            )

        output_name = f"validation_batch_{next_index}.json"
        batch_tasks.append(make_task(
            kind=kind,
            task_id=f"{kind}-batch-{next_index}",
            prompt=prompt,
            output_name=output_name,
            inputs=inputs,
            phase_dir=output_dir,
            model_hint=model,
            extra={
                "batch_index": next_index,
                "group_indices": indices,
                "group_ids": [group_ids[idx] for idx in indices],
                "groups_count": len(indices),
                "is_high_priority": is_high,
                "plan_file": plan_file,
                "grouped_file": grouped_file,
            },
        ))
        next_index += 1

    stats = estimate_validation_batching_stats(groups, max_per_batch)

    # D2f's per-file ceiling on the materialized total. Reported rather than
    # enforced — see VALIDATION_TASKS_TOTAL_CHARS_CEILING for why a phase that
    # crosses it is still written.
    total_chars = materialized_total_chars(batch_tasks)
    if total_chars > VALIDATION_TASKS_TOTAL_CHARS_CEILING:
        print(
            f"[VALIDATION_TASKS_OVERSIZE] {kind}: {len(batch_tasks)} batches "
            f"materialize {total_chars} prompt chars, over the "
            f"{VALIDATION_TASKS_TOTAL_CHARS_CEILING}-char per-file ceiling "
            f"(D2f). The shared context excerpt is embedded once per batch, so "
            f"raise max_per_batch (currently {max_per_batch}) to cut the number "
            f"of copies, or validate fewer groups. Writing it anyway.",
            file=sys.stderr,
        )

    # The follow-up command is stored in its **durable, rebindable** form (mode
    # word + plan path), never as the resolved skill-dir-absolute string: this
    # document outlives the run that wrote it, and the Claude plugin cache is
    # keyed by commit hash, so a persisted absolute command names a directory
    # the next plugin update deletes (D1c). `resolved_reaggregate_command()`
    # rebuilds it at print time from the *running* skill dir — the same split
    # `.status.json` already makes via `cli_common.durable_reaggregate_fields`.
    extra: Dict[str, Any] = {
        "total_batches": len(batch_tasks),
        "batching_stats": stats,
        "grouped_file": grouped_file,
        "phase_dir": output_dir,
        **durable_reaggregate_fields(orchestrator, plan_file),
        "model_hint": model,
        "base_ref": base_ref,
    }
    if tasks_file:
        extra["tasks_file"] = tasks_file
    return make_task_file(
        kind=kind,
        tasks=batch_tasks,
        inputs=inputs,
        task_list_key="batches",
        extra=extra,
    )


def merge_batched_validation_results(
    output_dir: str,
    batch_metadata: Dict[str, Any],
    total_groups: int
) -> List[Dict[str, Any]]:
    """
    Merge results from multiple validation batch files.

    Args:
        output_dir: Directory containing validation_batch_N.json files
        batch_metadata: The batch metadata from prepare_batched_validation_tasks()
        total_groups: Total number of original groups

    Returns:
        Merged validation results list with proper group indices
    """
    output_path = Path(output_dir)

    # Compute required size from batch metadata — during reaggregation the
    # caller may pass a smaller total_groups (new group count) while the
    # batch metadata still references indices from the original grouping.
    effective_total = total_groups
    for batch_info in batch_metadata.get("batches", []):
        for idx in batch_info.get("group_indices", []):
            if idx >= effective_total:
                effective_total = idx + 1

    # Initialize results with failures for all groups
    merged_results: List[Dict[str, Any]] = [
        {
            "group_index": i,
            "status": "validation_failed",
            "reason": "Batch result not found",
            "confidence": 0.0,
            "error_type": ERROR_TYPE_UNKNOWN,
            "recoverable": True,
        }
        for i in range(effective_total)
    ]

    batches = batch_metadata.get("batches", [])
    batches_found = 0
    batches_missing = 0
    # Counters behind the single end-of-merge summary line below, instead of
    # one `[merge] Warning:` line printed per mismatched verdict.
    hash_join_total = 0
    hash_join_mismatched = 0

    for batch_info in batches:
        batch_index = batch_info["batch_index"]
        batch_file = output_path / f"validation_batch_{batch_index}.json"

        if not batch_file.exists():
            print(f"[merge] Warning: Batch file not found: {batch_file}")
            batches_missing += 1
            continue

        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)

            # Handle both wrapped format and direct list
            if isinstance(batch_data, dict):
                batch_results = batch_data.get("groups", [])
                batch_model = batch_data.get("metadata", {}).get("model")
            else:
                batch_results = batch_data
                batch_model = None

            # Tag every result with the model its own batch file reported, so
            # `save_validation_results` can derive an overall model when its
            # caller passes none (`_derive_model_from_results`), without this
            # function's return type growing a second value that every one of
            # its several cross-file callers would have to unpack.
            if batch_model:
                for _result in batch_results:
                    _result.setdefault("_source_model", batch_model)

            batches_found += 1

            # Map batch results to original group indices using cascading strategies.
            # LLMs may return global indices, local indices, extra results (filling
            # gaps in non-contiguous batches), or fewer results than expected.
            group_indices = batch_info["group_indices"]
            batch_group_ids = batch_info.get("group_ids", [])

            # Build reverse lookup: group_hash -> original_index
            hash_to_original_index: Dict[str, int] = {}
            for local_idx, gidx in enumerate(group_indices):
                if local_idx < len(batch_group_ids):
                    hash_to_original_index[batch_group_ids[local_idx]] = gidx

            # Strategy 0: Hash-based matching (group_hash is AUTHORITATIVE).
            # Whenever a result echoes a group_hash, route it by hash and ignore
            # its self-reported group_index — this corrects off-by-one and other
            # counting mistakes. Crucially, a result whose hash is *present but
            # foreign* to this batch means the subagent validated the WRONG group;
            # we discard it rather than letting it fall through to the index-based
            # strategies, where its (correct-looking) index would silently stamp
            # the wrong-group reasoning onto a correct slot. The unfilled target is
            # left as validation_failed so it gets revalidated instead of reported
            # with someone else's reasoning. Only results that omit a hash entirely
            # fall through. (See validation-join-by-hash: join by hash, never by
            # position.)
            filled_indices: set = set()
            unhashed_results: List[Dict[str, Any]] = []
            for result in batch_results:
                result_hash = result.get("group_hash") or result.get("group_id")
                if not result_hash:
                    unhashed_results.append(result)
                    continue
                if result_hash in hash_to_original_index:
                    original_index = hash_to_original_index[result_hash]
                    reported_idx = _reported_group_index(result)
                    hash_join_total += 1
                    if reported_idx != original_index:
                        hash_join_mismatched += 1
                    result["group_index"] = original_index
                    result["group_id"] = result_hash
                    merged_results[original_index] = result
                    filled_indices.add(original_index)
                else:
                    print(f"[merge] Warning: Batch {batch_index}: discarding result with "
                          f"foreign group_hash {result_hash} (not assigned to this batch — "
                          f"subagent likely validated the wrong group); leaving target "
                          f"group for revalidation")

            # Every expected group resolved by hash — nothing left to fall back on.
            if filled_indices and len(filled_indices) == len(group_indices):
                continue

            # Narrow the index-based fallback to groups still unfilled and to
            # results that carried no usable hash (hash-bearing results were already
            # routed or discarded above). Keep group_indices/batch_group_ids aligned.
            remaining_pairs = [
                (gidx, batch_group_ids[i] if i < len(batch_group_ids) else None)
                for i, gidx in enumerate(group_indices)
                if gidx not in filled_indices
            ]
            group_indices = [gi for gi, _ in remaining_pairs]
            batch_group_ids = [gid for _, gid in remaining_pairs]
            batch_results = unhashed_results
            global_indices_set = set(group_indices)

            # Strategy 1: Partial global index match — extract results whose
            # group_index matches an expected global index (ignore extras)
            matched_globals: Dict[int, Dict] = {}
            for result in batch_results:
                result_index = _reported_group_index(result)
                if result_index in global_indices_set:
                    matched_globals[result_index] = result

            if len(matched_globals) == len(group_indices):
                # All expected indices found via global match
                for original_index, result in matched_globals.items():
                    result["group_index"] = original_index
                    local_idx = group_indices.index(original_index)
                    if local_idx < len(batch_group_ids):
                        result["group_id"] = batch_group_ids[local_idx]
                    merged_results[original_index] = result
                continue

            # Strategy 2: Positional match — correct count, wrong indices
            if len(batch_results) == len(group_indices):
                for i, result in enumerate(batch_results):
                    original_index = group_indices[i]
                    result["group_index"] = original_index
                    if i < len(batch_group_ids):
                        result["group_id"] = batch_group_ids[i]
                    merged_results[original_index] = result
                continue

            # Strategy 3: Salvage partial global matches
            if matched_globals:
                for original_index, result in matched_globals.items():
                    result["group_index"] = original_index
                    local_idx = group_indices.index(original_index)
                    if local_idx < len(batch_group_ids):
                        result["group_id"] = batch_group_ids[local_idx]
                    merged_results[original_index] = result
                unmatched = global_indices_set - set(matched_globals.keys())
                if unmatched:
                    print(f"[merge] Warning: Batch {batch_index}: matched {len(matched_globals)}/{len(group_indices)}, missing {unmatched}")
                continue

            # Strategy 4: Legacy local index fallback
            for result in batch_results:
                result_index = _reported_group_index(result)
                if 0 <= result_index < len(group_indices):
                    original_index = group_indices[result_index]
                    result["group_index"] = original_index
                    if result_index < len(batch_group_ids):
                        result["group_id"] = batch_group_ids[result_index]
                    merged_results[original_index] = result
                else:
                    print(f"[merge] Warning: Index {result_index} out of range for batch {batch_index} (expected 0-{len(group_indices)-1})")

        except (json.JSONDecodeError, IOError) as e:
            print(f"[merge] Error reading batch file {batch_file}: {e}")
            batches_missing += 1

    print(f"[merge] Merged {batches_found} batches, {batches_missing} missing")
    # One summary line for the whole merge rather than one `[merge] Warning:`
    # per verdict (a correct 16-batch run used to print 41 of them): the hash
    # join is authoritative regardless, so a per-verdict echo mismatch is
    # advisory noise, not an error to itemize. The codebase has no
    # MULTI_LLM_DEBUG/verbose convention for validation.py to gate a per-line
    # form behind, so the summary is what prints.
    if hash_join_mismatched:
        print(f"[merge] {hash_join_mismatched} of {hash_join_total} verdicts "
              f"echoed a different group_index; all joined by hash")

    # Sort by group index
    merged_results.sort(key=lambda x: x.get("group_index", 0))

    return merged_results


def prepare_batched_revalidation_tasks(
    groups: List[Dict[str, Any]],
    validation_results: List[Dict[str, Any]],
    context: str,
    output_dir: str,
    plan_file: str = "",
    include_all_human: bool = False,
    model: str = "auto",
    max_per_batch: int = 4,
    orchestrator: str = "review_plan_orchestrator.py",
    base_ref: str = ""
) -> Dict[str, Any]:
    """
    Prepare batched revalidation tasks for items that failed validation.

    Similar to prepare_batched_validation_tasks, but only includes items that
    need revalidation (validation_failed status, optionally needs-human-decision).

    Args:
        groups: Original grouped suggestions/issues
        validation_results: Previous validation results
        context: Context for validation (plan content or diff)
        output_dir: Directory for batch result files
        plan_file: Path to the plan file for reference
        include_all_human: If True, also revalidate needs-human-decision items
        model: Model hint for reference
        max_per_batch: Maximum groups per batch
        orchestrator: Name of the orchestrator script for reaggregate command
        base_ref: Git ref for code diff context (empty string for plan reviews)

    Returns:
        Dict with batches and metadata, or empty batches if nothing to revalidate
    """
    # Identify items needing revalidation
    items_to_revalidate = []
    item_indices = []

    for i, val in enumerate(validation_results):
        status = val.get("status", "needs-human-decision")
        error_type = val.get("error_type", ERROR_TYPE_UNKNOWN)

        should_revalidate = (
            status == "validation_failed" or
            (include_all_human and status == "needs-human-decision" and
             error_type != ERROR_TYPE_AMBIGUOUS)
        )

        if should_revalidate and i < len(groups):
            items_to_revalidate.append(groups[i])
            item_indices.append(i)

    if not items_to_revalidate:
        grouped_file = str(Path(output_dir) / "grouped.json")
        return make_task_file(
            kind="revalidation",
            tasks=[],
            inputs=build_inputs(
                plan_file=plan_file,
                grouped_file=grouped_file,
                diff_base=base_ref,
            ),
            task_list_key="batches",
            extra={
                "total_batches": 0,
                "items_to_revalidate": 0,
                "item_indices": [],
                "original_validation": validation_results,
                "batching_stats": estimate_validation_batching_stats([]),
                "grouped_file": grouped_file,
                "plan_file": plan_file,
                "phase_dir": output_dir,
                "model_hint": model,
                "base_ref": base_ref,
            },
        )

    # Create batched tasks for the subset
    batched_tasks = prepare_batched_validation_tasks(
        groups=items_to_revalidate,
        context=context,
        output_dir=output_dir,
        plan_file=plan_file,
        model=model,
        max_per_batch=max_per_batch,
        orchestrator=orchestrator,
        base_ref=base_ref,
        kind="revalidation",
    )

    # Add mapping info for merging back
    batched_tasks["items_to_revalidate"] = len(items_to_revalidate)
    batched_tasks["item_indices"] = item_indices
    batched_tasks["original_validation"] = validation_results

    return batched_tasks


def merge_batched_revalidation_results(
    output_dir: str,
    revalidation_metadata: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Merge batched revalidation results back into original validation results.

    Args:
        output_dir: Directory containing revalidation batch files
        revalidation_metadata: Metadata from prepare_batched_revalidation_tasks()

    Returns:
        Updated validation results with revalidated items merged in
    """
    original_validation = revalidation_metadata.get("original_validation", [])
    item_indices = revalidation_metadata.get("item_indices", [])

    if not item_indices:
        return original_validation

    # Merge the batch results
    revalidated_count = revalidation_metadata.get("items_to_revalidate", 0)
    batch_results = merge_batched_validation_results(
        output_dir=output_dir,
        batch_metadata=revalidation_metadata,
        total_groups=revalidated_count
    )

    # Map back to original indices
    updated_results = list(original_validation)
    for j, original_idx in enumerate(item_indices):
        if j < len(batch_results):
            result = batch_results[j]
            result["group_index"] = original_idx
            result["revalidated"] = True
            updated_results[original_idx] = result

    return updated_results
