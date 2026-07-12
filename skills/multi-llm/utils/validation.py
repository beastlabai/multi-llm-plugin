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
from typing import Any, Dict, List, Optional

# Import the invoke_for_json utility
try:
    from .llm_client import invoke_for_json
    from .state_manager import generate_group_id, CURRENT_FORMAT_VERSION
except ImportError:
    from utils.llm_client import invoke_for_json
    from utils.state_manager import generate_group_id, CURRENT_FORMAT_VERSION


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


def _classify_validation_error(error: str, http_status: Optional[int] = None) -> str:
    """
    Classify a validation error by type.

    Note: This function is for FAILURES during validation (parsing, timeout, etc.).
    For genuine ambiguity where the LLM returns needs-human-decision, use
    ERROR_TYPE_AMBIGUOUS directly in the validation result handler.

    Args:
        error: The error message string
        http_status: Optional HTTP status code if available (e.g., 429 for rate limit)

    Returns:
        One of the ERROR_TYPE_* constants
    """
    error_lower = error.lower()

    # Check HTTP status codes first (most reliable)
    if http_status:
        if http_status == 429:
            return ERROR_TYPE_RATE_LIMITED
        elif http_status in (408, 504):
            return ERROR_TYPE_TIMEOUT
        elif http_status in (500, 502, 503):
            return ERROR_TYPE_MODEL_FAILURE

    # Parsing errors - be specific to avoid false positives
    parsing_patterns = [
        "json",
        "parse error",
        "parsing error",
        "decode error",
        "decodeerror",
        "syntax error",
        "no valid json",
        "unexpected token",
        "malformed",
        "invalid json",
        "expecting value",
        "expecting property",
    ]
    if any(pattern in error_lower for pattern in parsing_patterns):
        return ERROR_TYPE_PARSING

    # Also classify non-success-but-parsable responses (raw/malformed data) as parsing errors
    if "raw" in error_lower and ("response" in error_lower or "output" in error_lower):
        return ERROR_TYPE_PARSING

    # Timeout errors
    timeout_patterns = [
        "timeout",
        "timed out",
        "deadline exceeded",
        "connection timed",
        "request timed",
    ]
    if any(pattern in error_lower for pattern in timeout_patterns):
        return ERROR_TYPE_TIMEOUT

    # Rate limiting - use specific phrases to avoid matching unrelated "rate" or "limit" words
    rate_limit_patterns = [
        "rate limit",
        "rate-limit",
        "ratelimit",
        "too many requests",
        "quota exceeded",
        "quota limit",
        "throttled",
        "throttling",
        "429",
    ]
    if any(pattern in error_lower for pattern in rate_limit_patterns):
        return ERROR_TYPE_RATE_LIMITED

    # Model/infrastructure failures
    model_failure_patterns = [
        "binary not found",
        "not found in path",
        "model not found",
        "service unavailable",
        "internal server error",
        "server error",
        "connection refused",
        "connection failed",
        "api key",
        "authentication",
        "unauthorized",
    ]
    if any(pattern in error_lower for pattern in model_failure_patterns):
        return ERROR_TYPE_MODEL_FAILURE

    return ERROR_TYPE_UNKNOWN


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
        group_index = item.get("group_index", -1)
        status = item.get("status", "needs-human-decision")
        reason = item.get("reason", "No reason provided")
        confidence = item.get("confidence", 0.5)

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

    Uses group_id (stable content hash) for matching when available, with
    fallback to group_index for backward compatibility. This ensures validation
    results remain correctly associated with groups even after reaggregation
    changes the sort order.

    Args:
        groups: Original groups (list of dicts or SuggestionGroup objects)
        validation_results: Results from validate_groups()

    Returns:
        Groups with validation_status and validation_reason added
    """
    # Build lookups - prefer group_id, fallback to group_index
    validation_by_id = {
        r.get("group_id"): r
        for r in validation_results
        if r.get("group_id")
    }
    validation_by_index = {r["group_index"]: r for r in validation_results}

    default_validation = {
        "status": "needs-human-decision",
        "reason": "No validation result",
        "confidence": 0.0
    }

    updated_groups = []
    for i, group in enumerate(groups):
        # Handle both dict and SuggestionGroup objects
        if hasattr(group, 'to_dict'):
            group_dict = group.to_dict()
        else:
            group_dict = dict(group)

        # Compute group_id for matching
        gid = generate_group_id(group_dict)

        # Try group_id first, then fall back to index
        validation = (
            validation_by_id.get(gid) or
            validation_by_index.get(i) or
            default_validation
        )

        group_dict["validation_status"] = validation.get("status", "needs-human-decision")
        group_dict["validation_reason"] = validation.get("reason", "")
        group_dict["validation_confidence"] = validation.get("confidence", 0.0)
        # Copy error_type and recoverable fields if present
        if "error_type" in validation:
            group_dict["validation_error_type"] = validation.get("error_type")
        if "recoverable" in validation:
            group_dict["validation_recoverable"] = validation.get("recoverable")

        updated_groups.append(group_dict)

    return updated_groups


def save_validation_results(
    validation_results: List[Dict[str, Any]],
    output_path: Path,
    model: str = "unknown",
    groups: Optional[List[Dict[str, Any]]] = None
) -> None:
    """
    Save validation results to a JSON file with metadata.

    When groups are provided, computes and stores group_id for each result
    to enable stable matching after reaggregation.

    Args:
        validation_results: List of validation results
        output_path: Path to save the JSON file
        model: Model used for validation
        groups: Optional list of groups to compute group_ids from
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
            "confidence": v.get("confidence", 0.0),
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
) -> Dict[str, Any]:
    """
    Prepare batched validation tasks for sequential subagent processing.

    Instead of creating one task with all groups, this creates multiple batch
    tasks that can be processed sequentially by subagents with fresh context.

    This function uses a reference-based approach: instead of embedding full
    prompts in the validation_tasks.json file, it stores references to the
    grouped.json and plan files. Subagents read these files directly and
    construct their own understanding.

    Args:
        groups: List of suggestion groups to validate
        context: Context for validation (plan content or diff) - no longer embedded
        output_dir: Directory where subagents should write batch results
        plan_file: Path to the plan file for reference
        model: Model hint (not used by Claude Code, but preserved for reference)
        max_per_batch: Maximum groups per batch for normal items
        orchestrator: Name of the orchestrator script for reaggregate command
        base_ref: Git ref for code diff context (empty string for plan reviews)
        tasks_file: Path to the tasks file for review-tasks validation context

    Returns:
        Dict with:
        - batches: List of batch task dicts with group_indices, output_path, etc.
        - total_batches: Number of batches
        - batching_stats: Statistics about the batching
        - grouped_file: Path to grouped.json for subagent to read
        - plan_file: Path to plan file for subagent to read
        - phase_dir: Directory for this phase
        - reaggregate_command: Command to run after all batches complete
        - base_ref: Git ref for code diff context
    """
    if not groups:
        return {
            "batches": [],
            "total_batches": 0,
            "batching_stats": estimate_validation_batching_stats([]),
            "grouped_file": str(Path(output_dir) / "grouped.json"),
            "plan_file": plan_file,
            "phase_dir": output_dir,
            "model_hint": model,
            "base_ref": base_ref,
        }

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

    # Build batch task for each batch - reference-based (no embedded prompts)
    batch_tasks = []
    for batch in validation_batches:
        output_path = Path(output_dir) / f"validation_batch_{batch.batch_index}.json"

        # Include group_ids parallel to group_indices for stable matching
        batch_group_ids = [group_ids[idx] for idx in batch.group_indices]

        batch_tasks.append({
            "batch_index": batch.batch_index,
            "group_indices": batch.group_indices,
            "group_ids": batch_group_ids,
            "groups_count": batch.size,
            "is_high_priority": batch.is_high_priority,
            "output_path": str(output_path),
        })

    stats = estimate_validation_batching_stats(groups, max_per_batch)

    # Build reaggregate command hint
    reaggregate_command = (
        f'uv run --project "${{CLAUDE_SKILL_DIR}}" -- python "${{CLAUDE_SKILL_DIR}}/{orchestrator}" '
        f'--plan-file "{plan_file}" --reaggregate'
    )

    result = {
        "batches": batch_tasks,
        "total_batches": len(batch_tasks),
        "batching_stats": stats,
        "grouped_file": str(Path(output_dir) / "grouped.json"),
        "plan_file": plan_file,
        "phase_dir": output_dir,
        "reaggregate_command": reaggregate_command,
        "model_hint": model,
        "base_ref": base_ref,
    }
    if tasks_file:
        result["tasks_file"] = tasks_file
    return result


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
            else:
                batch_results = batch_data

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
                    reported_idx = result.get("group_index", result.get("index", -1))
                    if reported_idx != original_index:
                        print(f"[merge] Warning: Batch {batch_index}: hash {result_hash} "
                              f"-> index {original_index}, LLM reported index "
                              f"{reported_idx} (corrected via hash)")
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
                result_index = result.get("group_index", result.get("index", -1))
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
                result_index = result.get("group_index", result.get("index", -1))
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
        return {
            "batches": [],
            "total_batches": 0,
            "items_to_revalidate": 0,
            "item_indices": [],
            "original_validation": validation_results,
            "batching_stats": estimate_validation_batching_stats([]),
            "grouped_file": str(Path(output_dir) / "grouped.json"),
            "plan_file": plan_file,
            "phase_dir": output_dir,
            "model_hint": model,
            "base_ref": base_ref,
        }

    # Create batched tasks for the subset
    batched_tasks = prepare_batched_validation_tasks(
        groups=items_to_revalidate,
        context=context,
        output_dir=output_dir,
        plan_file=plan_file,
        model=model,
        max_per_batch=max_per_batch,
        orchestrator=orchestrator,
        base_ref=base_ref
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
