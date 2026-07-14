"""HTML report generator for validated suggestion groups.

Generates interactive HTML reports from grouped.json data with
model badges, section context previews, and embedded log snippets.
Supports both grouped.json and consolidated.json as data sources.
"""

import base64
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    from .stream_bootstrap import bootstrap_streams
except ImportError:
    # Direct script invocation (`python utils/html_report_generator.py`):
    # sys.path[0] is the utils/ directory, so import the sibling module directly.
    from stream_bootstrap import bootstrap_streams

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model icon mapping — maps model families to OpenRouter icon data URIs
# ---------------------------------------------------------------------------

# Icons directory (SVGs downloaded from OpenRouter)
_ICONS_DIR = Path(__file__).parent.parent / "assets" / "icons"

# Cache of loaded icon data URIs
_icon_cache: Dict[str, str] = {}


def _load_icon_data_uri(filename: str) -> str:
    """Load an icon file (SVG or PNG) and return a data URI string."""
    if filename in _icon_cache:
        return _icon_cache[filename]
    icon_path = _ICONS_DIR / filename
    if icon_path.exists():
        raw = icon_path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        mime = "image/svg+xml" if filename.endswith(".svg") else "image/png"
        uri = f"data:{mime};base64,{b64}"
    else:
        uri = ""
    _icon_cache[filename] = uri
    return uri


def _make_letter_icon_data_uri(letter: str, color: str) -> str:
    """Generate a simple SVG letter icon as a data URI."""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">'
        f'<rect width="64" height="64" rx="14" fill="{color}"/>'
        f'<text x="32" y="43" text-anchor="middle" font-family="system-ui,sans-serif" '
        f'font-size="36" font-weight="700" fill="white">{letter}</text></svg>'
    )
    b64 = base64.b64encode(svg.encode()).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


# Mapping from icon key to (icon_filename_or_None, fallback_letter, fallback_color)
_ICON_DEFS: Dict[str, tuple] = {
    "anthropic":  ("Anthropic.svg", "A", "#CC9B7A"),
    "openai":     ("OpenAI.svg", "O", "#10A37F"),
    "google":     ("GoogleGemini.svg", "G", "#4285F4"),
    "xai":        ("xAI.svg", "X", "#1DA1F2"),
    "moonshotai": ("MoonshotAI.svg", "K", "#5C3DAA"),
    "zhipu":      ("Zhipu.png", "Z", "#3366FF"),
    "minimax":    ("MiniMax.png", "M", "#FF6633"),
    "cursor":     ("Cursor.svg", "C", "#1E1E1E"),
    "kilocode":   ("Kilocode.svg", "K", "#FF4444"),
    "opencode":   ("Opencode.svg", "O", "#333333"),
    "codex":      ("Codex.svg", "C", "#111111"),
    "nvidia":     ("Nvidia.svg", "N", "#77B900"),
    "qwen":       ("Qwen.svg", "Q", "#615CED"),
    # No svgl.app logo available — letter-icon fallback only
    "xiaomi":     (None, "M", "#FF6900"),
    "arcee":      (None, "T", "#3B82F6"),
}


# Provider (CLI tool) → icon key. Used to render a provider icon alongside
# the model's company icon when they differ.
_PROVIDER_ICON_KEYS: Dict[str, str] = {
    "claude-code":  "anthropic",
    "gemini":       "google",
    "codex":        "codex",
    "cursor-agent": "cursor",
    "kilocode":     "kilocode",
    "opencode":     "opencode",
}


def _get_icon_data_uri(icon_key: str) -> str:
    """Get icon data URI for a given icon key."""
    if icon_key not in _ICON_DEFS:
        return ""
    svg_file, letter, color = _ICON_DEFS[icon_key]
    if svg_file:
        uri = _load_icon_data_uri(svg_file)
        if uri:
            return uri
    return _make_letter_icon_data_uri(letter, color)


def _resolve_model_icon_key(provider: str, model_name: str) -> str:
    """Determine the icon key for a provider:model combination.

    Maps the model name (and provider as fallback) to the underlying AI
    company whose icon should be displayed.
    """
    full = f"{provider}:{model_name}".lower()
    model_lower = model_name.lower()

    # Check model name patterns first (most specific)
    if any(k in model_lower for k in ("sonnet", "opus", "haiku", "claude")):
        return "anthropic"
    if any(k in model_lower for k in ("gpt", "codex", "o1", "o3", "o4")):
        return "openai"
    if "gemini" in model_lower:
        return "google"
    if "grok" in model_lower:
        return "xai"
    if any(k in full for k in ("kimi", "moonshot")):
        return "moonshotai"
    if any(k in full for k in ("glm", "z-ai", "zhipu")):
        return "zhipu"
    if "minimax" in full:
        return "minimax"
    if any(k in full for k in ("nemotron", "nvidia")):
        return "nvidia"
    if "qwen" in full:
        return "qwen"
    if any(k in full for k in ("mimo", "xiaomi")):
        return "xiaomi"
    if any(k in full for k in ("trinity", "arcee")):
        return "arcee"

    # Fall back to the provider's own icon when no model-family pattern matched
    return _PROVIDER_ICON_KEYS.get(provider.lower(), "")


def _resolve_provider_icon_key(provider: str) -> str:
    """Return the icon key for the invocation provider (CLI tool) itself."""
    return _PROVIDER_ICON_KEYS.get(provider.lower(), "")


# ---------------------------------------------------------------------------
# Canonical sort-order constants (single source of truth for the codebase)
# ---------------------------------------------------------------------------

VALIDATION_ORDER = {
    "needs-human-decision": 0,
    "valid": 1,
    "validation_failed": 2,
    "invalid": 3,
    "pending": 4,
}
UNKNOWN_STATUS_RANK = 5

IMPORTANCE_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "": 3}
UNKNOWN_IMPORTANCE_RANK = 3


# ---------------------------------------------------------------------------
# Shared sorting helpers
# ---------------------------------------------------------------------------

def compute_max_importance(suggestions: Optional[List[Dict[str, Any]]]) -> str:
    """Return the highest importance level among *suggestions*.

    Returns ``""`` when *suggestions* is ``None``, empty, or contains no
    recognised importance values — this maps to ``IMPORTANCE_ORDER[""]``
    (rank 3, after LOW) so that empty groups sort last within their
    validation-status tier.
    """
    if not suggestions:
        return ""
    best_rank = UNKNOWN_IMPORTANCE_RANK
    best_label = ""
    for s in suggestions:
        imp = (s.get("importance") or "").upper()
        rank = IMPORTANCE_ORDER.get(imp, UNKNOWN_IMPORTANCE_RANK)
        if rank < best_rank:
            best_rank = rank
            best_label = imp
    return best_label


def _sort_by_priority(
    items: List[Dict[str, Any]],
    status_fn: Callable[[Dict[str, Any]], str],
    importance_fn: Callable[[Dict[str, Any]], str],
) -> List[Dict[str, Any]]:
    """Generic priority sort.  Stamps ``originalIndex``, returns a new sorted list.

    Args:
        items: List of group dicts to sort.
        status_fn: Callable(item) -> str — extracts the validation status string.
        importance_fn: Callable(item) -> str — extracts the importance string.
    """
    for i, item in enumerate(items):
        if "originalIndex" not in item:
            item["originalIndex"] = i + 1
    def _key(item):
        return (
            VALIDATION_ORDER.get(status_fn(item), UNKNOWN_STATUS_RANK),
            IMPORTANCE_ORDER.get(importance_fn(item), UNKNOWN_IMPORTANCE_RANK),
            item.get("originalIndex", 0),
        )
    return sorted(items, key=_key)


def sort_groups_by_priority(
    groups: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Sort camelCase report-data dicts (post ``_build_group_data()``)."""
    return _sort_by_priority(
        groups,
        status_fn=lambda g: g.get("validationStatus", ""),
        importance_fn=lambda g: g.get("maxImportance", ""),
    )


def sort_raw_groups_by_priority(
    groups: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Sort raw snake_case grouped.json dicts."""
    return _sort_by_priority(
        groups,
        status_fn=lambda g: g.get("validation_status") or "",
        importance_fn=lambda g: compute_max_importance(g.get("suggestions", [])),
    )


def sort_consolidated_groups_by_priority(
    groups: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Sort consolidated group dicts (direct importance field, aggregate status)."""
    return _sort_by_priority(
        groups,
        status_fn=lambda g: g.get("validation_status") or "",
        importance_fn=lambda g: g.get("importance") or "",
    )


def derive_aggregate_validation_status(
    statuses: List[str],
) -> str:
    """Derive aggregate validation status from a list of constituent statuses.

    Selects the status with the **lowest** ``VALIDATION_ORDER`` rank (i.e.,
    highest priority wins).  Statuses not listed in ``VALIDATION_ORDER``
    receive ``UNKNOWN_STATUS_RANK`` and lose to all recognised statuses.

    Returns ``""`` when *statuses* is empty.
    """
    if not statuses:
        return ""
    # Use min() on rank — lowest rank = highest priority
    return min(
        statuses,
        key=lambda s: VALIDATION_ORDER.get(s, UNKNOWN_STATUS_RANK),
    )


def build_sort_config() -> Dict[str, Any]:
    """Build the ``sortConfig`` object for injection into HTML report data."""
    return {
        "validationOrder": VALIDATION_ORDER,
        "importanceOrder": IMPORTANCE_ORDER,
        "unknownStatusRank": UNKNOWN_STATUS_RANK,
        "unknownImportanceRank": UNKNOWN_IMPORTANCE_RANK,
    }


def hsl_to_hex(h: int, s: int, l: int) -> str:
    """Convert HSL values to hex color string.

    Args:
        h: Hue (0-360)
        s: Saturation (0-100)
        l: Lightness (0-100)

    Returns:
        Hex color string (e.g., "#B8860B")
    """
    s = s / 100
    l = l / 100

    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2

    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    r = int((r + m) * 255)
    g = int((g + m) * 255)
    b = int((b + m) * 255)

    return f"#{r:02X}{g:02X}{b:02X}"


def string_to_color(s: str) -> str:
    """Generate a consistent color from a string using hash.

    Uses HSL color space to produce visually distinct, pleasant colors.
    Same string always produces the same color.

    Args:
        s: Input string (e.g., model name "claude-3-opus")

    Returns:
        Hex color string (e.g., "#B8860B")
    """
    if not s:
        return "#6B7280"  # Default gray for empty string

    # Hash the string
    hash_bytes = hashlib.md5(s.encode()).digest()

    # Use first 3 bytes to generate HSL values
    # Hue: full 0-360 range
    hue = (hash_bytes[0] + hash_bytes[1] * 256) % 360

    # Saturation: 50-70% for good color visibility
    saturation = 50 + (hash_bytes[2] % 21)

    # Lightness: 45-55% for readability on both light/dark backgrounds
    lightness = 45 + (hash_bytes[3] % 11)

    # Convert HSL to RGB then to hex
    return hsl_to_hex(hue, saturation, lightness)


def parse_model_string(model_string: str) -> tuple[str, str]:
    """Parse a model string into provider and model name components.

    Handles formats like:
    - "anthropic:claude-3-opus" -> ("anthropic", "claude-3-opus")
    - "openai:gpt-4" -> ("openai", "gpt-4")
    - "claude-3-opus" -> ("", "claude-3-opus")

    Args:
        model_string: Full model identifier

    Returns:
        Tuple of (provider, model_name). Provider is empty string if not present.
    """
    if not model_string:
        return ("", "")

    if ":" in model_string:
        parts = model_string.split(":", 1)
        return (parts[0], parts[1])

    return ("", model_string)


def get_model_metadata(model_string: str) -> Dict[str, Any]:
    """Get color info for model badges using hash-based color generation.

    Generates separate consistent colors for provider and model name.
    Same provider always produces the same provider color.
    Same model name always produces the same model color.

    Args:
        model_string: Full model identifier (e.g., "anthropic:claude-3-opus", "gpt-4")

    Returns:
        Dict with:
        - 'full': The original model string
        - 'provider': Provider name (empty string if not present)
        - 'model': Model name
        - 'provider_color': Color for provider badge (gray if no provider)
        - 'model_color': Color for model badge
    """
    provider, model_name = parse_model_string(model_string)

    model_icon_key = _resolve_model_icon_key(provider, model_name)
    provider_icon_key = _resolve_provider_icon_key(provider)
    model_icon_uri = _get_icon_data_uri(model_icon_key) if model_icon_key else ""
    # Only emit the provider icon when it's distinct from the model icon —
    # avoids duplicate logos when the CLI and model are the same brand
    # (e.g. claude-code:opus, gemini:gemini-2.5-pro).
    provider_icon_uri = (
        _get_icon_data_uri(provider_icon_key)
        if provider_icon_key and provider_icon_key != model_icon_key
        else ""
    )

    return {
        "full": model_string,
        "provider": provider,
        "model": model_name,
        "provider_color": string_to_color(provider) if provider else "#6B7280",
        "model_color": string_to_color(model_name),
        "icon": model_icon_uri,
        "provider_icon": provider_icon_uri,
    }


def extract_section_contexts(
    plan_content: str,
    section_refs: Set[str],
    full_section: bool = False
) -> Dict[str, str]:
    """Extract context around each referenced section header.

    Takes the plan markdown content and a set of section references
    (like "### Step 2") and returns surrounding context text for each.

    By default, extracts ~15 lines around each header (for tooltip/hover use).
    When full_section=True, extracts the full text from the matched ### header
    to the next ### header (or end of content), including any nested headers
    (e.g., #### within a ### section).

    Args:
        plan_content: The full markdown content of the plan
        section_refs: Set of section references to find (e.g., {"### Step 2"})
        full_section: If True, capture full section text between ### headers
            instead of ~15 lines around each header. Default False.

    Returns:
        Dict mapping section reference to surrounding context text
    """
    if not plan_content or not section_refs:
        return {}

    lines = plan_content.split('\n')
    contexts = {}

    for ref in section_refs:
        if not ref:
            continue

        # Find line index where this section header appears
        ref_lower = ref.lower().strip()
        found_idx = -1

        for i, line in enumerate(lines):
            line_stripped = line.strip().lower()
            # Match exact header or header that starts with ref
            if line_stripped == ref_lower or line_stripped.startswith(ref_lower):
                found_idx = i
                break

        if found_idx == -1:
            # Try partial match on the text after ### markers
            ref_text = ref.lstrip('#').strip().lower()
            for i, line in enumerate(lines):
                line_text = line.lstrip('#').strip().lower()
                if ref_text in line_text:
                    found_idx = i
                    break

        if found_idx >= 0:
            if full_section:
                # Determine the header level of the matched line
                matched_line = lines[found_idx].lstrip()
                header_level = 0
                for ch in matched_line:
                    if ch == '#':
                        header_level += 1
                    else:
                        break

                # Find the next header at the same or higher level (fewer or equal #'s)
                # Headers with more #'s (e.g., #### inside ###) are nested and included
                end_idx = len(lines)
                for j in range(found_idx + 1, len(lines)):
                    line_stripped = lines[j].lstrip()
                    if line_stripped.startswith('#'):
                        # Count the header level of this line
                        level = 0
                        for ch in line_stripped:
                            if ch == '#':
                                level += 1
                            else:
                                break
                        # A header at same or higher level ends the section
                        if 0 < level <= header_level:
                            end_idx = j
                            break

                context_lines = lines[found_idx:end_idx]
                # Strip trailing blank lines for cleaner output
                while context_lines and context_lines[-1].strip() == '':
                    context_lines.pop()
                contexts[ref] = '\n'.join(context_lines)
            else:
                # Extract ~15 lines of context (7 before, the line, 7 after)
                start = max(0, found_idx - 7)
                end = min(len(lines), found_idx + 8)
                context_lines = lines[start:end]
                contexts[ref] = '\n'.join(context_lines)

    return contexts


def extract_plan_section_order(plan_content: str) -> List[str]:
    """Return an ordered list of ``###`` section headers as they appear in the plan.

    This captures the plan's original narrative order so the HTML report can
    offer a "Plan Order" view alongside the priority-sorted view.
    """
    if not plan_content:
        return []
    order: List[str] = []
    for line in plan_content.splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            order.append(stripped)
    return order


def embed_log_snippets(
    log_dir: Path,
    models: List[str],
    max_lines: int = 50
) -> Dict[str, str]:
    """Extract relevant portions of log files for each model.

    Looks for log_{model}.txt files in log_dir and returns
    the last max_lines of each log file.

    Args:
        log_dir: Directory containing log files
        models: List of model names to look for
        max_lines: Maximum number of lines to include (default 50)

    Returns:
        Dict mapping model name to log content (last max_lines)
    """
    log_snippets = {}

    if not log_dir.exists():
        return log_snippets

    for model in models:
        # Sanitize model name for filename (replace colons, slashes, etc.)
        safe_model = model.replace(':', '_').replace('/', '_').replace('\\', '_')
        log_file = log_dir / f"log_{safe_model}.txt"

        if not log_file.exists():
            # Try without sanitization
            log_file = log_dir / f"log_{model}.txt"

        if log_file.exists():
            try:
                content = log_file.read_text(encoding='utf-8', errors='replace')
                lines = content.split('\n')
                # Get last max_lines
                if len(lines) > max_lines:
                    lines = lines[-max_lines:]
                log_snippets[model] = '\n'.join(lines)
            except OSError:
                pass

    return log_snippets


def _build_suggestion_data(
    suggestion: Dict[str, Any],
    group_index: int,
    suggestion_index: int
) -> Dict[str, Any]:
    """Build the data structure for a single suggestion.

    Args:
        suggestion: Raw suggestion dict from grouped.json
        group_index: 1-based group index
        suggestion_index: 1-based suggestion index within group

    Returns:
        Formatted suggestion dict for report data
    """
    # Build file reference for code-review style suggestions
    file_path = suggestion.get("file", "")
    line_range = suggestion.get("line_range")
    file_ref = ""
    if file_path:
        file_ref = file_path
        if line_range and isinstance(line_range, (list, tuple)) and len(line_range) >= 2:
            file_ref += f":{line_range[0]}-{line_range[1]}"
        elif line_range and isinstance(line_range, (list, tuple)) and len(line_range) == 1:
            file_ref += f":{line_range[0]}"

    # Use stable content hash if available, fall back to positional ID
    suggestion_id = suggestion.get(
        "suggestion_hash", f"G{group_index}S{suggestion_index}"
    )

    result = {
        "id": suggestion_id,
        "title": suggestion.get("title", "Untitled"),
        "description": suggestion.get("desc", suggestion.get("description", "")),
        "importance": suggestion.get("importance", "MEDIUM").upper(),
        "type": suggestion.get("type", "unknown"),
        "sectionRef": suggestion.get("reference", suggestion.get("section", "")),
        "fileRef": file_ref,  # For code-review: file path with line range
        "model": suggestion.get("source_model", "unknown"),
        # PR-style contextual fields
        "anchorText": suggestion.get("anchor_text", None),
        "suggestedFix": suggestion.get("suggested_fix", None),
        "lineRange": suggestion.get("line_range", None),
    }

    # Add display label and display hash when available (Phase 1 stamps)
    display_label = suggestion.get("display_label")
    if display_label is not None:
        result["displayLabel"] = display_label
    display_hash = suggestion.get("display_hash")
    if display_hash is not None:
        result["displayHash"] = display_hash

    return result


def _build_group_data(
    group: Dict[str, Any],
    index: int,
    validation_results: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Build the data structure for a single group.

    Args:
        group: Raw group dict from grouped.json
        index: 0-based group index
        validation_results: Optional validation results list

    Returns:
        Formatted group dict for report data
    """
    # Get validation info from the group itself or from validation_results
    validation_status = group.get("validation_status")
    validation_reason = group.get("validation_reason", "")
    validation_confidence = group.get("validation_confidence", 0.0)

    # Try to get from validation_results if not in group
    if validation_results and index < len(validation_results):
        val_result = validation_results[index]
        if validation_status is None:
            validation_status = val_result.get("status")
        if not validation_reason:
            validation_reason = val_result.get("reason", "")
        if validation_confidence == 0.0:
            validation_confidence = val_result.get("confidence", 0.0)

    # Build suggestions list
    suggestions = []
    original_index = group.get("originalIndex", index + 1)
    raw_suggestions = group.get("suggestions", group.get("issues", []))
    for j, sugg in enumerate(raw_suggestions, start=1):
        suggestions.append(_build_suggestion_data(sugg, original_index, j))

    # Compute max importance from raw suggestions for sorting
    max_importance = compute_max_importance(raw_suggestions)

    result = {
        "index": index + 1,  # 1-based for display
        "originalIndex": group.get("originalIndex", index + 1),
        "theme": group.get("theme", "Unknown Theme"),
        "category": group.get("category", "unknown"),
        "models": group.get("models", []),
        "priorityScore": group.get("priority_score", 0),
        "validationStatus": validation_status or "pending",
        "validationReason": validation_reason,
        "validationConfidence": validation_confidence,
        "maxImportance": max_importance,
        "suggestions": suggestions,
    }

    # Add stable hash fields when available (Phase 1 stamps)
    group_hash = group.get("group_hash")
    if group_hash is not None:
        result["groupHash"] = group_hash
    display_label = group.get("display_label")
    if display_label is not None:
        result["displayLabel"] = display_label
    display_hash = group.get("display_hash")
    if display_hash is not None:
        result["displayHash"] = display_hash

    return result


# ---------------------------------------------------------------------------
# Consolidated.json support
# ---------------------------------------------------------------------------


def _validate_consolidated_schema(data: Any) -> bool:
    """Check whether *data* has the expected consolidated.json schema.

    Expected shapes:
    - Object with ``consolidated_groups`` key whose items contain
      ``underlying_group_ids`` and ``title``.
    - Top-level array of objects with ``underlying_group_ids``.
    """
    if isinstance(data, dict):
        groups = data.get("consolidated_groups")
        if not isinstance(groups, list):
            return False
        if len(groups) == 0:
            return True  # Empty is valid
        first = groups[0]
        return (
            isinstance(first, dict)
            and "underlying_group_ids" in first
            and "title" in first
        )
    if isinstance(data, list):
        if len(data) == 0:
            return True
        first = data[0]
        return isinstance(first, dict) and "underlying_group_ids" in first
    return False


def _load_consolidated_groups(
    consolidated_path: Path,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    """Load consolidated groups from *consolidated_path*.

    Returns ``(groups_list, metadata_dict)`` on success.
    Returns ``(None, None)`` if the file is missing or malformed.
    Logs a warning on malformed data.
    """
    if not consolidated_path.exists():
        return None, None
    try:
        raw = json.loads(consolidated_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "consolidated.json is malformed, falling back to grouped.json: %s", exc
        )
        return None, None

    if not _validate_consolidated_schema(raw):
        logger.warning(
            "consolidated.json is malformed, falling back to grouped.json"
        )
        return None, None

    if isinstance(raw, list):
        return raw, None
    # dict form
    return raw.get("consolidated_groups", []), raw.get("metadata")


def _build_grouped_lookup(
    groups: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Build a lookup table mapping group hashes/IDs to their group dicts.

    Looks at ``group_hash``, ``group_id``, ``groupHash`` (camelCase report
    data), and falls back to using the list index as a string key.
    """
    lookup: Dict[str, Dict[str, Any]] = {}
    for i, group in enumerate(groups):
        for key_field in ("group_hash", "group_id", "groupHash"):
            key = group.get(key_field)
            if key:
                lookup[key] = group
        # Also index by list position (string)
        lookup[str(i)] = group
    return lookup


def _resolve_underlying_groups(
    consolidated_group: Dict[str, Any],
    grouped_lookup: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Resolve ``underlying_group_ids`` to actual group dicts.

    Falls back to ``underlying_group_indices`` if ID resolution fails.
    Returns the resolved list (may be empty if resolution is impossible).
    """
    resolved: List[Dict[str, Any]] = []

    # Try underlying_group_ids first
    underlying_ids = consolidated_group.get("underlying_group_ids", [])
    for uid in underlying_ids:
        grp = grouped_lookup.get(uid)
        if grp is not None:
            resolved.append(grp)

    # If ID resolution yielded nothing, try index-based
    if not resolved:
        underlying_indices = consolidated_group.get("underlying_group_indices", [])
        for idx in underlying_indices:
            grp = grouped_lookup.get(str(idx))
            if grp is not None:
                resolved.append(grp)

    return resolved


def _expand_consolidated_into_groups(
    consolidated_groups: List[Dict[str, Any]],
    raw_groups: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Expand consolidated groups into underlying group dicts for rendering.

    Returns:
        (expanded_groups, umbrella_map)
        - expanded_groups: flat list of underlying group dicts (post
          ``_build_group_data()``) with consolidated metadata attached.
        - umbrella_map: maps underlying group hash to its consolidated
          group metadata (title, description, consolidated_id).
          Used by the template to render umbrella banners.
    """
    grouped_lookup = _build_grouped_lookup(raw_groups)
    expanded: List[Dict[str, Any]] = []
    umbrella_map: Dict[str, Dict[str, Any]] = {}
    seen_group_hashes: set = set()

    for cg in consolidated_groups:
        resolved = _resolve_underlying_groups(cg, grouped_lookup)
        consolidated_meta = {
            "consolidatedId": cg.get("consolidated_id", ""),
            "consolidatedTitle": cg.get("title", ""),
            "consolidatedDescription": cg.get("description", ""),
            "consolidatedImportance": cg.get("importance", ""),
            "consolidatedType": cg.get("type", ""),
            "isSingleton": cg.get("is_singleton", False),
            "displayIndex": cg.get("display_index", 0),
            "reasoning": cg.get("reasoning", ""),
        }

        if not resolved:
            # Cannot resolve underlying groups - create a flat card from
            # consolidated data directly
            flat_group = {
                "theme": cg.get("title", "Consolidated Suggestion"),
                "category": cg.get("type", "unknown"),
                "models": [],
                "validation_status": "",
                "suggestions": [{
                    "title": cg.get("title", ""),
                    "desc": cg.get("description", ""),
                    "importance": cg.get("importance", "MEDIUM"),
                    "type": cg.get("type", "modification"),
                    "reference": cg.get("reference", ""),
                    "source_model": "consolidated",
                }],
                "_consolidated_meta": consolidated_meta,
                "_is_consolidated_flat": True,
            }
            expanded.append(flat_group)
        else:
            # Tag each resolved group with consolidated metadata.
            # Use shallow copies to avoid mutating the original group dicts
            # from raw_groups (which are shared references via the lookup).
            first_in_umbrella = True
            for grp in resolved:
                grp_hash = grp.get("group_hash", grp.get("group_id", ""))
                if grp_hash and grp_hash in seen_group_hashes:
                    continue
                if grp_hash:
                    seen_group_hashes.add(grp_hash)

                # Shallow copy before attaching metadata so callers'
                # original group dicts are not mutated.
                grp_copy = {**grp}
                grp_copy["_consolidated_meta"] = consolidated_meta
                if first_in_umbrella:
                    grp_copy["_umbrella_first"] = True
                    first_in_umbrella = False

                # Map group hash to umbrella info for template rendering
                if grp_hash:
                    umbrella_map[grp_hash] = consolidated_meta

                expanded.append(grp_copy)

    return expanded, umbrella_map


def _load_data_source(
    phase_dir: Path,
    data_source: str,
    groups: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Dict[str, Any]]], List[Dict[str, str]]]:
    """Determine and load the data source for report generation.

    Args:
        phase_dir: Phase directory containing grouped.json / consolidated.json
        data_source: ``"auto"``, ``"consolidated"``, or ``"grouped"``
        groups: The groups already passed by the caller (from grouped.json)

    Returns:
        (effective_groups, umbrella_map, notices)
        - effective_groups: The groups to render (may be expanded from consolidated)
        - umbrella_map: Mapping of group hash -> consolidated metadata, or None
        - notices: List of notice dicts ``{"type": ..., "message": ...}``
    """
    notices: List[Dict[str, str]] = []

    if data_source == "grouped":
        return groups, None, notices

    consolidated_path = phase_dir / "consolidated.json"
    grouped_path = phase_dir / "grouped.json"

    if data_source == "consolidated":
        if not consolidated_path.exists():
            raise FileNotFoundError(
                f"consolidated.json not found at {consolidated_path}"
            )
        cons_groups, _meta = _load_consolidated_groups(consolidated_path)
        if cons_groups is None:
            raise ValueError(
                f"consolidated.json at {consolidated_path} is malformed"
            )
        expanded, umbrella_map = _expand_consolidated_into_groups(
            cons_groups, groups
        )
        return expanded, umbrella_map, notices

    # data_source == "auto"
    if consolidated_path.exists():
        cons_groups, _meta = _load_consolidated_groups(consolidated_path)
        if cons_groups is not None:
            expanded, umbrella_map = _expand_consolidated_into_groups(
                cons_groups, groups
            )
            return expanded, umbrella_map, notices
        else:
            # Malformed consolidated.json - fall back
            notices.append({
                "type": "warning",
                "message": (
                    "Report generated from grouped data \u2014 "
                    "consolidated data was unavailable"
                ),
            })
            return groups, None, notices

    # No consolidated.json - use grouped data
    return groups, None, notices


def _parse_tasks_markdown(content: str) -> Dict[str, Any]:
    """Parse a tasks.md markdown file into a metadata dict keyed by task ref.

    The tasks.md format uses headers like ``### Task T001: Create base config``
    followed by metadata lines (Dependencies, Files, Complexity) and body text.

    Returns a dict mapping the full section ref (e.g.
    ``"### Task T001: Create base config"``) to a metadata dict with keys:
    ``task_id``, ``title``, ``dependencies``, ``files``, ``complexity``,
    ``description``.
    """
    # Match task headers: ## Task T001: ... or ### Task T001: ...
    header_pattern = re.compile(
        r'^(#{2,3}\s+(?:Task\s+)?T\d+[^:\n]*(?::\s*.*)?)', re.MULTILINE
    )
    headers = list(header_pattern.finditer(content))
    if not headers:
        return {}

    result: Dict[str, Any] = {}
    for i, match in enumerate(headers):
        header_text = match.group(1).strip()
        start = match.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(content)
        body = content[start:end].strip()

        # Extract task ID (e.g. "T001")
        tid_match = re.search(r'(T\d+)', header_text)
        task_id = tid_match.group(1) if tid_match else ""

        # Extract title (text after "Task T001:" or "T001:")
        title_match = re.search(r'T\d+[^:]*:\s*(.*)', header_text)
        title = title_match.group(1).strip() if title_match else header_text

        # Extract structured metadata from body lines
        deps = ""
        files = ""
        complexity = ""
        desc_lines = []
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("**dependencies**"):
                deps = re.sub(r'\*\*Dependencies\*\*\s*:\s*', '', stripped, flags=re.IGNORECASE).strip()
            elif stripped.lower().startswith("**files"):
                files = re.sub(r'\*\*Files[^*]*\*\*\s*:\s*', '', stripped, flags=re.IGNORECASE).strip()
            elif stripped.lower().startswith("**complexity**"):
                complexity = re.sub(r'\*\*Complexity\*\*\s*:\s*', '', stripped, flags=re.IGNORECASE).strip()
            elif stripped and not stripped.startswith("---"):
                desc_lines.append(stripped)

        meta: Dict[str, Any] = {
            "task_id": task_id,
            "title": title,
            "dependencies": deps,
            "files": files,
            "complexity": complexity,
            "description": "\n".join(desc_lines[:5]),  # First 5 lines
        }

        # Key by the full header text to match sectionRef values
        result[header_text] = meta
        # Also key by bare task ID for fallback matching
        if task_id:
            result[task_id] = meta

    return result


def generate_html_report(
    groups: List[Dict[str, Any]],
    plan_path: Path,
    phase_dir: Path,
    phase_type: str,
    models: List[str],
    failed_models: Optional[Dict[str, str]] = None,
    validation_results: Optional[List[Dict[str, Any]]] = None,
    base_ref: Optional[str] = None,
    tasks_path: Optional[Path] = None,
    template_style: str = 'pr',
    diff_data: Optional[Dict] = None,
    data_source: str = 'auto',
    file_snapshots: Optional[Dict[str, list]] = None,
) -> str:
    """Generate complete HTML report and return the HTML string.

    Args:
        groups: Validated groups from grouped.json
        plan_path: Path to the original plan file
        phase_dir: Directory for this phase (for log files)
        phase_type: Phase identifier ("review-plan", "code-review", "review-tasks")
        models: List of models that were used
        failed_models: Optional dict of model name -> error message for failures
        validation_results: Optional list of validation results
        base_ref: Optional git ref to diff against for PR-style code context.
            When provided (and ``diff_data`` is ``None``), ``capture_diff_hunks()``
            is called to obtain structured diff data for files referenced by
            suggestions.
        tasks_path: Optional path to a tasks JSON file.  When provided, task
            metadata is loaded and passed to ``build_task_view()``.
        template_style: Template to use — ``'pr'`` selects
            ``pr_report_template.html`` (default), ``'flat'`` selects the
            original ``report_template.html``.
        diff_data: Optional pre-computed diff hunk data (as returned by
            ``capture_diff_hunks()``).  When provided, it is used directly as
            ``reportData.diffData``, bypassing ``base_ref``-driven capture.
            This allows callers like ``reaggregate_from_existing_files()`` to
            inject cached diff data without re-running the git operation.
        data_source: Data source strategy — ``"auto"`` (default) checks for
            ``consolidated.json`` first, falls back to ``grouped.json``;
            ``"consolidated"`` uses ``consolidated.json`` directly (raises
            if missing); ``"grouped"`` uses ``grouped.json`` directly.
        file_snapshots: Optional dict mapping file paths to pre-captured
            line lists (as returned by ``capture_file_snapshots()``).  When
            provided, ``capture_file_context()`` uses these snapshots instead
            of reading files from disk, ensuring the report reflects file
            state at review start rather than at report-generation time.

    Returns:
        Complete HTML string for the report

    Raises:
        FileNotFoundError: When ``data_source="consolidated"`` and the file
            does not exist.
        ValueError: When ``data_source`` is not one of the allowed values,
            or when ``data_source="consolidated"`` and the file is malformed.
    """
    if data_source not in ("auto", "consolidated", "grouped"):
        raise ValueError(
            f"data_source must be 'auto', 'consolidated', or 'grouped', "
            f"got {data_source!r}"
        )

    # --- Resolve data source (consolidated vs grouped) ---
    effective_groups, umbrella_map, source_notices = _load_data_source(
        phase_dir, data_source, groups
    )

    # Select template based on template_style
    templates_dir = Path(__file__).parent.parent / 'templates'
    if template_style == 'pr':
        template_path = templates_dir / 'pr_report_template.html'
        if not template_path.exists():
            import warnings
            warnings.warn(
                f"PR template not found at {template_path}, falling back to flat template"
            )
            template_path = templates_dir / 'report_template.html'
    else:
        template_path = templates_dir / 'report_template.html'

    if not template_path.exists():
        # Return a basic error HTML if template is missing
        return f"""<!DOCTYPE html>
<html>
<head><title>Report Error</title></head>
<body>
<h1>Error: Template Not Found</h1>
<p>Could not find template at: {template_path}</p>
<p>Please ensure the template file exists.</p>
</body>
</html>"""

    template = template_path.read_text(encoding='utf-8')

    # Gather all section references from suggestions
    section_refs: Set[str] = set()
    for group in effective_groups:
        for sugg in group.get("suggestions", group.get("issues", [])):
            ref = sugg.get("reference", sugg.get("section", ""))
            if ref:
                section_refs.add(ref)

    # Read plan content for section contexts
    plan_content = ""
    if plan_path.exists():
        try:
            plan_content = plan_path.read_text(encoding='utf-8')
        except OSError:
            pass

    # Extract section contexts for hover previews
    section_contexts = extract_section_contexts(plan_content, section_refs)

    # Get log snippets
    log_snippets = embed_log_snippets(phase_dir, models)

    # Build groups data
    groups_data = []
    for i, group in enumerate(effective_groups):
        built = _build_group_data(group, i, validation_results)
        # Carry through consolidated metadata if present
        cons_meta = group.get("_consolidated_meta")
        if cons_meta is not None:
            built["_consolidatedMeta"] = cons_meta
        if group.get("_umbrella_first"):
            built["_umbrellaFirst"] = True
        if group.get("_is_consolidated_flat"):
            built["_isConsolidatedFlat"] = True
        groups_data.append(built)

    # Sort groups by priority (validation status, then importance, then original index)
    groups_data = sort_groups_by_priority(groups_data)

    # Build model metadata for badges
    model_metadata = {}
    for model in models:
        model_metadata[model] = get_model_metadata(model)

    # Extract plan title from markdown (first # heading), fallback to filename
    plan_title = ""
    for line in plan_content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            plan_title = stripped[2:].strip()
            break
    if not plan_title:
        plan_title = plan_path.stem

    # --- PR-style view indexes ---
    file_view = build_file_view(groups_data)
    section_view = build_section_view(groups_data)

    # Load task metadata if tasks_path provided.
    # Supports both JSON (tasks.json) and markdown (tasks.md) formats.
    tasks_metadata: Optional[Dict[str, Any]] = None
    if tasks_path is not None:
        try:
            raw_text = tasks_path.read_text(encoding='utf-8')
            if tasks_path.suffix == '.json':
                tasks_metadata = json.loads(raw_text)
            elif tasks_path.suffix == '.md':
                tasks_metadata = _parse_tasks_markdown(raw_text)
            else:
                # Try JSON first, fall back to markdown parsing
                try:
                    tasks_metadata = json.loads(raw_text)
                except json.JSONDecodeError:
                    tasks_metadata = _parse_tasks_markdown(raw_text)
        except OSError as exc:
            logger.warning("Could not load tasks metadata from %s: %s", tasks_path, exc)
            tasks_metadata = None

    task_view = build_task_view(groups_data, tasks_metadata)

    # --- Diff data: pre-computed takes precedence over base_ref capture ---
    # Normalize empty string base_ref to None so we don't attempt a
    # diff capture with an invalid (empty) ref.
    if not base_ref:
        base_ref = None
    resolved_diff_data: Optional[Dict[str, Any]] = None
    if diff_data is not None:
        # Use pre-computed diff data directly
        resolved_diff_data = diff_data
    elif base_ref is not None:
        # Collect file paths referenced by suggestions
        referenced_files: List[str] = []
        seen_files: Set[str] = set()
        for group in groups_data:
            for sugg in group.get("suggestions", []):
                file_ref = sugg.get("fileRef", "")
                fpath = _extract_file_path(file_ref)
                if fpath and fpath not in seen_files:
                    seen_files.add(fpath)
                    referenced_files.append(fpath)
        if referenced_files:
            try:
                from .git_utils import capture_diff_hunks
                resolved_diff_data = capture_diff_hunks(base_ref, referenced_files)
            except Exception as exc:
                import warnings
                warnings.warn(f"Failed to capture diff hunks: {exc}")
                resolved_diff_data = None

    # --- File contexts fallback chain ---
    # For each file referenced by suggestions, try:
    # 1. diff hunks (already in resolved_diff_data)
    # 2. file content via capture_file_context()
    # 3. anchor_text from the suggestion itself
    file_contexts: Dict[str, Any] = {}
    for group in groups_data:
        for sugg in group.get("suggestions", []):
            file_ref = sugg.get("fileRef", "")
            fpath = _extract_file_path(file_ref)
            if not fpath or fpath in file_contexts:
                continue

            # Check if diff data already covers this file
            if resolved_diff_data and fpath in resolved_diff_data:
                file_contexts[fpath] = {"source": "diff", "available": True}
                continue

            # Try file content fallback
            line_range = sugg.get("lineRange")
            if line_range and isinstance(line_range, (list, tuple)) and len(line_range) >= 1:
                try:
                    from .git_utils import capture_file_context
                    ctx = capture_file_context(
                        fpath, list(line_range),
                        file_snapshots=file_snapshots,
                    )
                    if ctx is not None:
                        file_contexts[fpath] = {
                            "source": "file",
                            "lines": ctx,
                            "available": True,
                        }
                        continue
                except Exception:
                    pass

            # Final fallback: anchor_text
            anchor = sugg.get("anchorText")
            if anchor:
                file_contexts[fpath] = {
                    "source": "anchor",
                    "anchorText": anchor,
                    "available": True,
                }
            else:
                file_contexts[fpath] = {
                    "source": "none",
                    "available": False,
                }

    # --- Augment file contexts with lines for ALL suggestion lineRanges ---
    # The loop above only captures context for the first suggestion per file.
    # For files with diff data, outside-diff suggestions have no context lines.
    # This second pass captures file content covering all suggestion lineRanges
    # so the template can render context around outside-diff suggestions.
    file_line_ranges: Dict[str, List[List[int]]] = {}
    for group in groups_data:
        for sugg in group.get("suggestions", []):
            file_ref = sugg.get("fileRef", "")
            fpath = _extract_file_path(file_ref)
            line_range = sugg.get("lineRange")
            if (
                fpath
                and line_range
                and isinstance(line_range, (list, tuple))
                and len(line_range) >= 1
            ):
                file_line_ranges.setdefault(fpath, []).append(list(line_range))

    for fpath, ranges in file_line_ranges.items():
        entry = file_contexts.get(fpath, {})
        # Skip if we already have context lines
        if entry.get("lines"):
            continue
        # Merge all ranges into one broad range — the template filters
        # to ±5 lines around each specific suggestion's lineRange.
        all_starts = [r[0] for r in ranges]
        all_ends = [r[1] if len(r) > 1 else r[0] for r in ranges]
        merged_range = [min(all_starts), max(all_ends)]
        try:
            from .git_utils import capture_file_context
            ctx = capture_file_context(
                fpath, merged_range,
                context_lines=5,
                file_snapshots=file_snapshots,
            )
            if ctx is not None:
                if fpath in file_contexts:
                    file_contexts[fpath]["lines"] = ctx
                else:
                    file_contexts[fpath] = {
                        "source": "file",
                        "lines": ctx,
                        "available": True,
                    }
        except Exception:
            pass

    # --- Plan section ordering (narrative order from markdown) ---
    section_order = extract_plan_section_order(plan_content)

    # --- Full section contexts for sectionView ---
    # Extract for ALL section_order headers + suggestion refs so every
    # plan section has rendered content in the PR view.
    all_section_refs: Set[str] = set(section_order)
    all_section_refs.update(section_refs)
    full_section_contexts: Dict[str, str] = {}
    if all_section_refs:
        full_section_contexts = extract_section_contexts(
            plan_content, all_section_refs, full_section=True
        )

    # --- Merge all plan sections into sectionView ---
    # Ensure every section from section_order appears in section_view,
    # even if it has no suggestions. This enables the "full document" PR view.
    def _normalize_section_ref(ref: str) -> str:
        """Strip leading # and lowercase for fuzzy matching."""
        return ref.lstrip('#').strip().lower()

    def _find_canonical_match(
        ref: str, canonical_map: Dict[str, str]
    ) -> Optional[str]:
        """Find a canonical sectionView key matching a section_order ref."""
        norm = _normalize_section_ref(ref)
        # Exact match
        if norm in canonical_map:
            return canonical_map[norm]
        # Starts-with fallback (handles truncated refs)
        for canon_norm, canon_key in canonical_map.items():
            if canon_norm.startswith(norm) or norm.startswith(canon_norm):
                return canon_key
        return None

    # Build normalized map of existing sectionView keys
    sv_canonical: Dict[str, str] = {}  # normalized -> original key
    for sv_key in section_view:
        sv_canonical[_normalize_section_ref(sv_key)] = sv_key

    # Merge sectionView keys that match section_order entries,
    # and inject empty entries for sections not already in sectionView.
    for so_ref in section_order:
        match = _find_canonical_match(so_ref, sv_canonical)
        if match is None:
            # Section has no suggestions - inject empty entry
            section_view[so_ref] = {
                "suggestions": [],
                "suggestionCount": 0,
                "maxImportance": "",
                "hasSuggestions": False,
            }
        else:
            # Mark existing entry as having suggestions
            section_view[match]["hasSuggestions"] = True

    # Ensure hasSuggestions is set on ALL entries
    for key in section_view:
        if "hasSuggestions" not in section_view[key]:
            has = len(section_view[key].get("suggestions", [])) > 0
            section_view[key]["hasSuggestions"] = has

    # --- Merge all tasks into taskView ---
    # If tasks_metadata is available, ensure every task appears in task_view.
    if tasks_metadata:
        # Extract bare task IDs (T001, T002, etc.) to avoid duplicates
        existing_task_ids: Set[str] = set()
        for tk in task_view:
            if tk.startswith('_'):
                continue
            # Normalize: extract T\d+ pattern
            m = re.match(r'(T\d+)', tk)
            if m:
                existing_task_ids.add(m.group(1))
            else:
                existing_task_ids.add(tk)

        for task_id, meta in tasks_metadata.items():
            # Extract bare ID
            bare_id_match = re.match(r'(T\d+)', task_id)
            bare_id = bare_id_match.group(1) if bare_id_match else task_id
            if bare_id not in existing_task_ids:
                task_view[task_id] = {
                    "taskMetadata": meta,
                    "suggestions": [],
                    "suggestionCount": 0,
                    "maxImportance": "",
                    "hasSuggestions": False,
                }
                existing_task_ids.add(bare_id)

    # Ensure hasSuggestions is set on ALL task entries
    for key in task_view:
        if key.startswith('_'):
            continue
        if "hasSuggestions" not in task_view[key]:
            has = len(task_view[key].get("suggestions", [])) > 0
            task_view[key]["hasSuggestions"] = has

    # --- Global suggestions: those without file or section anchors ---
    global_suggestions_list: List[Dict[str, Any]] = []
    global_entry = file_view.get("_global", {})
    for grp in global_entry.get("suggestions", []):
        global_suggestions_list.extend(grp.get("suggestions", []))

    # Build the complete report data
    report_data = {
        "planPath": str(plan_path),
        "planTitle": plan_title,
        "title": plan_title,
        "baseRef": base_ref,
        "phase": phase_type,
        # Intentionally False for all phases — the review-tasks phase now
        # supports skip/override actions, so interactive controls are enabled
        # everywhere.  See review_orchestrator_base.py aggregate_results().
        "readOnly": False,
        "generatedAt": datetime.now().isoformat(),
        "models": models,
        "modelMetadata": model_metadata,
        "failedModels": failed_models or {},
        "sortConfig": build_sort_config(),
        "groups": groups_data,
        "sectionContexts": section_contexts,
        "logSnippets": log_snippets,
        # PR-style view indexes
        "fileView": file_view,
        "sectionView": section_view,
        "taskView": task_view,
        "globalSuggestions": global_suggestions_list,
        "fullSectionContexts": full_section_contexts,
        "sectionOrder": section_order,
        "notices": source_notices,
        "summary": {
            "totalGroups": len(effective_groups),
            "totalSuggestions": sum(
                len(g.get("suggestions", g.get("issues", [])))
                for g in effective_groups
            ),
            "validCount": sum(
                1 for g in groups_data
                if g["validationStatus"] == "valid"
            ),
            "invalidCount": sum(
                1 for g in groups_data
                if g["validationStatus"] == "invalid"
            ),
            "needsHumanCount": sum(
                1 for g in groups_data
                if g["validationStatus"] == "needs-human-decision"
            ),
            "validationFailedCount": sum(
                1 for g in groups_data
                if g["validationStatus"] == "validation_failed"
            ),
        }
    }

    # Conditionally add diff data and file contexts
    if resolved_diff_data is not None:
        report_data["diffData"] = resolved_diff_data
    if file_contexts:
        report_data["fileContexts"] = file_contexts

    # Add consolidated metadata when using consolidated data source
    if umbrella_map is not None:
        report_data["umbrellaMap"] = umbrella_map
        report_data["dataSource"] = "consolidated"
    else:
        report_data["dataSource"] = "grouped"

    # Record the template style so a later post-apply regeneration
    # (regenerate_report_with_human_decisions) re-renders with the same template.
    report_data["templateStyle"] = template_style

    html = _embed_report_data(report_data, template)

    # Persist the assembled report_data so the apply phase can overlay human
    # decisions onto it and re-embed without reconstructing runtime inputs
    # (diff_data, base_ref, models, ...). Best-effort: never fail report
    # generation if the sidecar cannot be written.
    try:
        (phase_dir / "report_data.json").write_text(
            json.dumps(report_data, indent=2), encoding="utf-8"
        )
    except OSError:
        pass

    return html


def _embed_report_data(report_data: Dict[str, Any], template: str) -> str:
    """Embed ``report_data`` as JSON into ``template`` at the data placeholder.

    Shared by initial generation and post-apply regeneration so both paths
    escape and inject the data identically.
    """
    # Escape sequences that would break out of the surrounding <script> tag:
    #   </script>, </ in general, and <!-- (HTML comment opener inside JS).
    # \/ is valid JSON (RFC 8259 §7), so this stays parseable.
    json_data = (
        json.dumps(report_data, indent=2)
        .replace("</", "<\\/")
        .replace("<!--", "<\\!--")
    )

    # Replace the placeholder in the template
    # The template has: const reportData = /* REPORT_DATA_PLACEHOLDER */null;
    # We replace the entire "/* ... */null" pattern with the actual JSON data
    if "/* REPORT_DATA_PLACEHOLDER */null" in template:
        return template.replace("/* REPORT_DATA_PLACEHOLDER */null", json_data)
    elif "/* REPORT_DATA_PLACEHOLDER */" in template:
        return template.replace("/* REPORT_DATA_PLACEHOLDER */", json_data)
    elif "REPORT_DATA_PLACEHOLDER" in template:
        return template.replace("REPORT_DATA_PLACEHOLDER", json_data)
    else:
        # Fallback: inject before </body>
        inject_script = f"""
<script>
const reportData = {json_data};
// Initialize report if renderReport function exists
if (typeof renderReport === 'function') {{
    renderReport(reportData);
}}
</script>
"""
        return template.replace("</body>", f"{inject_script}</body>")


def write_html_report(html: str, phase_dir: Path) -> Path:
    """Write HTML report to the phase directory.

    Args:
        html: The complete HTML content to write
        phase_dir: Directory to write the report to

    Returns:
        Path to the written report file
    """
    phase_dir.mkdir(parents=True, exist_ok=True)
    report_path = phase_dir / "report.html"
    report_path.write_text(html, encoding='utf-8')
    return report_path


# ---------------------------------------------------------------------------
# PR-style view index builders
# ---------------------------------------------------------------------------

def _extract_file_path(file_ref: str) -> str:
    """Extract the file path from a fileRef string, stripping line ranges.

    Examples:
        "src/auth/login.py:43-50" -> "src/auth/login.py"
        "src/auth/login.py:43" -> "src/auth/login.py"
        "src/auth/login.py" -> "src/auth/login.py"
        "" -> ""
    """
    if not file_ref:
        return ""
    # Strip trailing :line or :line-line
    idx = file_ref.rfind(":")
    if idx > 0:
        after_colon = file_ref[idx + 1:]
        # Only strip if the part after the colon looks like a line range
        if after_colon and all(c in "0123456789-" for c in after_colon):
            return file_ref[:idx]
    return file_ref


def _get_group_file_path(group: Dict[str, Any]) -> str:
    """Determine the primary file path for a group.

    Uses the first non-empty ``fileRef`` among the group's suggestions.
    Returns ``""`` if no suggestion has a ``fileRef``.
    """
    for suggestion in group.get("suggestions", []):
        file_ref = suggestion.get("fileRef", "")
        if file_ref:
            return _extract_file_path(file_ref)
    return ""


def _get_group_section_ref(group: Dict[str, Any]) -> str:
    """Determine the primary section reference for a group.

    Uses the first non-empty ``sectionRef`` among the group's suggestions.
    Returns ``""`` if no suggestion has a ``sectionRef``.
    """
    for suggestion in group.get("suggestions", []):
        section_ref = suggestion.get("sectionRef", "")
        if section_ref:
            return section_ref
    return ""


def _get_group_task_ref(group: Dict[str, Any]) -> str:
    """Determine the primary task reference for a group.

    Uses the first non-empty ``sectionRef`` among the group's suggestions
    (task review stores the task ID in the ``sectionRef`` / ``reference``
    field, which maps to ``sectionRef`` in camelCase report data).
    Returns ``""`` if no suggestion has a ``sectionRef``.
    """
    for suggestion in group.get("suggestions", []):
        section_ref = suggestion.get("sectionRef", "")
        if section_ref:
            return section_ref
    return ""


def _collect_all_suggestions(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten all suggestions from a list of groups into a single list."""
    result = []
    for group in groups:
        result.extend(group.get("suggestions", []))
    return result


def _file_sort_key(file_path: str) -> tuple:
    """Sort key for file paths: directory first, then filename alphabetically.

    Sorts by (directory, filename) so files in the same directory are
    grouped together, and within each directory files are sorted
    alphabetically.  The ``_global`` key always sorts last.
    """
    if file_path == "_global":
        # Sort _global last
        return ("\xff", "\xff")
    parts = file_path.rsplit("/", 1)
    if len(parts) == 2:
        return (parts[0].lower(), parts[1].lower())
    # No directory component — sort at root level
    return ("", parts[0].lower())


def build_file_view(
    groups: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a file-based index from the flat groups list.

    Groups suggestions by ``fileRef`` file path for code-review PR view.
    Suggestions with no ``fileRef`` are placed under the ``_global`` key.

    Args:
        groups: List of processed group dicts (post ``_build_group_data()``).

    Returns:
        Dict mapping file paths (and ``_global``) to::

            {
                "suggestions": [group, ...],  # groups anchored to this file
                "suggestionCount": int,        # total individual suggestions
                "maxImportance": str,          # highest importance across all suggestions
            }

        Keys are sorted by directory then alphabetically, with ``_global`` last.
    """
    file_buckets: Dict[str, List[Dict[str, Any]]] = {}

    for group in groups:
        file_path = _get_group_file_path(group)
        key = file_path if file_path else "_global"
        file_buckets.setdefault(key, []).append(group)

    # Build the result dict with sorted keys
    sorted_keys = sorted(file_buckets.keys(), key=_file_sort_key)

    result: Dict[str, Any] = {}
    for key in sorted_keys:
        bucket_groups = file_buckets[key]
        all_suggestions = _collect_all_suggestions(bucket_groups)
        result[key] = {
            "suggestions": bucket_groups,
            "suggestionCount": len(all_suggestions),
            "maxImportance": compute_max_importance(all_suggestions),
        }

    # Ensure _global key exists even if empty
    if "_global" not in result:
        result["_global"] = {
            "suggestions": [],
            "suggestionCount": 0,
            "maxImportance": "",
        }

    return result


def build_section_view(
    groups: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a section-based index from the flat groups list.

    Groups suggestions by ``sectionRef`` for plan-review PR view.
    Groups with no ``sectionRef`` are placed under the ``_global`` key.

    Args:
        groups: List of processed group dicts (post ``_build_group_data()``).

    Returns:
        Dict mapping section references (and ``_global``) to::

            {
                "suggestions": [group, ...],  # groups anchored to this section
                "suggestionCount": int,        # total individual suggestions
                "maxImportance": str,          # highest importance across all suggestions
            }
    """
    section_buckets: Dict[str, List[Dict[str, Any]]] = {}

    for group in groups:
        section_ref = _get_group_section_ref(group)
        key = section_ref if section_ref else "_global"
        section_buckets.setdefault(key, []).append(group)

    result: Dict[str, Any] = {}
    for key, bucket_groups in section_buckets.items():
        all_suggestions = _collect_all_suggestions(bucket_groups)
        result[key] = {
            "suggestions": bucket_groups,
            "suggestionCount": len(all_suggestions),
            "maxImportance": compute_max_importance(all_suggestions),
        }

    return result


def build_task_view(
    groups: List[Dict[str, Any]],
    tasks_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a task-based index from the flat groups list.

    Groups suggestions by task ID (from the ``sectionRef`` / ``reference``
    field) for task-review PR view.  Suggestions whose reference is
    ``"Plan Coverage"`` are routed to the ``_coverageGaps`` bucket.
    Groups with no reference at all go to the ``_unanchored`` bucket.

    Args:
        groups: List of processed group dicts (post ``_build_group_data()``).
        tasks_metadata: Optional dict mapping task IDs (e.g. ``"T001"``)
            to task metadata dicts.  When ``None``, all ``taskMetadata``
            values in the result are ``None``.

    Returns:
        Dict mapping task IDs (and ``_coverageGaps``, ``_unanchored``) to::

            {
                "taskMetadata": {...} | None,
                "suggestions": [group, ...],
                "suggestionCount": int,
                "maxImportance": str,
            }

        The ``_coverageGaps`` entry contains groups whose reference is
        ``"Plan Coverage"``.  The ``_unanchored`` entry contains groups
        with no reference.
    """
    task_buckets: Dict[str, List[Dict[str, Any]]] = {}
    coverage_gaps: List[Dict[str, Any]] = []
    unanchored: List[Dict[str, Any]] = []

    for group in groups:
        task_ref = _get_group_task_ref(group)
        if task_ref == "Plan Coverage":
            coverage_gaps.append(group)
        elif task_ref:
            task_buckets.setdefault(task_ref, []).append(group)
        else:
            # Groups with no reference — unanchored findings
            unanchored.append(group)

    result: Dict[str, Any] = {}

    for task_id, bucket_groups in task_buckets.items():
        all_suggestions = _collect_all_suggestions(bucket_groups)
        task_meta = None
        if tasks_metadata is not None:
            task_meta = tasks_metadata.get(task_id, None)
        result[task_id] = {
            "taskMetadata": task_meta,
            "suggestions": bucket_groups,
            "suggestionCount": len(all_suggestions),
            "maxImportance": compute_max_importance(all_suggestions),
        }

    # Always include _coverageGaps
    coverage_suggestions = _collect_all_suggestions(coverage_gaps)
    result["_coverageGaps"] = {
        "suggestions": coverage_gaps,
        "suggestionCount": len(coverage_suggestions),
        "maxImportance": compute_max_importance(coverage_suggestions),
    }

    # Always include _unanchored (groups with no reference)
    unanchored_suggestions = _collect_all_suggestions(unanchored)
    result["_unanchored"] = {
        "suggestions": unanchored,
        "suggestionCount": len(unanchored_suggestions),
        "maxImportance": compute_max_importance(unanchored_suggestions),
    }

    return result


# ---------------------------------------------------------------------------
# Post-apply human-decision overlay
# ---------------------------------------------------------------------------

def _normalize_human_decision(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a state.json human-decision record for the report template.

    Maps the recorded shape (``decision`` plus ``batch_context`` markers) into
    the camelCase fields the template's ``getHumanDecisionInfo()`` reads,
    collapsing everything to a single ``outcome`` of approved / salvaged /
    skipped. A salvaged item is recorded with ``decision == "approved"`` and a
    ``claude_auto_decide_salvage`` marker, so the marker — not ``decision`` —
    determines the outcome.
    """
    bc = record.get("batch_context") or {}
    source = bc.get("decision_source") or bc.get("batch_action") or "user"
    decision = record.get("decision")
    is_salvage = (
        bc.get("decision_source") == "claude_auto_decide_salvage"
        or bc.get("batch_action") == "claude_auto_decide_salvage"
    )
    if is_salvage:
        outcome = "salvaged"
    elif decision == "approved":
        outcome = "approved"
    elif decision == "skipped":
        outcome = "skipped"
    else:
        outcome = decision or "decided"
    return {
        "outcome": outcome,
        "decision": decision,
        "claudeDecided": str(source).startswith("claude_auto_decide"),
        "source": source,
        "reason": record.get("reason", "") or "",
        "salvagedDescription": bc.get("salvaged_description", "") or "",
        "dropped": bc.get("dropped", "") or "",
        "importanceAtDecision": bc.get("importance_at_decision", "") or "",
    }


def _overlay_human_decisions(
    report_data: Dict[str, Any],
    human_decisions: Dict[str, Dict[str, Any]],
) -> int:
    """Attach normalized human decisions onto ``report_data`` groups in place.

    Decisions are keyed by group_id (== ``group_hash``), matched against each
    group's ``groupHash``. Returns the number of groups matched and stamps a
    ``humanDecisionsSummary`` onto ``report_data``.
    """
    summary = {
        "approved": 0,
        "salvaged": 0,
        "skipped": 0,
        "claudeDecided": 0,
        "total": 0,
    }
    matched = 0
    for group in report_data.get("groups", []):
        group_hash = group.get("groupHash")
        if not group_hash or group_hash not in human_decisions:
            continue
        info = _normalize_human_decision(human_decisions[group_hash])
        group["humanDecision"] = info
        matched += 1
        summary["total"] += 1
        if info["outcome"] in summary:
            summary[info["outcome"]] += 1
        if info["claudeDecided"]:
            summary["claudeDecided"] += 1
    report_data["humanDecisionsSummary"] = summary
    return matched


def _load_report_template(template_style: str) -> str:
    """Load the report template text for the given style ('pr' or 'flat')."""
    templates_dir = Path(__file__).parent.parent / "templates"
    if template_style == "pr":
        template_path = templates_dir / "pr_report_template.html"
        if not template_path.exists():
            template_path = templates_dir / "report_template.html"
    else:
        template_path = templates_dir / "report_template.html"
    if not template_path.exists():
        raise FileNotFoundError(f"Report template not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def regenerate_report_with_human_decisions(
    phase_dir: Path,
    human_decisions: Dict[str, Dict[str, Any]],
    template_style: Optional[str] = None,
) -> Optional[Path]:
    """Re-render ``{phase_dir}/report.html`` with human decisions overlaid.

    Loads the ``report_data.json`` sidecar (persisted at review time), overlays
    the supplied human decisions (keyed by ``group_hash``), and re-embeds into
    the template — no reconstruction of the original runtime inputs
    (diff_data, base_ref, models, ...).

    Returns the report path on success, or ``None`` if the sidecar is missing
    (e.g. a report generated before this feature existed) so callers can warn
    and continue rather than crash.
    """
    phase_dir = Path(phase_dir)
    sidecar = phase_dir / "report_data.json"
    if not sidecar.exists():
        return None
    report_data = json.loads(sidecar.read_text(encoding="utf-8"))
    _overlay_human_decisions(report_data, human_decisions or {})
    style = template_style or report_data.get("templateStyle", "pr")
    template = _load_report_template(style)
    html = _embed_report_data(report_data, template)
    report_path = phase_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    # Keep the sidecar in sync so a subsequent re-run overlays cleanly.
    try:
        sidecar.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    except OSError:
        pass
    return report_path


def _read_human_decisions_from_state(
    state_file: Path, apply_phase: str
) -> Dict[str, Dict[str, Any]]:
    """Read ``human_decisions_{apply_phase}`` from a state.json file."""
    state = json.loads(Path(state_file).read_text(encoding="utf-8"))
    return state.get(f"human_decisions_{apply_phase}", {}) or {}


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: overlay recorded human decisions onto an existing review report."""
    bootstrap_streams()
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Regenerate an HTML review report with human/Claude decisions overlaid."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)
    rd = sub.add_parser(
        "regenerate-decisions",
        help="Overlay recorded human decisions onto an existing review report.html",
    )
    rd.add_argument(
        "--phase-dir",
        required=True,
        help="Review phase dir containing report_data.json / report.html",
    )
    rd.add_argument(
        "--state-file",
        required=True,
        help="state.json holding human_decisions_{apply-phase}",
    )
    rd.add_argument(
        "--apply-phase",
        required=True,
        help=(
            "Apply phase name, e.g. apply-suggestions / apply-code-fixes / "
            "apply-task-suggestions"
        ),
    )
    rd.add_argument(
        "--template-style",
        default=None,
        help="Override template style (pr/flat); defaults to the original style",
    )
    args = parser.parse_args(argv)

    if args.command == "regenerate-decisions":
        phase_dir = Path(args.phase_dir)
        decisions = _read_human_decisions_from_state(
            Path(args.state_file), args.apply_phase
        )
        result = regenerate_report_with_human_decisions(
            phase_dir, decisions, args.template_style
        )
        if result is None:
            print(
                f"No report_data.json in {phase_dir}; cannot overlay decisions "
                "(report predates the decision-overlay feature). Skipping HTML "
                "update."
            )
            return 0
        print(
            f"HTML report regenerated with {len(decisions)} human decision(s): "
            f"{result}"
        )
        return 0
    return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
