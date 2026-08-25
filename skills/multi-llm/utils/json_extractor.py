"""Robust JSON extraction with multiple fallback strategies.

This module provides a robust way to extract JSON from text that may contain
extra content, duplicate JSON structures, or other issues that break simple
regex-based extraction.

The key insight is that JSON has balanced brackets. We find all complete JSON
structures by tracking bracket depth, handling strings correctly (not counting
brackets inside "..."), and returning each structure individually.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


def sanitize_model_name(model: str) -> str:
    """Sanitize model name for use in filenames.

    Replaces all non-alphanumeric characters except hyphen and underscore with underscore.

    Args:
        model: Model specification (e.g., 'cursor-agent:gpt-5.2-high')

    Returns:
        Sanitized string safe for filenames (e.g., 'cursor-agent_gpt-5_2-high')
    """
    return re.sub(r'[^a-zA-Z0-9\-_]', '_', model)


def build_unsanitize_map(phase_dir: str) -> Dict[str, str]:
    """Build a mapping from sanitized model names back to original specs.

    Reads ``.status.json`` in *phase_dir* and uses ``models_requested`` /
    ``models_completed`` to create a ``sanitize(original) -> original`` dict.

    Returns:
        Dict mapping sanitized names to original model specs.
        Empty dict if ``.status.json`` is missing or unreadable.
    """
    status_path = Path(phase_dir) / ".status.json"
    if not status_path.exists():
        return {}
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    mapping: Dict[str, str] = {}
    for key in ("models_requested", "models_completed"):
        for original in data.get(key, []):
            mapping[sanitize_model_name(original)] = original
    return mapping


def find_json_candidates(text: str, char_type: str = "array") -> List[str]:
    """Find all potential JSON structures using balanced bracket matching.

    Args:
        text: Text to search
        char_type: 'array' for [...], 'object' for {...}, 'both' for either

    Returns:
        List of candidate JSON strings, ordered by position in text
    """
    candidates: List[Tuple[int, str]] = []
    chars: List[Tuple[str, str]] = []

    if char_type in ("array", "both"):
        chars.append(("[", "]"))
    if char_type in ("object", "both"):
        chars.append(("{", "}"))

    for start_char, end_char in chars:
        i = 0
        while i < len(text):
            if text[i] == start_char:
                depth = 1
                in_string = False
                escape_next = False
                j = i + 1

                while j < len(text) and depth > 0:
                    c = text[j]
                    if escape_next:
                        escape_next = False
                    elif c == "\\" and in_string:
                        escape_next = True
                    elif c == '"' and not escape_next:
                        in_string = not in_string
                    elif not in_string:
                        if c == start_char:
                            depth += 1
                        elif c == end_char:
                            depth -= 1
                    j += 1

                if depth == 0:
                    candidates.append((i, text[i:j]))
            i += 1

    # Sort by position, return just the strings
    candidates.sort(key=lambda x: x[0])
    return [c[1] for c in candidates]


def _candidate_spans(text: str, char_type: str) -> List[Tuple[int, int, str]]:
    """Recover each candidate's ``(start, end, text)`` span, in position order.

    :func:`find_json_candidates` returns the candidate strings only, but the
    last-document rule has to know which candidates sit *inside* another one —
    reversing a flat list would return the inner ``[1, 2]`` of ``[{"a": [1, 2]}]``.
    At most one balanced run begins at any index, so candidate starts are
    strictly increasing and a forward-only search recovers every span without
    a second scan of the text.
    """
    spans: List[Tuple[int, int, str]] = []
    pos = 0
    for candidate in find_json_candidates(text, char_type):
        start = text.find(candidate, pos)
        if start < 0:  # pragma: no cover - defensive; find() cannot miss here
            continue
        spans.append((start, start + len(candidate), candidate))
        pos = start + 1
    return spans


def extract_json_from_text(text: str, prefer_arrays: bool = True) -> Dict[str, Any]:
    """Extract JSON with multiple fallback strategies.

    **The last document wins, at every level** — the last envelope record (the
    adapters' own rule), then the last top-level candidate inside that record's
    text — with shape preference consulted before position, and a contained
    candidate never outranking the one that contains it. Models routinely emit a
    draft or a decoy before the real answer in one turn; the answer is the one
    that comes last.

    Four clauses make that concrete:

    1. **Top-level first.** A candidate contained in another candidate's span is
       ranked *after* every top-level one rather than dropped, so an inner
       structure is still salvageable when the outer one does not parse.
    2. **Shape before position.** ``prefer_arrays`` still tries every array
       before every object; last-wins applies *within* each group, so a trailing
       illustrative object never displaces a preceding array.
    3. **Non-empty before empty.** The last candidate that parses to a non-empty
       value, else the last that parses at all.
    4. **Code fences follow the same rule.** The last fenced block that parses
       to a non-empty value wins, else the last that parses at all, so the
       fenced path and the bare-text path are on one rule.

    Strategies in order:
    1. Code blocks (```json ... ```), last non-empty parseable one
    2. Balanced bracket matching - find all candidates, ranked by the rule above
    3. Try parsing each candidate in rank order

    Args:
        text: Text that may contain JSON
        prefer_arrays: If True, try array candidates before objects

    Returns:
        Dict with 'success', 'data'/'error', optionally 'raw'
    """
    text = text.strip()

    # Strategy 1: Code blocks - try both json-tagged and untagged.
    # Clause 4: the LAST fenced block that parses wins - and clause 3 binds the
    # fenced path too, so a trailing empty fence never displaces an earlier
    # non-empty one. Walk the fences in reverse, keep the first empty document
    # seen as the fallback, and serve it only when no fence carries a value.
    fence_matches = list(
        re.finditer(r"```(?:json)?\s*([\[\{][\s\S]*?[\]\}])\s*```", text)
    )
    empty_fence = None
    for match in reversed(fence_matches):
        try:
            fence_data = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if fence_data:
            return {"success": True, "data": fence_data}
        if empty_fence is None:
            empty_fence = fence_data
    if empty_fence is not None:
        return {"success": True, "data": empty_fence}

    # Strategy 2: Balanced bracket extraction.
    # Clause 2: shape before position - every array is tried before every
    # object when prefer_arrays is set, and last-wins applies within a group.
    if prefer_arrays:
        groups = [_candidate_spans(text, "array"), _candidate_spans(text, "object")]
    else:
        groups = [_candidate_spans(text, "both")]

    # A candidate sits inside another one iff some earlier-starting candidate
    # ends at or after it — balanced runs nest, they never partially overlap —
    # so one pass over the position-sorted union settles containment across both
    # groups without comparing every pair.
    contained_spans = set()
    furthest_end = -1
    for start, end, _ in sorted(span for group in groups for span in group):
        if end <= furthest_end:
            contained_spans.add((start, end))
        else:
            furthest_end = end

    # Clause 1: within each group, every top-level candidate ranks ahead of
    # every contained one; within each tier the LAST by position ranks first.
    candidates: List[str] = []
    for group in groups:
        for contained in (False, True):
            candidates.extend(
                span[2]
                for span in sorted(group, key=lambda s: s[0], reverse=True)
                if ((span[0], span[1]) in contained_spans) is contained
            )

    # Strategy 3: Try parsing each candidate in rank order.
    # Clause 3: the highest-ranked candidate that parses to a NON-EMPTY value,
    # else the highest-ranked that parses at all.
    empty_fallback = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if data:
            return {"success": True, "data": data}
        if empty_fallback is None:
            empty_fallback = data
    if empty_fallback is not None:
        return {"success": True, "data": empty_fallback}

    return {
        "success": False,
        "error": "No valid JSON found in text",
        "raw": text,
        "data": None,
    }


#: ANSI colour/OSC escapes, removed before anything goes looking for a bracket.
#: A provider CLI paints its own diagnostics — kilocode's dead-account failure
#: arrives as ``\x1b[91m\x1b[1mError: \x1b[0mUser not found.`` — and every one of
#: those ``\x1b[`` sequences is an opening bracket with no JSON behind it. This
#: is a second copy of the pattern ``providers/cursor_agent`` keeps for reading
#: *its* human-readable output, and deliberately so: ``json_extractor`` is below
#: the provider layer and importing upward would invert the dependency.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")

#: A JSON structure that was **started**: ``[`` or ``{`` followed by a token
#: only JSON puts in that position — a string, a nested structure, or an
#: unquoted key (mechanical damage salvage exists to repair). Truncation is why
#: balanced matching alone is not enough: ``[{"title": "x"`` never closes, so
#: :func:`find_json_candidates` cannot see it, and a half-written array is
#: precisely what salvage is for. Requiring that token is what keeps ``[ERROR]``,
#: ``[91m`` and ``[docs]`` — brackets from log levels, escapes and prose — out.
_JSON_STRUCTURE_OPENING = re.compile(r'[\[{]\s*(?:["\[{]|[A-Za-z_][A-Za-z0-9_]*\s*:)')


def has_salvageable_json(text: str) -> bool:
    """True when *text* still holds a model answer a salvage pass could repair.

    Salvage repairs *mechanical* damage to an answer the model did produce —
    truncation, a markdown fence, prose around the array, a trailing comma. It
    cannot conjure an answer that was never produced, and its prompt ends "if
    nothing recoverable is present, print an empty {type}"
    (``subagent_prompts.build_salvage_prompt``). So dispatching salvage at output
    with no JSON in it at all buys exactly one thing: an empty array written to
    that model's result file, where reaggregation reads it as "this model
    reviewed the diff and found nothing". A hard credential failure
    (``Error: User not found.`` on a non-zero exit) is that shape, and it is also
    the case where a ``[SALVAGE_NEEDED]`` marker sends the operator to a JSON
    repair when the fix is the provider's auth.

    Two ways text qualifies, in order:

    1. **A balanced candidate that parses.** A fenced array, an array with prose
       either side of it, an object where an array was wanted, or the inner
       object of an array truncated after it — :func:`find_json_candidates`
       finds all of them, and parsing is what separates a real structure from
       ``[ERROR]``.
    2. **Failing that, a structure that was started.** Clause 1 cannot see a
       truncated document, because an unclosed bracket is not a candidate.

    The expected type is deliberately not a parameter: a model that wrapped the
    array in an object (``{"suggestions": [ …``) is the ordinary salvage case,
    not a second failure mode, so both shapes count.

    Args:
        text: Raw model output — whichever stream the failure left behind.

    Returns:
        True when a salvage subagent would have something to work with. False
        means this is not a parse failure at all, and the caller must fail the
        model cleanly instead (the D6 row-3 partial-results path already covers
        a model that produced nothing).
    """
    body = _ANSI_ESCAPE.sub("", text or "")
    if not body.strip():
        return False

    for candidate in find_json_candidates(body, "both"):
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return True

    return _JSON_STRUCTURE_OPENING.search(body) is not None


def generate_output_path(
    out_dir: Union[str, Path],
    prefix: str,
    phase: str,
    model_spec: str
) -> Path:
    """Generate output file path for JSON results in phase subdirectory.

    Creates a path for LLM JSON output using the pattern:
    {out_dir}/{phase_dir}/{model}.json

    Phase mapping (internal to kebab-case):
    - plan_review -> review-plan
    - code_review -> code-review

    Args:
        out_dir: Base output directory (plan's output folder)
        prefix: File prefix (unused in new structure, kept for compatibility)
        phase: Operation phase (e.g., 'code_review', 'plan_review')
        model_spec: Model specification (e.g., 'cursor-agent:auto')

    Returns:
        Path object for the output file
    """
    out_dir = Path(out_dir)

    # Map internal phase names to kebab-case directory names
    phase_dir_map = {
        'plan_review': 'review-plan',
        'code_review': 'code-review',
        'task_review': 'review-tasks',
    }
    phase_dir_name = phase_dir_map.get(phase, phase.replace('_', '-'))

    # Create phase subdirectory
    phase_dir = out_dir / phase_dir_name
    phase_dir.mkdir(parents=True, exist_ok=True)

    safe_model = sanitize_model_name(model_spec)
    filename = f"{safe_model}.json"

    return phase_dir / filename


def read_json_from_file(
    file_path: Union[str, Path],
    prefer_arrays: bool = True
) -> Dict[str, Any]:
    """Read and parse JSON from a file with fallback extraction.

    Attempts to read and parse JSON from a file. If direct parsing fails,
    falls back to using extract_json_from_text() to handle code blocks,
    extra content, or other formatting issues.

    Args:
        file_path: Path to the JSON file
        prefer_arrays: If True, prefer array candidates over objects

    Returns:
        Dict with keys:
        - 'success': bool indicating if JSON was successfully parsed
        - 'data': Parsed JSON data (if successful)
        - 'error': Error message (if failed)
        - 'source': Where the JSON came from ('file', 'file_extracted', 'missing', 'error', 'empty')
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return {
            "success": False,
            "error": f"File not found: {file_path}",
            "source": "missing",
            "data": None,
        }

    try:
        content = file_path.read_text(encoding="utf-8").strip()
    except (IOError, OSError, UnicodeDecodeError) as e:
        return {
            "success": False,
            "error": f"Read error: {type(e).__name__}: {e}",
            "source": "error",
            "data": None,
        }

    if not content:
        return {
            "success": False,
            "error": "File is empty",
            "source": "empty",
            "data": None,
        }

    # Try direct JSON parse first
    try:
        data = json.loads(content)
        return {"success": True, "data": data, "source": "file"}
    except json.JSONDecodeError:
        pass

    # Fall back to extraction (handles code blocks, etc.)
    result = extract_json_from_text(content, prefer_arrays=prefer_arrays)
    if result.get("success"):
        result["source"] = "file_extracted"
    else:
        result["source"] = "file_extraction_failed"
    return result
