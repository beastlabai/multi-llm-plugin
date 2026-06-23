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


def extract_json_from_text(text: str, prefer_arrays: bool = True) -> Dict[str, Any]:
    """Extract JSON with multiple fallback strategies.

    Strategies in order:
    1. Code blocks (```json ... ```)
    2. Balanced bracket matching - find all candidates
    3. Try parsing each candidate, return first success

    Args:
        text: Text that may contain JSON
        prefer_arrays: If True, try array candidates before objects

    Returns:
        Dict with 'success', 'data'/'error', optionally 'raw'
    """
    text = text.strip()

    # Strategy 1: Code blocks - try both json-tagged and untagged
    code_block_match = re.search(
        r"```(?:json)?\s*([\[\{][\s\S]*?[\]\}])\s*```", text
    )
    if code_block_match:
        try:
            return {"success": True, "data": json.loads(code_block_match.group(1))}
        except json.JSONDecodeError:
            pass

    # Strategy 2: Balanced bracket extraction
    if prefer_arrays:
        # Try arrays first, then objects
        candidates = find_json_candidates(text, "array")
        candidates.extend(find_json_candidates(text, "object"))
    else:
        candidates = find_json_candidates(text, "both")

    # Strategy 3: Try parsing each candidate
    for idx, candidate in enumerate(candidates):
        try:
            data = json.loads(candidate)
            # Prefer non-empty results if there are more candidates
            remaining = candidates[idx + 1 :]
            if data or not remaining:
                return {"success": True, "data": data}
        except json.JSONDecodeError:
            continue

    return {
        "success": False,
        "error": "No valid JSON found in text",
        "raw": text,
        "data": None,
    }


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
