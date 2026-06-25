"""Parse report.md to extract user skip decisions and descriptions."""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Regex for extracting display hashes from bracket notation in report headers.
# Matches 8-16 char lowercase hex strings inside square brackets.
_HASH_RE = re.compile(r'\[([0-9a-f]{8,16})\]')


def _load_stamped_groups_for_legacy(report_path: str) -> Optional[List[Dict]]:
    """Load grouped.json from the same directory as report_path and stamp stable IDs.

    Used as a legacy fallback when report.md lacks bracket-notation hashes.
    Returns stamped groups list, or None if grouped.json is unavailable.
    """
    try:
        from .state_manager import stamp_stable_ids, load_groups_payload
    except ImportError:
        try:
            from utils.state_manager import stamp_stable_ids, load_groups_payload
        except ImportError:
            return None

    report_dir = Path(report_path).parent
    grouped_path = report_dir / "grouped.json"
    if not grouped_path.exists():
        return None

    try:
        raw = json.loads(grouped_path.read_text(encoding='utf-8'))
        groups = load_groups_payload(raw)
        stamp_stable_ids(groups)
        return groups
    except (json.JSONDecodeError, ValueError, OSError) as e:
        print(
            f"WARNING: Failed to load grouped.json for legacy fallback: {e}",
            file=sys.stderr,
        )
        return None


def _build_positional_maps(
    groups: List[Dict],
) -> Tuple[Dict[int, str], Dict[str, str]]:
    """Build positional-to-hash maps from stamped groups.

    Returns:
        Tuple of:
        - group_index_to_hash: {1-based index -> group_hash}
        - suggestion_id_to_hash: {"G{N}S{M}" -> suggestion_hash}
    """
    group_idx_to_hash: Dict[int, str] = {}
    sugg_id_to_hash: Dict[str, str] = {}

    for g_idx, group in enumerate(groups):
        g_num = g_idx + 1
        ghash = group.get("group_hash", "")
        if ghash:
            group_idx_to_hash[g_num] = ghash

        suggestions = group.get("suggestions", group.get("issues", []))
        for s_idx, sugg in enumerate(suggestions):
            s_num = s_idx + 1
            shash = sugg.get("suggestion_hash", "")
            if shash:
                sugg_id_to_hash[f"G{g_num}S{s_num}"] = shash

    return group_idx_to_hash, sugg_id_to_hash


def parse_skipped_suggestions(report_path: str) -> Set[str]:
    """
    Parse report.md and return IDs of suggestions marked to skip.

    Looks for pattern:
        ### S001: Title
        - [x/etc] Skip

    Returns:
        Set of suggestion IDs marked with [x/etc] Skip (e.g., {"S001", "S003"})
    """
    if not Path(report_path).exists():
        return set()

    content = Path(report_path).read_text(encoding='utf-8')
    skipped = set()

    # Pattern: ### S001: ... followed by - [x/etc] Skip (case insensitive)
    # Allow any non-blank content inside checkbox brackets
    pattern = r'###\s+(S\d+):[^\n]*\n-\s*\[\s*[^\]\s]+\s*\]\s*skip'

    for match in re.finditer(pattern, content, re.IGNORECASE):
        suggestion_id = match.group(1)
        skipped.add(suggestion_id)

    return skipped


def parse_skipped_issues(report_path: str) -> Set[int]:
    """
    Parse code review report.md and return indices of issues marked to skip.

    Code review reports use numeric indices like:
        ### 1. Title
        - [x/etc] Skip

    Returns:
        Set of issue indices marked with [x/etc] Skip (e.g., {1, 3})
    """
    if not Path(report_path).exists():
        return set()

    content = Path(report_path).read_text(encoding='utf-8')
    skipped = set()

    # Pattern: ### 1. ... followed by - [x/etc] Skip (case insensitive)
    # Allow any non-blank content inside checkbox brackets
    pattern = r'###\s+(\d+)\.[^\n]*\n-\s*\[\s*[^\]\s]+\s*\]\s*skip'

    for match in re.finditer(pattern, content, re.IGNORECASE):
        issue_index = int(match.group(1))
        skipped.add(issue_index)

    return skipped


def parse_skipped_groups(report_path: str) -> Set[str]:
    """
    Parse report.md and return group hashes of groups marked to skip.

    Supports two header formats:
        v2: ## G1 [a1b2c3d4]: Theme  (hash extracted from brackets)
        v1: ## G1: Theme              (legacy -- mapped via grouped.json)

    Returns:
        Set of group hash strings marked with [x/etc] Skip this group
    """
    if not Path(report_path).exists():
        return set()

    content = Path(report_path).read_text(encoding='utf-8')
    skipped_hashes: Set[str] = set()
    skipped_indices: Set[int] = set()
    found_bracket_notation = False

    # Pattern: ## G{N} [{hash}]: ... or ## G{N}: ... followed by skip checkbox
    # The hash bracket is optional -- we detect format by checking whether any
    # matches contain it.
    pattern = (
        r'##\s+G(\d+)'
        r'(?:\s+\[([0-9a-f]{8,16})\])?'
        r':[^\n]*\n+\s*-\s*\[\s*[^\]\s]+\s*\]\s*skip\s+this\s+group'
    )

    for match in re.finditer(pattern, content, re.IGNORECASE):
        hash_val = match.group(2)
        if hash_val:
            found_bracket_notation = True
            skipped_hashes.add(hash_val)
        else:
            skipped_indices.add(int(match.group(1)))

    # Map any positional indices to hashes via grouped.json
    if skipped_indices:
        groups = _load_stamped_groups_for_legacy(report_path)
        if groups is not None:
            group_idx_to_hash, _ = _build_positional_maps(groups)
            for idx in skipped_indices:
                ghash = group_idx_to_hash.get(idx)
                if ghash:
                    skipped_hashes.add(ghash)
                else:
                    print(
                        f"WARNING: Skipped group index G{idx} could not be "
                        f"mapped to a hash (out of range). Ignoring.",
                        file=sys.stderr,
                    )
        else:
            if not found_bracket_notation:
                print(
                    "WARNING: Report uses legacy G{N} format but grouped.json "
                    "not found for hash migration. Skipped groups may not apply.",
                    file=sys.stderr,
                )

    return skipped_hashes


def parse_skipped_group_suggestions(report_path: str) -> Set[str]:
    """
    Parse report.md and return hashes of individual suggestions marked to skip.

    Supports two header formats:
        v2: ### G1S2 [e5f6a7b8]: Title  (hash extracted from brackets)
        v1: ### G1S2: Title              (legacy -- mapped via grouped.json)

    Returns:
        Set of suggestion hash strings marked with [x/etc] Skip
    """
    if not Path(report_path).exists():
        return set()

    content = Path(report_path).read_text(encoding='utf-8')
    skipped_hashes: Set[str] = set()
    skipped_positional: Set[str] = set()
    found_bracket_notation = False

    # Pattern: ### G{N}S{M} [{hash}]: ... or ### G{N}S{M}: ...
    # followed by skip checkbox (but NOT "Skip this group")
    pattern = (
        r'###\s+(G\d+S\d+)'
        r'(?:\s+\[([0-9a-f]{8,16})\])?'
        r':[^\n]*\n-\s*\[\s*[^\]\s]+\s*\]\s*skip\b(?!\s+this\s+group)'
    )

    for match in re.finditer(pattern, content, re.IGNORECASE):
        hash_val = match.group(2)
        if hash_val:
            found_bracket_notation = True
            skipped_hashes.add(hash_val)
        else:
            skipped_positional.add(match.group(1).upper())

    # Map any positional IDs to hashes via grouped.json
    if skipped_positional:
        groups = _load_stamped_groups_for_legacy(report_path)
        if groups is not None:
            _, sugg_id_to_hash = _build_positional_maps(groups)
            for sid in skipped_positional:
                shash = sugg_id_to_hash.get(sid)
                if shash:
                    skipped_hashes.add(shash)
                else:
                    print(
                        f"WARNING: Skipped suggestion {sid} could not be "
                        f"mapped to a hash. Ignoring.",
                        file=sys.stderr,
                    )
        else:
            if not found_bracket_notation:
                print(
                    "WARNING: Report uses legacy G{N}S{M} format but "
                    "grouped.json not found for hash migration. "
                    "Skipped suggestions may not apply.",
                    file=sys.stderr,
                )

    return skipped_hashes


def normalize_description(desc: str) -> str:
    """Normalize description for comparison (strip whitespace, remove validation blockquotes).

    - Strip leading/trailing whitespace
    - Collapse multiple spaces to single space
    - Normalize CRLF to LF
    - Remove validation reason blockquotes that start with `> **Validation Reason:**`
      (including continuation lines starting with `>`)
    - Preserve other blockquotes and markdown formatting
    """
    if not desc:
        return ""

    # Normalize CRLF to LF
    text = desc.replace('\r\n', '\n')

    # Remove validation reason blockquotes (> **Validation Reason:** and continuation lines)
    # Pattern matches lines starting with > **Validation Reason:** and subsequent > lines
    lines = text.split('\n')
    result_lines = []
    in_validation_blockquote = False

    for line in lines:
        stripped = line.strip()

        # Check if this line starts a validation reason blockquote
        if stripped.startswith('> **Validation Reason:**'):
            in_validation_blockquote = True
            continue

        # Check if we're in a validation blockquote and this is a continuation
        if in_validation_blockquote:
            if stripped.startswith('>'):
                # Still in the validation blockquote, skip this line
                continue
            else:
                # No longer in the validation blockquote
                in_validation_blockquote = False

        result_lines.append(line)

    text = '\n'.join(result_lines)

    # Collapse multiple spaces to single space (but preserve newlines)
    text = re.sub(r' +', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def parse_suggestion_descriptions(report_path: str) -> Dict[str, str]:
    """Extract descriptions from report.md keyed by suggestion hash.

    Supports two header formats:
        v2: ### G1S1 [e5f6a7b8]: Title  (hash extracted from brackets)
        v1: ### G1S1: Title              (legacy -- mapped via grouped.json)

    Extracts the description text that appears AFTER the metadata line
    (``**Validation:**...`` or ``**Importance:**...``) and BEFORE the
    ``---`` separator.

    Returns:
        Dict mapping suggestion hash to description text.
        Empty dict if file not found.
    """
    if not Path(report_path).exists():
        return {}

    content = Path(report_path).read_text(encoding='utf-8')
    descriptions_by_hash: Dict[str, str] = {}
    descriptions_by_positional: Dict[str, str] = {}
    found_bracket_notation = False

    # Pattern to find suggestion headers and capture everything until ---
    # Header format: ### G1S1 [hash]: Title  OR  ### G1S1: Title
    pattern = (
        r'###\s+(G\d+S\d+)'
        r'(?:\s+\[([0-9a-f]{8,16})\])?'
        r':[^\n]*\n(.*?)(?=\n---|\n###|\Z)'
    )

    for match in re.finditer(pattern, content, re.DOTALL):
        positional_id = match.group(1)
        hash_val = match.group(2)
        block = match.group(3)

        # Find the metadata line and extract text after it
        # Metadata starts with **Validation:** or **Importance:**
        metadata_pattern = r'\*\*(?:Validation|Importance):\*\*[^\n]*\n'
        metadata_match = re.search(metadata_pattern, block)

        if metadata_match:
            # Description is everything after the metadata line
            desc_start = metadata_match.end()
            description = block[desc_start:]
        else:
            # No metadata line found, use the whole block after the checkbox line
            # Skip the checkbox line (- [ ] Skip or - [x] Skip)
            checkbox_pattern = r'-\s*\[[^\]]*\]\s*[Ss]kip[^\n]*\n'
            checkbox_match = re.search(checkbox_pattern, block)
            if checkbox_match:
                description = block[checkbox_match.end():]
            else:
                description = block

        # Clean up the description
        description = description.strip()

        if description:
            if hash_val:
                found_bracket_notation = True
                descriptions_by_hash[hash_val] = description
            else:
                descriptions_by_positional[positional_id.upper()] = description

    # Map positional IDs to hashes via grouped.json fallback
    if descriptions_by_positional:
        groups = _load_stamped_groups_for_legacy(report_path)
        if groups is not None:
            _, sugg_id_to_hash = _build_positional_maps(groups)
            for sid, desc in descriptions_by_positional.items():
                shash = sugg_id_to_hash.get(sid)
                if shash:
                    # Don't overwrite a hash-based entry
                    if shash not in descriptions_by_hash:
                        descriptions_by_hash[shash] = desc
                else:
                    print(
                        f"WARNING: Description for {sid} could not be "
                        f"mapped to a hash. Ignoring.",
                        file=sys.stderr,
                    )
        else:
            if not found_bracket_notation:
                # Fallback: return positional IDs as-is so callers can still
                # use them (preserves backward compatibility when grouped.json
                # is unavailable)
                return descriptions_by_positional

    return descriptions_by_hash


def parse_issue_descriptions(report_path: str) -> Dict[int, str]:
    """Extract descriptions from code review report.md by issue index (1, 2, 3...).

    Looks for headers like `### 1. Title` or `### 2. Title`.
    Extracts the description text that appears AFTER the metadata line
    (`**Validation:**...` or `**Importance:**...`) and BEFORE the `---` separator.

    Returns:
        Dict mapping issue index to description text (e.g., {1: "Description..."})
        Empty dict if file not found.
    """
    if not Path(report_path).exists():
        return {}

    content = Path(report_path).read_text(encoding='utf-8')
    descriptions = {}

    # Pattern to find issue headers and capture everything until ---
    # Header format: ### 1. Title
    pattern = r'###\s+(\d+)\.[^\n]*\n(.*?)(?=\n---|\n###|\Z)'

    for match in re.finditer(pattern, content, re.DOTALL):
        issue_index = int(match.group(1))
        block = match.group(2)

        # Find the metadata line and extract text after it
        # Metadata starts with **Validation:** or **Importance:**
        metadata_pattern = r'\*\*(?:Validation|Importance):\*\*[^\n]*\n'
        metadata_match = re.search(metadata_pattern, block)

        if metadata_match:
            # Description is everything after the metadata line
            desc_start = metadata_match.end()
            description = block[desc_start:]
        else:
            # No metadata line found, use the whole block after the checkbox line
            # Skip the checkbox line (- [ ] Skip or - [x] Skip)
            checkbox_pattern = r'-\s*\[[^\]]*\]\s*[Ss]kip[^\n]*\n'
            checkbox_match = re.search(checkbox_pattern, block)
            if checkbox_match:
                description = block[checkbox_match.end():]
            else:
                description = block

        # Clean up the description
        description = description.strip()

        if description:
            descriptions[issue_index] = description

    return descriptions


def find_edited_descriptions(
    report_path: str, grouped: List[Dict]
) -> Dict[str, Tuple[str, str]]:
    """Compare report.md descriptions with grouped.json, return edited items.

    Args:
        report_path: Path to report.md file
        grouped: List of group dicts from grouped.json with structure:
                 [{theme: str, suggestions: [{title, desc, ...}]}, ...]

    Returns:
        Dict mapping suggestion hash to (original_desc, edited_desc) for items
        where normalized descriptions differ. Keys are suggestion_hash values
        (hex strings). If suggestion_hash is not available on a suggestion,
        falls back to the positional G{N}S{M} ID.
    """
    report_descriptions = parse_suggestion_descriptions(report_path)

    if not report_descriptions:
        return {}

    # Ensure groups are stamped so suggestion_hash is available
    try:
        from .state_manager import stamp_stable_ids
    except ImportError:
        try:
            from utils.state_manager import stamp_stable_ids
        except ImportError:
            stamp_stable_ids = None

    if stamp_stable_ids is not None:
        stamp_stable_ids(grouped)  # idempotent

    edited: Dict[str, Tuple[str, str]] = {}

    for group_idx, group in enumerate(grouped, start=1):
        suggestions = group.get('suggestions', [])
        for sugg_idx, suggestion in enumerate(suggestions, start=1):
            # Use suggestion_hash as the key; fall back to positional ID
            sugg_hash = suggestion.get('suggestion_hash', '')
            positional_id = f"G{group_idx}S{sugg_idx}"
            original_desc = suggestion.get('desc', '')

            # Check both hash-based and positional keys in report descriptions
            report_desc = None
            key_used = None
            if sugg_hash and sugg_hash in report_descriptions:
                report_desc = report_descriptions[sugg_hash]
                key_used = sugg_hash
            elif positional_id in report_descriptions:
                report_desc = report_descriptions[positional_id]
                key_used = sugg_hash if sugg_hash else positional_id

            if report_desc is not None and key_used:
                # Compare normalized descriptions
                if normalize_description(original_desc) != normalize_description(
                    report_desc
                ):
                    edited[key_used] = (original_desc, report_desc)

    return edited


def load_html_selections(
    phase_dir: Path,
    groups: Optional[list] = None,
    plan_path: Optional[str] = None,
) -> Optional[Dict]:
    """Load user_selections.json if present in phase directory.

    Detects format_version and migrates v1 to v2 if needed.

    Args:
        phase_dir: Path to phase directory (e.g., plans/my-feature/review-plan/)
        groups: Optional list of groups/issues from grouped.json for v1 migration
            and index validation. If provided, warns about out-of-range references.
        plan_path: Optional absolute path to the plan file being processed.
            If provided and the JSON contains a different plan_path, raises ValueError.

    Returns:
        Parsed JSON dict if file exists and is valid, None otherwise.
        Expected v2 structure:
        {
            "format_version": 2,
            "plan_path": str,
            "phase": str,
            "exported_at": str (ISO timestamp),
            "skipped_groups": List[str],  # group hashes
            "skipped_suggestions": List[str],  # suggestion hashes
            "edited_descriptions": Dict[str, str]  # suggestion_hash -> new text
        }

    Raises:
        ValueError: If plan_path is provided and does not match the JSON's plan_path.
    """
    selections_path = phase_dir / "user_selections.json"

    try:
        content = selections_path.read_text(encoding='utf-8')
        data = json.loads(content)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None

    # Validate plan_path matches if provided
    if plan_path is not None:
        json_plan_path = data.get("plan_path")
        if json_plan_path is None:
            print(
                "WARNING: user_selections.json does not contain a plan_path "
                "field. Cannot verify selections match the current plan. "
                "This may be an export from an older version.",
                file=sys.stderr,
            )
        elif os.path.normpath(json_plan_path) != os.path.normpath(plan_path):
            raise ValueError(
                f"Plan path mismatch in user_selections.json!\n"
                f"  Selections were exported for: {json_plan_path}\n"
                f"  Current plan being processed:  {plan_path}\n"
                f"This likely means user_selections.json was copied from a "
                f"different plan's directory. Remove or replace the file "
                f"and retry."
            )

    # Detect format version and migrate v1 -> v2 if needed
    try:
        from .state_manager import get_format_version
    except ImportError:
        try:
            from utils.state_manager import get_format_version
        except ImportError:
            get_format_version = None

    if get_format_version is not None:
        version = get_format_version(data)
        if version < 2 and groups is not None:
            try:
                from .selection_migration import _migrate_v1_html_selections
            except ImportError:
                try:
                    from utils.selection_migration import _migrate_v1_html_selections
                except ImportError:
                    _migrate_v1_html_selections = None

            if _migrate_v1_html_selections is not None:
                data = _migrate_v1_html_selections(data, groups)

    # Validate references against groups if provided (post-migration, so
    # validation works on the v2 hash-based keys)
    if groups is not None:
        num_groups = len(groups)
        for idx in data.get("skipped_groups", []):
            if isinstance(idx, int) and idx > num_groups:
                print(
                    f"WARNING: user_selections.json references group/issue {idx} "
                    f"but only {num_groups} groups exist. This may indicate a "
                    f"numbering mismatch.",
                    file=sys.stderr,
                )
        for sid in data.get("skipped_suggestions", []):
            if isinstance(sid, str):
                m = re.match(r"G(\d+)", sid)
                if m and int(m.group(1)) > num_groups:
                    print(
                        f"WARNING: user_selections.json references {sid} "
                        f"but only {num_groups} groups exist.",
                        file=sys.stderr,
                    )
        for key in data.get("validation_overrides", {}):
            key_str = str(key)
            m = re.match(r"G(\d+)", key_str)
            if m and int(m.group(1)) > num_groups:
                print(
                    f"WARNING: validation override for {key_str} references "
                    f"group beyond {num_groups} groups.",
                    file=sys.stderr,
                )

    return data


def merge_selections(
    html_selections: Optional[Dict],
    md_skipped_groups: Set[str],
    md_skipped_suggestions: Set[str],
    md_edited: Dict[str, Tuple[str, str]],
) -> Tuple[Set[str], Set[str], Dict[str, str]]:
    """Merge HTML and markdown selections additively.

    Skips are unioned: markdown skips form the base and HTML skips are
    added on top, so an empty HTML ``skipped_groups`` list never erases
    markdown skip decisions.  Edited descriptions use overlay semantics
    (HTML wins per-key, markdown preserved otherwise).

    All keys are hash-based strings (group hashes for skipped_groups,
    suggestion hashes for skipped_suggestions and edited_descriptions).

    Args:
        html_selections: Parsed user_selections.json or None
        md_skipped_groups: Group hashes marked skip in report.md
        md_skipped_suggestions: Suggestion hashes marked skip in report.md
        md_edited: Dict of {hash: (original, edited)} from markdown

    Returns:
        Tuple of:
        - merged_skipped_groups: Set[str]
        - merged_skipped_suggestions: Set[str]
        - merged_edited_descriptions: Dict[str, str] (hash -> new description)
    """
    # Convert md_edited from (original, edited) tuples to just edited text
    md_edited_descriptions = {k: v[1] for k, v in md_edited.items()}

    # If no HTML selections, return markdown values
    if html_selections is None:
        return (
            md_skipped_groups,
            md_skipped_suggestions,
            md_edited_descriptions,
        )

    # Markdown skips are the base; HTML/C-level skips are unioned on top
    merged_skipped_groups = set(md_skipped_groups) | set(
        html_selections.get('skipped_groups', [])
    )
    merged_skipped_suggestions = set(md_skipped_suggestions) | set(
        html_selections.get('skipped_suggestions', [])
    )

    # For edited descriptions: start with markdown, then overlay HTML edits
    merged_edited_descriptions = dict(md_edited_descriptions)
    html_edited = html_selections.get('edited_descriptions', {})
    merged_edited_descriptions.update(html_edited)

    return (
        merged_skipped_groups,
        merged_skipped_suggestions,
        merged_edited_descriptions,
    )


def parse_validation_overrides_groups(report_path: str) -> Dict[str, str]:
    """
    Parse report.md and return validation overrides for groups, keyed by group hash.

    Supports two header formats:
        v2: ## G1 [a1b2c3d4]: Theme  (hash extracted from brackets)
        v1: ## G1: Theme              (legacy -- mapped via grouped.json)

    If both "Mark valid" and "Mark invalid" are checked, "invalid" wins (safety-first).
    "Let Claude decide" maps to "claude_decide" (precedence: invalid > valid >
    claude_decide).

    Returns:
        Dict mapping group hash to override value ("valid", "invalid", or
        "claude_decide")
    """
    if not Path(report_path).exists():
        return {}

    content = Path(report_path).read_text(encoding='utf-8')
    overrides_by_hash: Dict[str, str] = {}
    overrides_by_index: Dict[int, str] = {}
    found_bracket_notation = False

    # Split content into group sections by ## G{n} [{hash}]: or ## G{n}: headers
    group_pattern = r'##\s+G(\d+)(?:\s+\[([0-9a-f]{8,16})\])?:[^\n]*'
    group_matches = list(re.finditer(group_pattern, content, re.IGNORECASE))

    for i, match in enumerate(group_matches):
        group_idx = int(match.group(1))
        hash_val = match.group(2)
        # Get the block between this header and the next ## or --- or end
        start = match.end()
        if i + 1 < len(group_matches):
            end = group_matches[i + 1].start()
        else:
            end = len(content)
        block = content[start:end]

        # Check for "Mark valid" checkbox (checked = any non-blank content in brackets)
        valid_checked = bool(re.search(
            r'-\s*\[\s*[^\]\s]+\s*\]\s*mark\s+valid',
            block, re.IGNORECASE
        ))
        # Check for "Mark invalid" checkbox
        invalid_checked = bool(re.search(
            r'-\s*\[\s*[^\]\s]+\s*\]\s*mark\s+invalid',
            block, re.IGNORECASE
        ))
        # Check for "Let Claude decide" checkbox (routing marker, not a status)
        claude_decide_checked = bool(re.search(
            r'-\s*\[\s*[^\]\s]+\s*\]\s*let\s+claude\s+decide',
            block, re.IGNORECASE
        ))

        # Precedence: invalid > valid > claude_decide (an explicit valid/invalid
        # is a firmer signal than "you decide").
        override_val = None
        if invalid_checked:
            override_val = "invalid"  # invalid wins if both checked
        elif valid_checked:
            override_val = "valid"
        elif claude_decide_checked:
            override_val = "claude_decide"

        if override_val:
            if hash_val:
                found_bracket_notation = True
                overrides_by_hash[hash_val] = override_val
            else:
                overrides_by_index[group_idx] = override_val

    # Map positional indices to hashes via grouped.json
    if overrides_by_index:
        groups = _load_stamped_groups_for_legacy(report_path)
        if groups is not None:
            group_idx_to_hash, _ = _build_positional_maps(groups)
            for idx, val in overrides_by_index.items():
                ghash = group_idx_to_hash.get(idx)
                if ghash:
                    if ghash not in overrides_by_hash:
                        overrides_by_hash[ghash] = val
                else:
                    print(
                        f"WARNING: Validation override for G{idx} could not be "
                        f"mapped to a hash. Ignoring.",
                        file=sys.stderr,
                    )
        else:
            if not found_bracket_notation:
                print(
                    "WARNING: Report uses legacy G{N} format but grouped.json "
                    "not found for hash migration. Validation overrides may "
                    "not apply.",
                    file=sys.stderr,
                )

    return overrides_by_hash


def parse_suggestion_validation_overrides(report_path: str) -> Dict[str, str]:
    """
    Parse report.md and return validation overrides for individual suggestions,
    keyed by suggestion hash.

    Supports two header formats:
        v2: ### G1S2 [e5f6a7b8]: Title  (hash extracted from brackets)
        v1: ### G1S2: Title              (legacy -- mapped via grouped.json)

    If both "Mark valid" and "Mark invalid" are checked, "invalid" wins (safety-first).
    "Let Claude decide" maps to "claude_decide" (precedence: invalid > valid >
    claude_decide).

    Returns:
        Dict mapping suggestion hash to override value ("valid", "invalid", or
        "claude_decide")
    """
    if not Path(report_path).exists():
        return {}

    content = Path(report_path).read_text(encoding='utf-8')
    overrides_by_hash: Dict[str, str] = {}
    overrides_by_positional: Dict[str, str] = {}
    found_bracket_notation = False

    # Split content by suggestion headers (### G{n}S{m} [{hash}]: ...)
    sugg_pattern = r'###\s+(G\d+S\d+)(?:\s+\[([0-9a-f]{8,16})\])?:[^\n]*'
    sugg_matches = list(re.finditer(sugg_pattern, content, re.IGNORECASE))

    for i, match in enumerate(sugg_matches):
        positional_id = match.group(1).upper()
        hash_val = match.group(2)
        start = match.end()
        if i + 1 < len(sugg_matches):
            end = sugg_matches[i + 1].start()
        else:
            # Look for next ## or --- as section boundary
            next_section = re.search(r'\n(?:##[^#]|---)', content[start:])
            end = start + next_section.start() if next_section else len(content)
        block = content[start:end]

        valid_checked = bool(re.search(
            r'-\s*\[\s*[^\]\s]+\s*\]\s*mark\s+valid',
            block, re.IGNORECASE
        ))
        invalid_checked = bool(re.search(
            r'-\s*\[\s*[^\]\s]+\s*\]\s*mark\s+invalid',
            block, re.IGNORECASE
        ))
        # Check for "Let Claude decide" checkbox (routing marker, not a status)
        claude_decide_checked = bool(re.search(
            r'-\s*\[\s*[^\]\s]+\s*\]\s*let\s+claude\s+decide',
            block, re.IGNORECASE
        ))

        # Precedence: invalid > valid > claude_decide.
        override_val = None
        if invalid_checked:
            override_val = "invalid"  # invalid wins if both checked
        elif valid_checked:
            override_val = "valid"
        elif claude_decide_checked:
            override_val = "claude_decide"

        if override_val:
            if hash_val:
                found_bracket_notation = True
                overrides_by_hash[hash_val] = override_val
            else:
                overrides_by_positional[positional_id] = override_val

    # Map positional IDs to hashes via grouped.json
    if overrides_by_positional:
        groups = _load_stamped_groups_for_legacy(report_path)
        if groups is not None:
            _, sugg_id_to_hash = _build_positional_maps(groups)
            for sid, val in overrides_by_positional.items():
                shash = sugg_id_to_hash.get(sid)
                if shash:
                    if shash not in overrides_by_hash:
                        overrides_by_hash[shash] = val
                else:
                    print(
                        f"WARNING: Validation override for {sid} could not be "
                        f"mapped to a hash. Ignoring.",
                        file=sys.stderr,
                    )
        else:
            if not found_bracket_notation:
                print(
                    "WARNING: Report uses legacy G{N}S{M} format but "
                    "grouped.json not found for hash migration. "
                    "Suggestion validation overrides may not apply.",
                    file=sys.stderr,
                )

    return overrides_by_hash


def parse_validation_overrides_issues(report_path: str) -> Dict[int, str]:
    """
    Parse code review report.md and return validation overrides for issues.

    Looks for pattern:
        ### 1. Title
        - [ ] Skip
        - [x] Mark valid
        - [ ] Mark invalid

    If both "Mark valid" and "Mark invalid" are checked, "invalid" wins (safety-first).
    "Let Claude decide" maps to "claude_decide" (precedence: invalid > valid >
    claude_decide).

    Returns:
        Dict mapping issue index (1-based) to override value ("valid",
        "invalid", or "claude_decide")
    """
    if not Path(report_path).exists():
        return {}

    content = Path(report_path).read_text(encoding='utf-8')
    overrides = {}

    # Split content into issue sections by ### {n}. headers
    issue_pattern = r'###\s+(\d+)\.[^\n]*'
    issue_matches = list(re.finditer(issue_pattern, content))

    for i, match in enumerate(issue_matches):
        issue_idx = int(match.group(1))
        # Get the block between this header and the next ### or ## or --- or end
        start = match.end()
        if i + 1 < len(issue_matches):
            end = issue_matches[i + 1].start()
        else:
            end = len(content)
        block = content[start:end]

        # Check for "Mark valid" checkbox (checked = any non-blank content in brackets)
        valid_checked = bool(re.search(
            r'-\s*\[\s*[^\]\s]+\s*\]\s*mark\s+valid',
            block, re.IGNORECASE
        ))
        # Check for "Mark invalid" checkbox
        invalid_checked = bool(re.search(
            r'-\s*\[\s*[^\]\s]+\s*\]\s*mark\s+invalid',
            block, re.IGNORECASE
        ))
        # Check for "Let Claude decide" checkbox (routing marker, not a status)
        claude_decide_checked = bool(re.search(
            r'-\s*\[\s*[^\]\s]+\s*\]\s*let\s+claude\s+decide',
            block, re.IGNORECASE
        ))

        # Precedence: invalid > valid > claude_decide.
        if invalid_checked:
            overrides[issue_idx] = "invalid"  # invalid wins if both checked
        elif valid_checked:
            overrides[issue_idx] = "valid"
        elif claude_decide_checked:
            overrides[issue_idx] = "claude_decide"

    return overrides


def parse_consolidated_skipped_groups(report_path: str) -> Set[str]:
    """
    Parse consolidated report.md and return consolidated_ids of groups marked to skip.

    Looks for pattern:
        ## CG1 [abc123def456]:
        - [x/etc] Skip this group

    Returns:
        Set of consolidated_id values (12-char hex) marked with [x/etc] Skip this group
    """
    if not Path(report_path).exists():
        return set()

    content = Path(report_path).read_text(encoding='utf-8')
    skipped = set()

    # Pattern: ## CG{N} [{consolidated_id}]: ... followed by - [x/etc] Skip this group
    # Allow any non-blank content inside checkbox brackets
    pattern = (
        r'##\s+CG\d+\s+\[([0-9a-fA-F]{12})\]:[^\n]*'
        r'\n+\s*-\s*\[\s*[^\]\s]+\s*\]\s*skip\s+this\s+group'
    )

    for match in re.finditer(pattern, content, re.IGNORECASE):
        consolidated_id = match.group(1)
        skipped.add(consolidated_id)

    return skipped


def parse_consolidated_validation_overrides(report_path: str) -> Dict[str, str]:
    """
    Parse consolidated report.md and return validation overrides for consolidated groups.

    Looks for pattern:
        ## CG1 [abc123def456]:
        - [ ] Skip this group
        - [x] Mark valid
        - [ ] Mark invalid
        - [ ] Needs human attention

    4-state priority: invalid > needs-human-decision > valid > claude_decide.

    Returns:
        Dict mapping consolidated_id (12-char hex) to override value
        ("valid", "invalid", "needs-human-decision", or "claude_decide")
    """
    if not Path(report_path).exists():
        return {}

    content = Path(report_path).read_text(encoding='utf-8')
    overrides = {}

    # Split content into consolidated group sections by ## CG{N} [{id}]: headers
    cg_pattern = r'##\s+CG\d+\s+\[([0-9a-fA-F]{12})\]:[^\n]*'
    cg_matches = list(re.finditer(cg_pattern, content))

    for i, match in enumerate(cg_matches):
        consolidated_id = match.group(1)
        # Get the block between this header and the next ## or end
        start = match.end()
        if i + 1 < len(cg_matches):
            end = cg_matches[i + 1].start()
        else:
            end = len(content)
        block = content[start:end]

        # Check for "Mark valid" checkbox (checked = any non-blank content in brackets)
        valid_checked = bool(re.search(
            r'-\s*\[\s*[^\]\s]+\s*\]\s*mark\s+valid',
            block, re.IGNORECASE
        ))
        # Check for "Mark invalid" checkbox
        invalid_checked = bool(re.search(
            r'-\s*\[\s*[^\]\s]+\s*\]\s*mark\s+invalid',
            block, re.IGNORECASE
        ))
        # Check for "Needs human attention" checkbox
        needs_human_checked = bool(re.search(
            r'-\s*\[\s*[^\]\s]+\s*\]\s*needs\s+human\s+attention',
            block, re.IGNORECASE
        ))
        # Check for "Let Claude decide" checkbox (routing marker, not a status)
        claude_decide_checked = bool(re.search(
            r'-\s*\[\s*[^\]\s]+\s*\]\s*let\s+claude\s+decide',
            block, re.IGNORECASE
        ))

        # 4-state priority: invalid > needs-human-decision > valid >
        # claude_decide (an explicit invalid/needs-human/valid is a firmer
        # signal than "you decide").
        if invalid_checked:
            overrides[consolidated_id] = "invalid"
        elif needs_human_checked:
            overrides[consolidated_id] = "needs-human-decision"
        elif valid_checked:
            overrides[consolidated_id] = "valid"
        elif claude_decide_checked:
            overrides[consolidated_id] = "claude_decide"

    return overrides


def load_consolidated_html_selections(
    phase_dir: Path,
    groups: Optional[List] = None,
    plan_path: Optional[str] = None,
) -> Optional[Dict]:
    """Load consolidated_user_selections.json if present in phase directory.

    Detects format_version and migrates v1 to v2 if needed.

    Args:
        phase_dir: Path to phase directory (e.g., plans/my-feature/review-plan/)
        groups: Optional list of groups for v1 migration.
        plan_path: Optional absolute path to the plan file being processed.
            If provided and the JSON contains a different plan_path, raises ValueError.

    Returns:
        Parsed JSON dict if file exists and is valid, None otherwise.
        Expected structure:
        {
            "plan_path": str,
            "phase": str,
            "exported_at": str (ISO timestamp),
            "skipped_groups": List[str],  # consolidated_ids (12-char hex)
            "validation_overrides": Dict[str, str]  # consolidated_id -> override value
        }

    Raises:
        ValueError: If plan_path is provided and does not match the JSON's plan_path.
    """
    selections_path = phase_dir / "consolidated_user_selections.json"

    try:
        content = selections_path.read_text(encoding='utf-8')
        data = json.loads(content)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None

    # Validate plan_path matches if provided
    if plan_path is not None:
        json_plan_path = data.get("plan_path")
        if json_plan_path is None:
            print(
                "WARNING: consolidated_user_selections.json does not contain a "
                "plan_path field. Cannot verify selections match the current plan. "
                "This may be an export from an older version.",
                file=sys.stderr,
            )
        elif os.path.normpath(json_plan_path) != os.path.normpath(plan_path):
            raise ValueError(
                f"Plan path mismatch in consolidated_user_selections.json!\n"
                f"  Selections were exported for: {json_plan_path}\n"
                f"  Current plan being processed:  {plan_path}\n"
                f"This likely means consolidated_user_selections.json was copied "
                f"from a different plan's directory. Remove or replace the file "
                f"and retry."
            )

    # Detect G-level format accidentally saved as consolidated_user_selections.json
    # G-level files have: skipped_suggestions (list of "G1S2" strings),
    # edited_descriptions (dict), and skipped_groups contains ints
    # C-level files have: skipped_groups as list of 12-char hex consolidated_ids
    has_g_level_markers = (
        "skipped_suggestions" in data
        or "edited_descriptions" in data
    )
    skipped_groups = data.get("skipped_groups", [])
    has_int_skipped = any(isinstance(g, int) for g in skipped_groups)

    if has_g_level_markers or has_int_skipped:
        print(
            "WARNING: consolidated_user_selections.json contains G-level format data. "
            "This file will be ignored for C-level decisions. If you exported from "
            "report.html, save as user_selections.json instead.",
            file=sys.stderr,
        )
        return None

    # Detect format version and migrate if needed
    try:
        from .state_manager import get_format_version
    except ImportError:
        try:
            from utils.state_manager import get_format_version
        except ImportError:
            get_format_version = None

    if get_format_version is not None:
        version = get_format_version(data)
        if version < 2 and groups is not None:
            try:
                from .selection_migration import _migrate_v1_html_selections
            except ImportError:
                try:
                    from utils.selection_migration import _migrate_v1_html_selections
                except ImportError:
                    _migrate_v1_html_selections = None

            if _migrate_v1_html_selections is not None:
                data = _migrate_v1_html_selections(data, groups)

    return data


def merge_consolidated_selections(
    html_selections: Optional[Dict],
    md_skipped: Set[str],
    md_overrides: Dict[str, str],
) -> Tuple[Set[str], Dict[str, str]]:
    """Merge HTML and markdown consolidated decisions with HTML taking precedence.

    Precedence rules:
    1. If item has decision in HTML selections -> use HTML decision
    2. If item not in HTML but in markdown -> use markdown decision
    3. If in neither -> not skipped, no override

    Args:
        html_selections: Parsed consolidated_user_selections.json or None
        md_skipped: Consolidated_ids marked skip in consolidated report.md
        md_overrides: Dict of {consolidated_id: override_value} from markdown

    Returns:
        Tuple of:
        - merged_skipped_groups: Set[str] (consolidated_ids)
        - merged_overrides: Dict[str, str] (consolidated_id -> override value)
    """
    # If no HTML selections, return markdown values
    if html_selections is None:
        return (
            md_skipped,
            md_overrides,
        )

    # HTML selections completely replace markdown equivalents
    merged_skipped = set(html_selections.get('skipped_groups', []))
    merged_overrides = dict(html_selections.get('validation_overrides', {}))

    return (
        merged_skipped,
        merged_overrides,
    )
