"""State manager for persisting implementation state across sessions."""

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .git_utils import get_current_head, get_branch_name
from .judge_vocab import upgrade_legacy
from .schema_validator import validate_state_file

logger = logging.getLogger(__name__)

# --- Constants ---
ID_ALGO_VERSION = 1
CURRENT_FORMAT_VERSION = 2

#: The phases that judge implement's output. A completion of one of them that
#: predates implement's latest completion reviewed (or fixed) an earlier
#: state of the code: typically the empty repository a review was run against
#: before implement existed, or code implement has since rewritten
#: (2026-10-02, Codex racing run: a plain `--review-code` was refused as
#: already done, and implement's next step pointed at `--apply-code-fixes` on
#: that stale review).
PHASES_STALED_BY_IMPLEMENT = ("code-review", "apply-code-fixes")


def _parse_completion_time(value) -> Optional[datetime]:
    """A ``phases_completed`` timestamp as a naive local datetime, or None."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


def mark_phase_started_in_state(state: Dict[str, Any], phase: str) -> None:
    """Record *phase* as handed out and not yet finished (``phases_started``).

    The native leg's record: a write phase's orchestrator writes this when it
    emits its batch document, and the host's closing step
    (``--mark-completed``, ``util finalize-tracking --mark-completed``) turns
    it into a completion. Until 2026-10-05 ``apply-suggestions`` and
    ``implement`` recorded themselves *completed* at that emit site, so
    ``status``, the next-step builder and the completed-phase guard read a
    phase as done while the host was still applying its batches — or after it
    had died halfway (racing re-run, 2026-10-04: implement stamped 10:43:30,
    six minutes before its last task). ``get_workflow_status`` reads this map
    as ``in_progress``, the one status that spells ``--resume``. Re-stamped on
    every emit (a ``--resume`` is a new run of the same phase), popped by
    :func:`mark_phase_completed_in_state` and by a skip. An EARLIER completion
    of the same phase (a previous review cycle's, or the implement run this
    one repeats) is left in ``phases_completed`` untouched — nothing rewrites
    history — and the two stamps decide: :meth:`StateManager.is_phase_started`
    and :meth:`StateManager.is_phase_completed` read the newer record as the
    phase's state (reviewer repro, 2026-10-05: a cycle-2 native re-run read
    ``completed`` on cycle 1's stamp, mid-loop).
    """
    started = state.get("phases_started")
    if not isinstance(started, dict):
        started = {}
        state["phases_started"] = started
    started[phase] = datetime.now().isoformat()


def mark_phase_completed_in_state(state: Dict[str, Any], phase: str) -> None:
    """``StateManager.mark_phase_completed``'s mutation on a raw state dict.

    Module-level, like :func:`compute_plan_hash`, for the file-level writers
    (``finalize_tracking``) that must not build a ``StateManager`` over a plan
    the native leg has just edited. The started record is cleared in the same
    write, so a phase is never both.
    """
    if "phases_completed" not in state:
        state["phases_completed"] = {}
    state["phases_completed"][phase] = datetime.now().isoformat()
    started = state.get("phases_started")
    if isinstance(started, dict):
        started.pop(phase, None)


def update_task_status_in_state(
    state: Dict[str, Any], task_id: str, status: str, reason: Optional[str] = None
) -> None:
    """``StateManager.update_task_status``'s mutation on a raw state dict.

    Module-level for the same reason :func:`mark_phase_completed_in_state`
    is: the native write leg's per-batch record (``util batch-done``) edits
    ``state.json`` as a file, mid-loop, and must not build a ``StateManager``
    over a plan the batch just edited.
    """
    if "task_status" not in state:
        state["task_status"] = {}
    state["task_status"][task_id] = status
    if reason is not None:
        if "task_status_reasons" not in state:
            state["task_status_reasons"] = {}
        state["task_status_reasons"][task_id] = reason


def mark_item_processed_in_state(
    state: Dict[str, Any],
    phase: str,
    item_id: str,
    status: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """``StateManager.mark_item_processed``'s mutation on a raw state dict.

    ``processed_<phase>`` is the map the apply base's ``--resume`` filters
    its merged groups on (``apply_orchestrator_base`` step 23), keyed by the
    group id a batch's ``expected_ids`` carry; ``phase`` is the base's own
    ``phase_name`` (the phase DIR word — ``apply-fixes`` for apply-code-fixes).
    """
    key = f"processed_{phase}"
    if key not in state or not isinstance(state[key], dict):
        state[key] = {}
    state[key][item_id] = {
        "status": status,
        "details": details or {},
        "timestamp": datetime.now().isoformat(),
    }


def compute_plan_hash(plan_path: Path) -> str:
    """Hash a plan's content the way ``state.json``'s ``plan_hash`` records it.

    Module-level so a *file-level* state writer — `finalize_tracking`, which
    edits the plan on the native leg — can re-stamp the field after its own
    edit without building a `StateManager` (which would read that edit as a
    human editing the plan and set `plan_changed`).
    """
    if plan_path.exists():
        content = plan_path.read_text(encoding="utf-8")
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    return ""


# --- Canonicalization & ID Generation ---

def canonicalize_hash_input(value) -> str:
    """Canonicalize a string for hash input.

    1. None/missing → ""
    2. Unicode NFC normalization
    3. Whitespace collapse (all runs → single space, strip)
    4. No case folding (casing preserved)
    """
    if value is None:
        return ""
    value = str(value)
    value = unicodedata.normalize("NFC", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def generate_group_id(group: Dict[str, Any], discriminator: int = 0) -> str:
    """
    Generate a stable identifier for a group based on its content.

    Uses theme + normalized suggestions + reference fields to create
    a hash that remains stable across re-runs.

    Args:
        group: A group dictionary with theme, suggestions, etc.
        discriminator: ``stamp_stable_ids``'s collision counter; ``0`` (the
            plain content hash) for every caller that computes an id to *join*
            on. Neither ``file`` nor ``line_range`` is hashed, so two
            word-identical findings in two files — which the grouper keeps
            apart on purpose — are two groups with one content hash; the
            counter is what tells them apart once stamped.

    Returns:
        A 16-character hex string identifier
    """
    # Extract stable identifying fields. The theme enters the hash at the
    # 50-character cut the grouper used to apply when it *built* themes: every
    # group persisted before 2026-09-13 carries a theme of at most 50 characters
    # (so this is the identity on it), and a theme built since — cut at a word
    # boundary within 60, with an ellipsis — hashes to what the old cut would
    # have produced. A theme is display text; a display change is not an
    # identity change, and the container goldens bake these ids in.
    theme = str(group.get("theme", ""))[:50]
    suggestions = group.get("suggestions", group.get("issues", []))

    # Normalize suggestions for hashing
    normalized = []
    for s in suggestions:
        normalized.append({
            "type": s.get("type", ""),
            "section": s.get("section", s.get("reference", "")),
            "details": str(s.get("details", s.get("desc", "")))[:200],  # Truncate for stability
        })

    payload: Dict[str, Any] = {"theme": theme, "suggestions": normalized}
    if discriminator:
        payload["discriminator"] = f"-{discriminator}"
    content = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def generate_suggestion_id(suggestion: Dict[str, Any], discriminator: int = 0) -> str:
    """Generate a stable 16-char hex hash for a suggestion.

    Uses title + type + full description + reference (parent group excluded
    for stability across regrouping). ``discriminator`` is the collision
    counter ``stamp_stable_ids`` applies (``0`` hashes the plain content; a
    non-zero value produces byte-for-byte the rehash it always stamped).
    """
    fields = {
        "v": ID_ALGO_VERSION,
        "title": canonicalize_hash_input(suggestion.get("title", "")),
        "type": canonicalize_hash_input(suggestion.get("type", "")),
        "description": canonicalize_hash_input(
            suggestion.get("details", suggestion.get("desc", suggestion.get("description", "")))
        ),
        "reference": canonicalize_hash_input(
            suggestion.get("reference", suggestion.get("section", ""))
        ),
    }
    if discriminator:
        fields["discriminator"] = f"-{discriminator}"
    content = json.dumps(fields, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def group_id_short(group_id_16: str) -> str:
    """Truncate a 16-char group/suggestion hash to 8 chars for display."""
    return group_id_16[:8]


def _anchor(suggestion: Dict[str, Any]) -> Tuple[str, str]:
    """The (file, line_range) a code-review finding is pinned to; empty for a
    plan suggestion. This is exactly what the identity hash leaves out."""
    return (
        str(suggestion.get("file", "") or ""),
        str(suggestion.get("line_range", suggestion.get("line", "")) or ""),
    )


def match_issues_to_suggestions(
    issues: List[Dict[str, Any]],
    groups: List[Dict[str, Any]],
) -> Dict[int, Tuple[int, int]]:
    """Map each 1-based issue to the ``(group_index, suggestion_index)``
    (both 1-based) that holds it in *groups*.

    The join key is the content hash ``generate_suggestion_id`` computes — the
    one key an ``issues.json`` entry (written before stamping, so it carries no
    ``suggestion_hash``) and a ``grouped.json`` member share. Same-hash
    duplicates — one reviewer's word-identical finding in two files, which the
    grouper keeps as two groups — are told apart by the stamped hash when the
    issue carries one, else by ``file`` and ``line_range``, else by order of
    occurrence; each grouped location is consumed once, so the discriminated
    second suggestion is reached rather than the first location twice. An
    issue with no free location is left out of the map (never guessed).
    """
    locations: Dict[str, List[Tuple[Tuple[int, int], Dict[str, Any]]]] = {}
    for g_idx, group in enumerate(groups, 1):
        members = group.get("suggestions", group.get("issues", []))
        for s_idx, sugg in enumerate(members, 1):
            if isinstance(sugg, dict):
                locations.setdefault(generate_suggestion_id(sugg), []).append(
                    ((g_idx, s_idx), sugg)
                )

    mapping: Dict[int, Tuple[int, int]] = {}
    consumed: set = set()
    for issue_num, issue in enumerate(issues, 1):
        if not isinstance(issue, dict):
            continue
        pool = [
            (loc, sugg) for loc, sugg in locations.get(generate_suggestion_id(issue), [])
            if loc not in consumed
        ]
        if not pool:
            continue
        chosen: Optional[Tuple[int, int]] = None
        stamped = issue.get("suggestion_hash")
        if stamped:
            chosen = next(
                (loc for loc, sugg in pool if sugg.get("suggestion_hash") == stamped),
                None,
            )
        if chosen is None and len(pool) > 1:
            want = _anchor(issue)
            chosen = next((loc for loc, sugg in pool if _anchor(sugg) == want), None)
            if chosen is None:
                chosen = next(
                    (loc for loc, sugg in pool if _anchor(sugg)[0] == want[0]), None
                )
        if chosen is None:
            chosen = pool[0][0]
        consumed.add(chosen)
        mapping[issue_num] = chosen
    return mapping


def stamp_stable_ids(groups: List[Dict[str, Any]]) -> None:
    """Idempotently stamp group_hash, suggestion_hash, display_label, display_hash
    on each group and its suggestions. Includes collision detection with
    deterministic rehashing.

    Mutates groups in-place.
    """
    # Collect all canonical hashes for collision detection. Groups and
    # suggestions are two namespaces (a group may legitimately share a hash
    # with a suggestion), so each collides only with its own kind.
    all_hashes: Dict[str, Tuple[str, int, int]] = {}  # hash -> (type, g_idx, s_idx)

    def _discriminated(kind: str, label: str, hash_of) -> str:
        """The first of ``hash_of(0)``, ``hash_of(1)``, … not yet stamped on
        a *kind*. Deterministic: the n-th same-content item in stamping order
        always gets discriminator n-1, so a re-stamp of the same groups
        reproduces every id."""
        discriminator = 0
        candidate = hash_of(discriminator)
        while candidate in all_hashes and all_hashes[candidate][0] == kind:
            discriminator += 1
            if discriminator > 10:
                raise ValueError(
                    f"Cannot resolve hash collision for {kind} {label} "
                    f"after 10 attempts"
                )
            candidate = hash_of(discriminator)
        return candidate

    for g_idx, group in enumerate(groups):
        g_num = g_idx + 1
        # Group hash, with the same collision loop the suggestions get. The
        # content hash reads neither `file` nor `line_range`, and the grouper
        # never merges across files, so one reviewer's boilerplate finding in
        # two files used to become two groups under ONE `group_hash` — every
        # consumer that filters by that hash (an apply-phase skip, an
        # `invalid` mark, a human decision) then hit both groups at once.
        ghash = _discriminated(
            "group", f"G{g_num}", lambda d: generate_group_id(group, d)
        )
        all_hashes[ghash] = ("group", g_idx, -1)
        group["group_hash"] = ghash
        group["display_label"] = f"G{g_num}"

        suggestions = group.get("suggestions", group.get("issues", []))
        for s_idx, sugg in enumerate(suggestions):
            s_num = s_idx + 1
            shash = _discriminated(
                "suggestion", f"G{g_num}S{s_num}",
                lambda d: generate_suggestion_id(sugg, d),
            )
            all_hashes[shash] = ("suggestion", g_idx, s_idx)
            sugg["suggestion_hash"] = shash
            sugg["display_label"] = f"G{g_num}S{s_num}"

    # Now compute display hashes (8-char prefix, extended on collision)
    _assign_display_hashes(groups)


def _assign_display_hashes(groups: List[Dict[str, Any]]) -> None:
    """Assign display_hash fields, extending from 8 to 10+ chars on prefix collision."""
    # Collect all canonical hashes and their locations
    entries: List[Tuple[str, str, Dict]] = []  # (canonical_hash, type, obj_ref)

    for group in groups:
        ghash = group.get("group_hash", "")
        if ghash:
            entries.append((ghash, "group", group))
        for sugg in group.get("suggestions", group.get("issues", [])):
            shash = sugg.get("suggestion_hash", "")
            if shash:
                entries.append((shash, "suggestion", sugg))

    # Group by 8-char prefix to detect collisions
    prefix_len = 8
    while prefix_len <= 16:
        prefix_groups: Dict[str, List[Tuple[str, str, Dict]]] = {}
        for canonical, etype, obj in entries:
            prefix = canonical[:prefix_len]
            if prefix not in prefix_groups:
                prefix_groups[prefix] = []
            prefix_groups[prefix].append((canonical, etype, obj))

        # Check for collisions (different canonical hashes sharing same prefix)
        has_collision = False
        for prefix, items in prefix_groups.items():
            unique_canonicals = set(canonical for canonical, _, _ in items)
            if len(unique_canonicals) > 1:
                has_collision = True
                break

        if not has_collision or prefix_len >= 16:
            break
        prefix_len += 2  # Extend by 2 chars at a time

    # Assign display hashes
    for canonical, etype, obj in entries:
        obj["display_hash"] = canonical[:prefix_len]


# --- Format Version Helpers ---

def get_format_version(data: dict) -> int:
    """Return format version of a loaded JSON file. Missing field = v1.

    NOT used for state.json — use get_state_schema_version() instead.
    """
    return data.get("format_version", 1)


def get_state_schema_version(data: dict) -> str:
    """Return schema version of state.json. Missing field = '1.0'."""
    return data.get("schema_version", "1.0")


def extract_versioned_payload(data, payload_key: str):
    """Extract payload from a versioned JSON envelope.

    Handles both v1 (bare payload, e.g., a raw list) and v2 (dict with
    format_version + payload_key). Returns (format_version, payload).
    """
    if isinstance(data, list):
        return (1, data)
    if isinstance(data, dict) and payload_key in data:
        return (get_format_version(data), data[payload_key])
    raise ValueError(f"Unexpected format: expected list or dict with '{payload_key}' key")


def load_groups_payload(data) -> List[Dict[str, Any]]:
    """Load groups from either v1 bare array or v2 envelope. Returns groups list."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "groups" in data:
        return data["groups"]
    raise ValueError("Unexpected grouped.json format: expected list or dict with 'groups' key")


def save_groups_payload(groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Wrap groups in v2 envelope for writing."""
    return {"format_version": CURRENT_FORMAT_VERSION, "groups": groups}


def handle_plan_hash_change(
    old_decisions: Dict[str, Any],
    old_groups: List[Dict[str, Any]],
    new_groups: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Remap decisions when plan content changes.

    Args:
        old_decisions: Previous human decisions keyed by group_id
        old_groups: Previous groups
        new_groups: Current groups after plan change

    Returns:
        Remapped decisions, dropping any that no longer match
    """
    # Build map of old group_ids to decisions
    old_id_to_decision = {}
    for old_group in old_groups:
        gid = generate_group_id(old_group)
        if gid in old_decisions:
            old_id_to_decision[gid] = old_decisions[gid]

    # Match new groups to old decisions
    remapped = {}
    for new_group in new_groups:
        new_gid = generate_group_id(new_group)
        if new_gid in old_id_to_decision:
            remapped[new_gid] = old_id_to_decision[new_gid].copy()
            remapped[new_gid]["remapped"] = True

    return remapped


class StateManager:
    """Manages persistent state for multi-LLM skill sessions."""

    SCHEMA_VERSION = "2.0"

    def __init__(
        self,
        plan_path: Path,
        state_dir: Optional[Path] = None,
        *,
        read_only: bool = False,
    ):
        """
        Initialize state manager.

        Args:
            plan_path: Path to the plan being implemented
            state_dir: Deprecated, ignored. State now stored in plan directory.
            read_only: Load for reading only, and never touch the file: no
                old-location move, no judge-name write-back (the upgrade
                still applies in memory), and ``save()`` raises. For the
                commands that only report (``status``, ``util next-command``,
                ``--status``), which a host polls while another process may
                be writing ``state.json``.
        """
        self.plan_path = Path(plan_path).resolve()
        self.read_only = read_only

        # State file lives in the plan's output directory
        from .output_handler import get_output_dir
        self.plan_dir = get_output_dir(self.plan_path)
        self.state_file = self.plan_dir / "state.json"

        self.state: Dict[str, Any] = {}
        if not read_only:
            self._migrate_from_old_location()  # Check for old state file
        self._load_or_create()

    def _migrate_from_old_location(self) -> None:
        """Migrate state file from old hash-based location if it exists.

        Note: Migration is silent unless verbose/debug logging is enabled.
        Uses copy+delete instead of rename for cross-filesystem compatibility.
        """
        old_state_dir = Path(__file__).parent.parent / "state"
        path_hash = hashlib.sha256(str(self.plan_path).encode()).hexdigest()[:16]
        old_state_file = old_state_dir / f"{path_hash}.json"

        if old_state_file.exists() and not self.state_file.exists():
            try:
                # Validate JSON is parseable before migrating
                with open(old_state_file, 'r', encoding='utf-8') as f:
                    json.load(f)

                # Use copy+delete instead of rename (works across filesystems)
                self.plan_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old_state_file, self.state_file)

                # Verify copy succeeded before deleting old file
                if self.state_file.exists():
                    os.unlink(old_state_file)
                    logger.debug(f"Migrated state file to: {self.state_file}")

            except (json.JSONDecodeError, OSError) as e:
                # Log warning but continue - old state may be corrupt or inaccessible
                logger.warning(f"Failed to migrate state from {old_state_file}: {e}")
                # Don't delete old file on failure - let user investigate

    def _compute_plan_path_hash(self) -> str:
        """Compute hash of plan PATH for state file naming (stable across content edits)."""
        path_str = str(self.plan_path)
        return hashlib.sha256(path_str.encode()).hexdigest()[:16]

    def _compute_plan_hash(self) -> str:
        """Compute hash of plan content for change detection."""
        return compute_plan_hash(self.plan_path)

    def _create_initial_state(self) -> Dict[str, Any]:
        """Create initial state structure."""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "plan_path": str(self.plan_path),
            "plan_hash": self._compute_plan_hash(),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "head_at_start": get_current_head(),
            "branch_name": get_branch_name() or "unknown",
            "review_phase_completed": False,
            "tracked_files": [],
            "task_status": {},
            "phases_completed": {},
            "phases_skipped": {},
        }

    def _load_or_create(self) -> None:
        """Load existing state or create new one."""
        if self.state_file.exists():
            try:
                self.state = self._read_upgraded()

                # Check for plan changes. Deliberately set-only: the flag is
                # cleared by whoever CONSUMES it, never by a bare load.
                # Setting it also re-stamps `plan_hash`, so the very next load
                # finds the hash agreeing — and a clear-on-agreement branch
                # therefore erased a genuine human edit as soon as any
                # intermediate process loaded and saved between the edit and
                # `implement --resume` (`full`'s `_record_progress` does
                # exactly that before every step; so do `review-tasks` and
                # `check_workflow_prerequisites`). Stale residue is kept out
                # at the other end instead, by every plugin-made plan edit
                # re-stamping the hash through `refresh_plan_hash()` (F1,
                # audited 2026-09-16).
                current_hash = self._compute_plan_hash()
                if self.state.get("plan_hash") != current_hash:
                    self.state["plan_hash"] = current_hash
                    self.state["plan_changed"] = True

            except json.JSONDecodeError:
                self.state = self._create_initial_state()
        else:
            self.state = self._create_initial_state()

    def _read_upgraded(self) -> Any:
        """``state.json`` with every legacy judge spelling upgraded.

        A run started before the judge rename recorded its decisions as
        ``claude_auto_decide*``; they are read as the host-neutral names, so
        ``--resume`` crosses the upgrade. An upgrade that changed something is
        also written back at once, because a load is not always followed by a
        save — a native-leg ``--resume`` of an apply phase saves nothing until
        the host's ``--mark-completed`` — and the host reads this file raw,
        filtering ``human_decisions_{phase}`` on the new names, for its
        "Decided by the judge agent" section (review 2026-10-02). Not under
        ``read_only``, and never when nothing changed.

        Raises what ``open``/``json.loads`` raise; a failed write-back does not.
        """
        with open(self.state_file, "r", encoding="utf-8") as f:
            text = f.read()
        loaded = upgrade_legacy(json.loads(text))
        if (not self.read_only and isinstance(loaded, dict)
                and loaded != json.loads(text)):
            self._write_back_upgrade(loaded, text)
        return loaded

    def _write_back_upgrade(self, document: Dict[str, Any], read_text: str) -> None:
        """Persist the judge-name upgrade of *read_text*, and only that.

        The document is written as loaded — no ``updated_at`` stamp, no
        plan-change flag — so the write renames keys and values and changes
        nothing else. Skipped when the file no longer holds *read_text*:
        another process saved in between, and its file is the newer one.
        """
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                if f.read() != read_text:
                    return
            self._write_atomic(document)
        except OSError as exc:
            # The in-memory upgrade still holds, and every Python reader maps
            # the old names; the host's raw read is the only one that misses.
            logger.warning(
                "Could not write the judge-name upgrade back to %s: %s",
                self.state_file, exc,
            )

    def _validate_state(self, state: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate state against schema.

        Args:
            state: State dictionary to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            is_valid, errors = validate_state_file(state)
            return is_valid, errors
        except Exception as e:
            # If schema validation fails, log but allow save
            # (schema might not exist yet or validation might be unavailable)
            return True, [str(e)]

    def save(self) -> None:
        """
        Save current state to file with atomic writes.

        Uses write-to-temp + atomic rename pattern to prevent corruption
        if the save operation is interrupted.

        Raises:
            IOError: If the save operation fails
            RuntimeError: On a ``read_only`` manager
        """
        if self.read_only:
            raise RuntimeError(
                f"read-only StateManager for {self.plan_path} cannot save"
            )
        self.state["updated_at"] = datetime.now().isoformat()

        # Validate state before saving
        is_valid, errors = self._validate_state(self.state)
        if not is_valid:
            # Log validation errors but allow save to proceed
            import warnings
            warnings.warn(f"State validation warnings: {errors}")

        self._write_atomic(self.state)

    def _write_atomic(self, document: Dict[str, Any]) -> None:
        """Write *document* to ``state.json`` via a temp file and ``os.replace``.

        Raises:
            IOError: If the write fails
        """
        # Write to temp file first, then atomically rename
        temp_fd = None
        temp_path = None
        try:
            # Create a temp file in the same directory as the target
            # Use state_file.parent since state is now plan-local
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            temp_fd, temp_path = tempfile.mkstemp(
                dir=str(self.state_file.parent),
                suffix='.tmp',
                prefix='.state_'
            )

            # Write state to temp file
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                json.dump(document, f, indent=2)
            temp_fd = None  # fdopen closes the file descriptor

            # Atomically rename temp file to target location
            os.replace(temp_path, str(self.state_file))
            temp_path = None  # File has been moved

        except Exception as e:
            # Clean up temp file if something went wrong
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except OSError:
                    pass
            if temp_path is not None and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            raise IOError(f"Failed to save state: {e}") from e

    def reload(self) -> None:
        """Re-read ``state.json`` from disk, replacing the in-memory dict.

        **Every unsaved in-memory change is discarded** — a caller with pending
        edits must ``save()`` first. This exists for the one legitimate case of
        a second writer: a file-level helper such as ``finalize_tracking.py``
        reads ``state.json``, adds to it and writes it back, while ``save()``
        dumps this object's dict wholesale and never merges. Whoever saves last
        wins, and until 2026-09-16 that was `_finalize`'s closing `save()`,
        which erased the `tracked_files` `finalize_tracking` had just computed —
        an implement run that wrote `index.html` finalized with the two stale
        entries earlier phases had left, and `review-code` would have reviewed
        those (F3). Calling this straight after the external writer makes the
        later `save()` carry its work instead of overwriting it.

        Unlike construction, this does not re-run the plan-change detection:
        `plan_changed` answers "did a human edit the plan since the last
        session", which is a question about the session, not about a re-read
        mid-run (a write phase edits the plan itself).
        """
        if not self.state_file.exists():
            return
        try:
            loaded = self._read_upgraded()
        except (json.JSONDecodeError, OSError):
            # A corrupt or unreadable file is not worth losing the dict we hold:
            # the next `save()` rewrites it wholesale anyway.
            return
        if isinstance(loaded, dict):
            self.state = loaded

    def get(self, key: str, default: Any = None) -> Any:
        """Get a state value."""
        return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a state value."""
        self.state[key] = value

    # --- Metrics (in-memory; see utils/metrics.py for the file-level CLI) ---

    def start_metrics_phase(self, phase: str, total_batches: int) -> None:
        """Stamp ``metrics_progress.{phase}`` so ETAs have a baseline.

        The in-memory twin of ``metrics.start_phase``. Routing metrics through
        this object rather than through ``metrics.py``'s own reader/writer is
        what stops the two racing on ``state.json`` — see :meth:`reload`.
        """
        from . import metrics as _metrics

        _metrics.start_phase_in_state(self.state, phase, int(total_batches))

    def finish_metrics_phase(
        self, phase: str
    ) -> Optional[Tuple[Optional[int], str]]:
        """The in-memory twin of ``metrics.finish_phase``.

        Returns ``(elapsed_ms, line)`` — the closing ``[ETA] … phase complete``
        line is handed back rather than printed, the way :meth:`record_metric`
        already hands back its own, so the scheduler can route it to the phase
        log the rest of that run's ETA lines went to (F7).
        """
        from . import metrics as _metrics

        if phase not in (self.state.get("metrics_progress") or {}):
            return None
        return _metrics.finish_phase_in_state(self.state, phase)

    def record_metric(self, phase: str, label: str, **kwargs: Any) -> Optional[str]:
        """Append one ``metrics.{phase}`` entry in memory; ``save()`` persists it.

        The in-memory twin of ``metrics.record_metric``; returns the rendered
        ``[ETA]`` line (or None) for the caller to place, and prints nothing.
        """
        from . import metrics as _metrics

        return _metrics.record_metric_in_state(self.state, phase, label, **kwargs)

    def update_task_status(self, task_id: str, status: str, reason: Optional[str] = None) -> None:
        """Update status of a task.

        Args:
            task_id: Task identifier
            status: New status value
            reason: Optional reason for the status change (e.g., why a task was skipped)
        """
        update_task_status_in_state(self.state, task_id, status, reason)

    def get_task_status(self, task_id: str) -> Optional[str]:
        """Get status of a task."""
        return self.state.get("task_status", {}).get(task_id)

    def get_task_status_reason(self, task_id: str) -> Optional[str]:
        """Get the reason for a task's current status."""
        return self.state.get("task_status_reasons", {}).get(task_id)

    def get_all_task_statuses(self) -> Dict[str, str]:
        """Get all task statuses."""
        return self.state.get("task_status", {}).copy()

    #: The id shape every task the task decomposer generates carries — mirrors
    #: ``utils.task_decomposer.TASK_ID_PATTERN``. Restated here rather than
    #: imported so this low-level module stays free of a dependency on the
    #: (much heavier) decomposer; if that pattern ever changes, this one has to
    #: follow. ``task_status`` is one flat map shared by every write-batch
    #: scheduler: implement's own decomposed tasks AND the batch ids the
    #: apply-suggestions / apply-code-fixes / apply-task-suggestions phases
    #: assign their write batches (``apply-batch-N``, ``apply-fix-batch-N``,
    #: ``apply-task-batch-N`` — see ``utils.write_phase_bridge``). Only the
    #: decomposer's own ids look like ``T###``.
    _IMPLEMENT_TASK_ID_PATTERN = re.compile(r"^T[0-9]{3}$")

    def implementation_task_statuses(self) -> Dict[str, str]:
        """Task statuses that belong to the implement phase's own task graph.

        A caller reporting implement's progress (``check_workflow_prerequisites``,
        or anything else) must not count another phase's write-batch ids as
        implement tasks — see :attr:`_IMPLEMENT_TASK_ID_PATTERN`.
        """
        return {
            task_id: status
            for task_id, status in self.state.get("task_status", {}).items()
            if self._IMPLEMENT_TASK_ID_PATTERN.match(task_id)
        }

    def record_task_titles(
        self, titles: Dict[str, str], *, human_ids: Optional[Iterable[str]] = None,
    ) -> bool:
        """Remember each implement task's title by id, under ``task_titles``.

        ``implement`` records them when it decomposes the task list, on both
        legs, so a deferred task can be named later by what it is and not only
        by its id (2026-10-04: a complete workflow said nothing about the two
        manual tasks it left the user). *titles* is that whole decomposition:
        the titles merge, but its id set REPLACES ``implement_task_ids``, which
        is what :meth:`deferred_implementation_tasks` scopes to — nothing
        resets ``task_status`` on a fresh implement, so an id an earlier
        decomposition skipped stays ``skipped`` after tasks.md is regenerated
        (adversarial review, 2026-10-05). *human_ids*, when given, are that
        decomposition's manual tasks and REPLACE ``implement_human_task_ids``
        the same way — the key is absent when there are none, so a plan with
        no manual task gains no key (fix round 7: the backstop in
        :meth:`deferred_implementation_tasks`). Returns whether anything
        changed; the caller saves.
        """
        recorded = dict(self.state.get("task_titles") or {})
        merged = {**recorded, **{str(k): str(v) for k, v in titles.items() if k}}
        current = sorted(str(k) for k in titles if k)
        human = (
            None if human_ids is None
            else sorted(str(k) for k in human_ids if k) or None
        )
        human_changed = (
            human_ids is not None
            and self.state.get("implement_human_task_ids") != human
        )
        if (
            merged == recorded
            and self.state.get("implement_task_ids") == current
            and not human_changed
        ):
            return False
        self.state["task_titles"] = merged
        self.state["implement_task_ids"] = current
        if human is not None:
            self.state["implement_human_task_ids"] = human
        elif human_ids is not None:
            self.state.pop("implement_human_task_ids", None)
        return True

    #: The recorded reason of an implement task the USER declined at
    #: pause-and-ask (``write_batch_scheduler.resolve_human_task_plan``, and
    #: the same words ``instructions/implement.md`` gives the native host). A
    #: decision, not a deferral: the task is not left for them to do.
    _USER_DECLINED_REASON = "skipped by user during pause-and-ask"

    #: The recorded reason's prefix for a failed code task the user answered
    #: ``--human-task <id>=skip`` (``write_batch_scheduler.ANSWERED_SKIP_PREFIX``
    #: spells it). A decision too: the user gave the task up, so it is not
    #: owed as manual work, and neither is a dependent skipped with it.
    _ANSWERED_SKIP_PREFIX = "skipped — you answered --human-task "

    #: The recorded reason of a dependent the transitive skip walk took with
    #: its parent (both legs spell it ``skipped — depends on skipped task
    #: {parent_id} ({parent_title})``); owed exactly when the parent is.
    _DEPENDS_ON_SKIPPED = re.compile(r"^skipped — depends on skipped task (T[0-9]{3})\b")

    def _deferred_to_user(self, task_id: str, seen: Optional[set] = None) -> bool:
        """Whether *task_id*'s skip leaves the task for the user to do.

        A deferral (a human task under a skip strategy, ``deferred — …``) and
        a dependent of one are owed; a pause-and-ask "Skip", a failed code
        task answered ``--human-task <id>=skip``, and a dependent of either
        are not. A reason this does not recognise — none recorded, a state
        written before reasons were — stays owed, as every skip used to be.
        """
        reason = str((self.state.get("task_status_reasons") or {}).get(task_id) or "")
        if reason.startswith((self._USER_DECLINED_REASON, self._ANSWERED_SKIP_PREFIX)):
            return False
        match = self._DEPENDS_ON_SKIPPED.match(reason)
        seen = set() if seen is None else seen
        if match is None or task_id in seen:
            return True
        seen.add(task_id)
        return self._deferred_to_user(match.group(1), seen)

    def deferred_implementation_tasks(self) -> List[Dict[str, Optional[str]]]:
        """The implement tasks the run left for the user, as ``{id, title}``, by id.

        Read from ``task_status`` (both legs record a deferred human task, and
        a dependent ``skip-dependents`` skipped with it, as ``skipped``), never
        by re-parsing tasks.md: what the run deferred is what state says it
        deferred. Scoped to the decomposition ``implement`` recorded last
        (``implement_task_ids``, when there is one) and to the skips that are
        deferrals (:meth:`_deferred_to_user`), since 2026-10-05: an earlier
        cycle's skip and a task the user declined are not owed. ``title`` is
        ``None`` when ``task_titles`` has no entry (a state written before
        titles were recorded).
        """
        titles = self.state.get("task_titles") or {}
        current = self.state.get("implement_task_ids")
        scope = set(current) if isinstance(current, list) else None
        statuses = self.implementation_task_statuses()
        owed = {
            task_id for task_id, status in statuses.items()
            if status == "skipped" and self._deferred_to_user(task_id)
        }
        # The backstop (fix round 7): implement is on record as completed and
        # a manual task of its decomposition has no answer at all — neither
        # `completed` (done) nor a recorded skip. Nobody did it, so it is
        # owed: a host that resolved the strategy itself and recorded nothing
        # must never turn a deferred task into a clean `[WORKFLOW_COMPLETE]`.
        if self.is_phase_completed("implement"):
            for task_id in self.state.get("implement_human_task_ids") or []:
                if statuses.get(task_id) not in ("completed", "skipped"):
                    owed.add(str(task_id))
        return [
            {"id": task_id, "title": titles.get(task_id) or None}
            for task_id in sorted(owed)
            if scope is None or task_id in scope
        ]

    def track_file(self, path: str, action: str, task_id: str) -> None:
        """
        Track a file modification.

        Args:
            path: File path
            action: Action performed (created, modified, deleted)
            task_id: ID of the task that modified this file
        """
        if "tracked_files" not in self.state:
            self.state["tracked_files"] = []

        # Check if file already tracked
        for entry in self.state["tracked_files"]:
            if entry["path"] == path:
                entry["action"] = action
                entry["task_id"] = task_id
                return

        self.state["tracked_files"].append({
            "path": path,
            "action": action,
            "task_id": task_id,
        })

    def get_tracked_files(self) -> List[Dict[str, Any]]:
        """Get list of tracked files."""
        return self.state.get("tracked_files", [])

    def get_files_by_task(self, task_id: str) -> List[str]:
        """Get files modified by a specific task."""
        return [
            entry["path"]
            for entry in self.state.get("tracked_files", [])
            if entry.get("task_id") == task_id
        ]

    def mark_review_phase_completed(self) -> None:
        """Mark the plan review phase as completed."""
        self.state["review_phase_completed"] = True

    def is_review_phase_completed(self) -> bool:
        """Check if the plan review phase is completed."""
        return self.state.get("review_phase_completed", False)

    # --- Phase Completion Tracking ---

    def mark_phase_completed(self, phase: str) -> None:
        """
        Mark a workflow phase as completed.

        Args:
            phase: Phase name (review-plan, apply-suggestions, generate-tasks,
                   implement, review-code, apply-code-fixes)
        """
        mark_phase_completed_in_state(self.state, phase)
        # `review_phase_completed` predates this general map and nothing ever
        # set it (F11): a fully completed review-plan phase left it `False`
        # forever, contradicting `phases_completed["review-plan"]` in the same
        # file. `mark_review_phase_completed()`'s own docstring is "the *plan*
        # review phase", so this is the one phase name that flips it.
        if phase == "review-plan":
            self.mark_review_phase_completed()

    def mark_phase_started(self, phase: str) -> None:
        """Record *phase* handed to an invoker and not yet finished.

        See :func:`mark_phase_started_in_state`. The write phases' native
        emit path calls this where it used to call (or not call)
        :meth:`mark_phase_completed`; the executor leg never does, because
        ``WriteBatchScheduler._finalize`` completes the phase in the same
        process.
        """
        mark_phase_started_in_state(self.state, phase)

    def is_phase_started(self, phase: str) -> bool:
        """Whether *phase*'s newest record is a start, not a completion.

        A started record beside no completion is a start. Beside one, the
        stamps decide: a start newer than the completion is a re-run of a
        phase that finished once before (a second review cycle, an implement
        repeated), and the phase is mid-loop again; a start older than the
        completion is the run that completion closed, and is ignored. A
        stamp that does not parse keeps the completion (absence of a date
        is not a re-run).
        """
        started_at = _parse_completion_time(self.phase_started_at(phase))
        if started_at is None:
            return False
        completed_at = self.state.get("phases_completed", {}).get(phase)
        if completed_at is None:
            return True
        completed = _parse_completion_time(completed_at)
        return completed is not None and started_at > completed

    def phase_started_at(self, phase: str) -> Optional[str]:
        """The ISO timestamp *phase* was recorded started, or None."""
        started = self.state.get("phases_started")
        if not isinstance(started, dict):
            return None
        value = started.get(phase)
        return value if isinstance(value, str) else None

    def phase_completion_is_stale(self, phase: str) -> bool:
        """Whether *phase*'s recorded completion predates implement's.

        Only the two phases that judge implement's output are ever stale
        (:data:`PHASES_STALED_BY_IMPLEMENT`), and only when both timestamps
        parse: a missing or unreadable date keeps the completion, since the
        absence of a date is not staleness (the rule
        ``check_workflow_prerequisites._apply_batches_predate_review`` applies
        to apply batches after a review re-run).
        """
        if phase not in PHASES_STALED_BY_IMPLEMENT:
            return False
        completed = self.state.get("phases_completed", {})
        recorded = _parse_completion_time(completed.get(phase))
        implemented = _parse_completion_time(completed.get("implement"))
        if recorded is None or implemented is None:
            return False
        return recorded < implemented

    def is_phase_completed(self, phase: str) -> bool:
        """
        Check if a workflow phase is completed.

        A ``code-review`` or ``apply-code-fixes`` completion recorded before
        implement's latest completion is not one (see
        :meth:`phase_completion_is_stale`), and neither is a completion older
        than the phase's own started record (see :meth:`is_phase_started`):
        the raw record stays in ``phases_completed``, but every reader — the
        next-step builder, the completed-phase guard, ``status`` — sees the
        phase as pending (or ``in_progress``) again.

        Args:
            phase: Phase name to check

        Returns:
            True if the phase has been marked as completed
        """
        if phase not in self.state.get("phases_completed", {}):
            return False
        if self.is_phase_started(phase):
            # A start newer than the completion: the phase is running again
            # (its re-run's emit), and its old completion is not its state.
            return False
        return not self.phase_completion_is_stale(phase)

    def mark_phase_skipped(self, phase: str, reason: str) -> None:
        """
        Mark a workflow phase as explicitly skipped.

        Args:
            phase: Phase name (e.g., apply-suggestions)
            reason: Reason for skipping (e.g., "User chose to skip")
        """
        if "phases_skipped" not in self.state:
            self.state["phases_skipped"] = {}
        self.state["phases_skipped"][phase] = {
            "skipped_at": datetime.now().isoformat(),
            "reason": reason
        }
        # An explicit skip ends a run the host abandoned: a started record
        # left beside it would keep offering `--resume` for a phase the user
        # chose not to finish.
        started = self.state.get("phases_started")
        if isinstance(started, dict):
            started.pop(phase, None)

    def is_phase_skipped(self, phase: str) -> bool:
        """
        Check if a workflow phase was explicitly skipped.

        Args:
            phase: Phase name to check

        Returns:
            True if the phase has been marked as skipped
        """
        return phase in self.state.get("phases_skipped", {})

    def get_phase_skip_reason(self, phase: str) -> Optional[str]:
        """
        Get the reason a phase was skipped.

        Args:
            phase: Phase name

        Returns:
            Reason string if phase was skipped, None otherwise
        """
        skip_info = self.state.get("phases_skipped", {}).get(phase)
        return skip_info.get("reason") if skip_info else None

    def has_plan_changed(self) -> bool:
        """Check if plan has changed since last run."""
        return self.state.get("plan_changed", False)

    def clear_plan_changed_flag(self) -> None:
        """Clear the plan changed flag."""
        self.state.pop("plan_changed", None)

    def refresh_plan_hash(self) -> None:
        """Re-stamp ``plan_hash`` to the plan's current on-disk content.

        `_load_or_create` hashes the plan once, at construction — before an
        apply-suggestions/apply-task-suggestions/apply-code-fixes/implement run
        makes its own edits. Without this, a run's `save()` persists that stale
        hash, and the *next* `StateManager` built on this plan sees its own
        edit as a human editing the plan mid-workflow (F5, audited 2026-09-16):
        `plan_changed` goes `True` and `implement_orchestrator.py` prints
        "Plan has changed since last session" for a change the plugin made.
        Call this once, after the run's writes are done, so the stamped hash
        matches what is actually on disk and `plan_changed` has nothing left
        to report.
        """
        self.state["plan_hash"] = self._compute_plan_hash()
        self.clear_plan_changed_flag()

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session state."""
        task_statuses = self.state.get("task_status", {})
        status_counts = {}
        for status in task_statuses.values():
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "plan_path": self.state.get("plan_path"),
            "created_at": self.state.get("created_at"),
            "updated_at": self.state.get("updated_at"),
            "branch": self.state.get("branch_name"),
            "review_phase_completed": self.state.get("review_phase_completed", False),
            "total_tasks": len(task_statuses),
            "status_counts": status_counts,
            "files_tracked": len(self.state.get("tracked_files", [])),
        }

    def reset(self) -> None:
        """Reset state to initial values (preserves plan reference)."""
        self.state = self._create_initial_state()

    def delete(self) -> None:
        """Delete the state file."""
        if self.state_file.exists():
            self.state_file.unlink()

    # --- Human Task Strategy ---

    def save_human_task_strategy(self, strategy: str) -> None:
        """Persist the chosen human task handling strategy.

        Args:
            strategy: One of "pause-and-ask", "skip-continue", "skip-dependents", "cancel"
        """
        self.state["human_task_strategy"] = strategy

    def get_human_task_strategy(self) -> Optional[str]:
        """Get the previously chosen human task handling strategy.

        Returns:
            Strategy string if set, None otherwise (user should be re-prompted)
        """
        return self.state.get("human_task_strategy")

    # --- Dependency Override Tracking ---

    def record_dependency_override(self, task_id: str, overridden_deps: List[str]) -> None:
        """Record that a task's human-task dependencies were overridden.

        Used with "skip-continue" strategy where human task dependencies are
        treated as satisfied even though the human tasks were skipped.

        Args:
            task_id: The task whose dependencies were overridden
            overridden_deps: List of human task IDs that were skipped but treated as satisfied
        """
        if "dependency_overrides" not in self.state:
            self.state["dependency_overrides"] = {}
        self.state["dependency_overrides"][task_id] = {
            "overridden_deps": overridden_deps,
            "timestamp": datetime.now().isoformat(),
        }

    def get_dependency_overrides(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get dependency override info for a task."""
        return self.state.get("dependency_overrides", {}).get(task_id)

    def get_all_dependency_overrides(self) -> Dict[str, Dict[str, Any]]:
        """Get all dependency overrides."""
        return self.state.get("dependency_overrides", {}).copy()

    # --- Human Decision Tracking ---

    def record_human_decision(
        self,
        phase: str,
        group_id: str,
        decision: str,
        reason: Optional[str] = None,
        batch_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a human decision for a validation item.

        This allows resuming interrupted sessions without re-asking
        the same questions.

        Args:
            phase: Phase identifier (e.g., "apply-suggestions", "apply-fixes")
            group_id: Stable hash-based identifier (from generate_group_id())
            decision: Decision made ("approved", "skipped", "deferred")
            reason: Optional reason for the decision
            batch_context: Optional batch context for batch decisions
        """
        key = f"human_decisions_{phase}"
        if key not in self.state:
            self.state[key] = {}

        decision_record = {
            "decision": decision,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        if batch_context:
            decision_record["batch_context"] = batch_context

        self.state[key][group_id] = decision_record

    def get_human_decision(self, phase: str, group_id: str) -> Optional[Dict[str, Any]]:
        """Get previously recorded human decision for an item."""
        key = f"human_decisions_{phase}"
        decisions = self.state.get(key, {})
        return decisions.get(group_id)

    def get_all_human_decisions(self, phase: str) -> Dict[str, Dict[str, Any]]:
        """Get all recorded human decisions for a phase."""
        key = f"human_decisions_{phase}"
        return self.state.get(key, {}).copy()

    def clear_human_decisions(self, phase: str) -> None:
        """Clear all human decisions for a phase (for fresh start)."""
        key = f"human_decisions_{phase}"
        self.state.pop(key, None)

    # --- Processing Progress Tracking ---

    def record_processing_progress(
        self,
        phase: str,
        total_items: int,
        processed_items: int,
        current_batch: int,
        total_batches: int
    ) -> None:
        """
        Record processing progress for resume capability.

        Args:
            phase: Phase identifier (e.g., "apply-suggestions", "apply-fixes")
            total_items: Total number of items to process
            processed_items: Number of items processed so far
            current_batch: Current batch number (0-indexed)
            total_batches: Total number of batches
        """
        key = f"progress_{phase}"
        self.state[key] = {
            "total_items": total_items,
            "processed_items": processed_items,
            "current_batch": current_batch,
            "total_batches": total_batches,
            "last_updated": datetime.now().isoformat()
        }

    def get_processing_progress(self, phase: str) -> Optional[Dict[str, Any]]:
        """Get processing progress for resume."""
        key = f"progress_{phase}"
        return self.state.get(key)

    # --- Processed Item Tracking ---

    def mark_item_processed(
        self,
        phase: str,
        group_id: str,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Mark an individual item as processed.

        Args:
            phase: Phase identifier (e.g., "apply-suggestions", "apply-fixes")
            group_id: Stable hash-based identifier (from generate_group_id())
            status: Processing status ("applied", "skipped", "failed")
            details: Optional details about the processing
        """
        mark_item_processed_in_state(self.state, phase, group_id, status, details)

    def get_processed_items(self, phase: str) -> Dict[str, Dict[str, Any]]:
        """Get all processed items for a phase."""
        key = f"processed_{phase}"
        return self.state.get(key, {}).copy()

    def is_item_processed(self, phase: str, group_id: str) -> bool:
        """Check if an item has already been processed."""
        key = f"processed_{phase}"
        return group_id in self.state.get(key, {})

    def clear_processed_items(self, phase: str) -> None:
        """Clear all processed items for a phase (for fresh start)."""
        key = f"processed_{phase}"
        self.state.pop(key, None)

    def clear_processing_progress(self, phase: str) -> None:
        """Clear processing progress for a phase."""
        key = f"progress_{phase}"
        self.state.pop(key, None)


def get_or_create_state(plan_path: Path, state_dir: Optional[Path] = None) -> StateManager:
    """
    Convenience function to get or create state for a plan.

    Args:
        plan_path: Path to the plan file
        state_dir: Deprecated, ignored. State now stored in plan directory.

    Returns:
        StateManager instance
    """
    return StateManager(plan_path)


def list_active_sessions(plans_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    List all active implementation sessions.

    Args:
        plans_dir: Directory containing plan files (searches for state.json in subdirs)

    Returns:
        List of session summaries with plan_path, status, and metadata
    """
    if plans_dir is None:
        plans_dir = Path('plans')

    sessions = []

    if not plans_dir.exists():
        return sessions

    # Search for state.json files in plan directories
    # e.g., plans/*/state.json
    for state_file in plans_dir.glob('*/state.json'):
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            sessions.append({
                'plan_path': state_file.parent.parent / f"{state_file.parent.name}.md",
                'state_file': state_file,
                'status': state.get('status', 'unknown'),
                'current_phase': state.get('current_phase'),
                'last_updated': state.get('updated_at'),
                'branch': state.get('branch_name'),
                'task_count': len(state.get('task_status', {})),
            })
        except (json.JSONDecodeError, OSError) as e:
            # Skip malformed or inaccessible state files
            logger.warning(f"Could not read state file {state_file}: {e}")
            continue

    return sorted(sessions, key=lambda s: s.get("last_updated", ""), reverse=True)
