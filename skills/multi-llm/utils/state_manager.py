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
from typing import Any, Dict, List, Optional, Tuple

from .git_utils import get_current_head, get_branch_name
from .schema_validator import validate_state_file

logger = logging.getLogger(__name__)

# --- Constants ---
ID_ALGO_VERSION = 1
CURRENT_FORMAT_VERSION = 2


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


def generate_group_id(group: Dict[str, Any]) -> str:
    """
    Generate a stable identifier for a group based on its content.

    Uses theme + normalized suggestions + reference fields to create
    a hash that remains stable across re-runs.

    Args:
        group: A group dictionary with theme, suggestions, etc.

    Returns:
        A 16-character hex string identifier
    """
    # Extract stable identifying fields
    theme = group.get("theme", "")
    suggestions = group.get("suggestions", group.get("issues", []))

    # Normalize suggestions for hashing
    normalized = []
    for s in suggestions:
        normalized.append({
            "type": s.get("type", ""),
            "section": s.get("section", s.get("reference", "")),
            "details": str(s.get("details", s.get("desc", "")))[:200],  # Truncate for stability
        })

    content = json.dumps({"theme": theme, "suggestions": normalized}, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def generate_suggestion_id(suggestion: Dict[str, Any]) -> str:
    """Generate a stable 16-char hex hash for a suggestion.

    Uses title + type + full description + reference (parent group excluded
    for stability across regrouping).
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
    content = json.dumps(fields, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def group_id_short(group_id_16: str) -> str:
    """Truncate a 16-char group/suggestion hash to 8 chars for display."""
    return group_id_16[:8]


def stamp_stable_ids(groups: List[Dict[str, Any]]) -> None:
    """Idempotently stamp group_hash, suggestion_hash, display_label, display_hash
    on each group and its suggestions. Includes collision detection with
    deterministic rehashing.

    Mutates groups in-place.
    """
    # Collect all canonical hashes for collision detection
    all_hashes: Dict[str, Tuple[str, int, int]] = {}  # hash -> (type, g_idx, s_idx)

    for g_idx, group in enumerate(groups):
        g_num = g_idx + 1
        # Group hash (reuse existing generate_group_id)
        ghash = generate_group_id(group)
        group["group_hash"] = ghash
        group["display_label"] = f"G{g_num}"

        suggestions = group.get("suggestions", group.get("issues", []))
        for s_idx, sugg in enumerate(suggestions):
            s_num = s_idx + 1
            # Generate suggestion hash with collision detection
            shash = generate_suggestion_id(sugg)

            # Check for canonical hash collision
            discriminator = 0
            base_hash = shash
            while shash in all_hashes:
                existing_type, existing_g, existing_s = all_hashes[shash]
                if existing_type == "suggestion":
                    # Same hash for different content — rehash with discriminator
                    discriminator += 1
                    if discriminator > 10:
                        raise ValueError(
                            f"Cannot resolve hash collision for suggestion "
                            f"G{g_num}S{s_num} after 10 attempts"
                        )
                    # Rehash with discriminator suffix
                    fields = {
                        "v": ID_ALGO_VERSION,
                        "title": canonicalize_hash_input(sugg.get("title", "")),
                        "type": canonicalize_hash_input(sugg.get("type", "")),
                        "description": canonicalize_hash_input(
                            sugg.get("details", sugg.get("desc", sugg.get("description", "")))
                        ),
                        "reference": canonicalize_hash_input(
                            sugg.get("reference", sugg.get("section", ""))
                        ),
                        "discriminator": f"-{discriminator}",
                    }
                    content = json.dumps(fields, sort_keys=True)
                    shash = hashlib.sha256(content.encode()).hexdigest()[:16]
                else:
                    break  # Collision with a group hash is fine (different namespace)

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

    def __init__(self, plan_path: Path, state_dir: Optional[Path] = None):
        """
        Initialize state manager.

        Args:
            plan_path: Path to the plan being implemented
            state_dir: Deprecated, ignored. State now stored in plan directory.
        """
        self.plan_path = Path(plan_path).resolve()

        # State file lives in the plan's output directory
        from .output_handler import get_output_dir
        self.plan_dir = get_output_dir(self.plan_path)
        self.state_file = self.plan_dir / "state.json"

        self.state: Dict[str, Any] = {}
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
                with open(old_state_file, 'r') as f:
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
        if self.plan_path.exists():
            content = self.plan_path.read_text(encoding="utf-8")
            return hashlib.sha256(content.encode()).hexdigest()[:16]
        return ""

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
                with open(self.state_file, "r", encoding="utf-8") as f:
                    self.state = json.load(f)

                # Check for plan changes
                current_hash = self._compute_plan_hash()
                if self.state.get("plan_hash") != current_hash:
                    self.state["plan_hash"] = current_hash
                    self.state["plan_changed"] = True

            except json.JSONDecodeError:
                self.state = self._create_initial_state()
        else:
            self.state = self._create_initial_state()

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
        """
        self.state["updated_at"] = datetime.now().isoformat()

        # Validate state before saving
        is_valid, errors = self._validate_state(self.state)
        if not is_valid:
            # Log validation errors but allow save to proceed
            import warnings
            warnings.warn(f"State validation warnings: {errors}")

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
                json.dump(self.state, f, indent=2)
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

    def get(self, key: str, default: Any = None) -> Any:
        """Get a state value."""
        return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a state value."""
        self.state[key] = value

    def update_task_status(self, task_id: str, status: str, reason: Optional[str] = None) -> None:
        """Update status of a task.

        Args:
            task_id: Task identifier
            status: New status value
            reason: Optional reason for the status change (e.g., why a task was skipped)
        """
        if "task_status" not in self.state:
            self.state["task_status"] = {}
        self.state["task_status"][task_id] = status

        if reason is not None:
            if "task_status_reasons" not in self.state:
                self.state["task_status_reasons"] = {}
            self.state["task_status_reasons"][task_id] = reason

    def get_task_status(self, task_id: str) -> Optional[str]:
        """Get status of a task."""
        return self.state.get("task_status", {}).get(task_id)

    def get_task_status_reason(self, task_id: str) -> Optional[str]:
        """Get the reason for a task's current status."""
        return self.state.get("task_status_reasons", {}).get(task_id)

    def get_all_task_statuses(self) -> Dict[str, str]:
        """Get all task statuses."""
        return self.state.get("task_status", {}).copy()

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
        if "phases_completed" not in self.state:
            self.state["phases_completed"] = {}
        self.state["phases_completed"][phase] = datetime.now().isoformat()

    def is_phase_completed(self, phase: str) -> bool:
        """
        Check if a workflow phase is completed.

        Args:
            phase: Phase name to check

        Returns:
            True if the phase has been marked as completed
        """
        return phase in self.state.get("phases_completed", {})

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
        key = f"processed_{phase}"
        if key not in self.state:
            self.state[key] = {}

        self.state[key][group_id] = {
            "status": status,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }

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
            with open(state_file, 'r') as f:
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
