"""The facts one write phase's run is about, shared by the scheduler's collaborators.

``WriteBatchScheduler`` hands a :class:`RunContext` to each collaborator it
composes (``write_verify.VerifyStep``, ``write_sweep.ContainmentSweep``) instead
of sharing ``self`` with them: what a collaborator may read about the run is the
list of fields below, and nothing it does can reassign one
(docs/development/write-scheduler.md#where-the-schedulers-helpers-live).

The objects behind the fields are the scheduler's own, not copies, and the
scheduler never rebinds one after construction. ``state`` is
the run's one ``StateManager``, the only writer of ``state.json`` for the run: a
collaborator writes through it, under the lock the scheduler hands it, and never
reloads it. ``log`` calls the scheduler's current ``log`` at each line, so a
caller that quiets the scheduler's log for a while (``_dispatch_walk``) quiets
its collaborators with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

from .write_safety import safety_dir

#: Write kinds whose ``output_name`` is keyed by the task id
#: (``subagent_tasks.implement_output_name``) rather than a batch position.
#: ``_warn_stale_result_files`` (N6) scopes its stale-file sweep to this set:
#: a positional kind's result filenames have never been renamed, so there is
#: no earlier naming to be stale relative to.
_ID_KEYED_WRITE_KINDS = frozenset({"implement"})


@dataclass(frozen=True)
class RunContext:
    """One write phase's run, as the scheduler resolved it at construction."""

    kind: str
    phase: str
    phase_dir: str
    output_dir: str
    repo_root: str
    anchor: Optional[str]
    tasks: Sequence[Dict[str, Any]]
    state: Any
    policy: Any
    audit: Any
    log: Callable[..., None]
    summary_file: Optional[str]
    dry_run: bool

    @property
    def safety_dir(self) -> str:
        """Where this phase's pre-batch snapshots, rollback store and audit
        log live (``write_safety.safety_dir``), read at each call: a jailed
        repo-wide run claims an out-of-grant directory part-way through."""
        return safety_dir(self.phase_dir)
