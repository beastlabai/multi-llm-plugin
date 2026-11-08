"""A real ``WriteBatchScheduler`` for tests that drive one of its methods.

Built through ``__init__`` with the fewest valid inputs, never through
``WriteBatchScheduler.__new__`` with a few attributes set by hand: ``__init__``
is the complete list of the scheduler's state, so production code reads
``self.<name>`` directly and a misspelt attribute raises.

Construction reads no model and writes nothing: it pins the repository root,
loads the write policy and records the phase's grant (about 2 ms).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

from utils.write_batch_scheduler import WriteBatchScheduler  # noqa: E402


def make_scheduler(
    directory: Any,
    *,
    kind: str,
    output_dir: Any = None,
    phase_dir: Any = None,
    phase: Optional[str] = None,
    state: Any = None,
    tasks: Sequence[Dict[str, Any]] = (),
    plan_file: Any = None,
    repo_root: Any = None,
    **over: Any,
) -> WriteBatchScheduler:
    """A scheduler for *kind* whose folders default to *directory*.

    The repository root is resolved from *plan_file* when one is given (the
    root ``util write-prompt`` resolves for the native leg), else it is
    *directory*. Any other constructor argument passes through *over*.
    """
    output_dir = str(output_dir if output_dir is not None else directory)
    if repo_root is None and plan_file is None:
        repo_root = directory
    return WriteBatchScheduler(
        tasks=list(tasks),
        phase_dir=str(phase_dir if phase_dir is not None else output_dir),
        kind=kind,
        output_dir=output_dir,
        phase=phase,
        state=state,
        plan_file=None if plan_file is None else str(plan_file),
        repo_root=None if repo_root is None else str(repo_root),
        log=over.pop("log", lambda line: None),
        **over,
    )
