#!/usr/bin/env python3
"""`--full`, in Python: a resumable state machine over the other modes (D1a/D1e).

``instructions/full-workflow.md`` used to *be* the sequencer: 460 lines telling a
driving agent which orchestrator to shell out to next, when to skip a phase, and
what to do when one failed. That made the whole plan-to-code workflow reachable on
exactly one host. This module is the same sequence as code — the markdown stays as
narrative, the **sequencing** lives here — so a single-shot host reaches it with
one invocation and a detached run can own it end to end.

What that buys, and what it costs:

* **One state machine, on the existing ``state.json``.** There is no second
  progress file: a phase is done when *it* recorded itself done
  (``phases_completed`` / ``phases_skipped``), which is the same record `status`
  reports and the same one every standalone invocation writes. So resume is not a
  feature bolted on here — it is what the loop already reads. Every phase boundary
  is a resume point.
* **The outcome is verified, never the phase's claim.** A phase that returns 0 but
  did not record itself complete (the ordinary native-host case: it printed
  ``[VALIDATION_PENDING]`` and stopped for a subagent to run) is *not* progress, so
  the workflow stops there with exit **3**, naming the phase, the follow-up that
  phase printed, and the ``--resume`` re-invocation. That is D6 row 14: under
  ``--yes`` a hard failure is terminal, never a question.
* **This is ``ensure_prerequisites``' missing caller.** Rows 5 and 6 auto-run the
  missing apply phase **once** and re-check (D6a); the runner is injected from
  here, because applying suggestions needs a subagent channel this module has and
  ``check_workflow_prerequisites.py`` does not. The auto-run is bounded here too:
  at most once per phase per run, and a second unmet re-check is exit 3 rather than
  a retry.
* **No phase is entered on a guess.** Row 12 (review the tasks?) resolves from
  ``--yes`` / ``--review-tasks`` and records a skip otherwise; the two conditional
  apply phases ask the *existing* prerequisite counters whether there is anything
  to apply, and record their skip when there is not — so `status` never shows a
  phase as perpetually pending.

Usage::

    uv run --project <skill-dir> -- python <skill-dir>/multi_llm.py \\
        full <plan> [--yes] [--models …|--quick] [--review-tasks] [--resume]
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Absolute, symlinks preserved — see utils/cli_common.py.
sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent))

import modes as mode_table
from check_workflow_prerequisites import (
    check_apply_task_suggestions_prerequisite,
    ensure_prerequisites,
    record_prerequisite_skip,
)
from utils.cli_common import (
    EXIT_ERROR,
    EXIT_NEEDS_HUMAN,
    EXIT_NOTHING_TO_DO,
    EXIT_OK,
    active_result,
    build_parent_parser,
    dispatcher_command,
    record_artifact,
    record_next,
)
from utils.output_handler import get_output_dir, get_phase_dir
from utils.state_manager import StateManager
from utils.stream_bootstrap import bootstrap_streams
from utils import unattended as unattended_mod
from utils.unattended import NeedsHuman

#: The mode word, and the run-dir name ``--detach`` claims. `full` records no
#: workflow phase of its own — each phase it runs records itself — but it does own
#: a directory for ``.status.json``, the run log and the lock.
MODE_NAME = "full"
RUN_DIR_PHASE = "full"

#: Where the workflow's own bookkeeping lands in ``state.json``. Display and
#: diagnostics only: the resume decision reads ``phases_completed`` /
#: ``phases_skipped``, never this key, so a hand-deleted record cannot make the
#: workflow re-run a phase that really did complete.
STATE_KEY = "full_workflow"

#: Step outcomes the driver branches on.
DONE = "done"
SKIPPED = "skipped"
ALREADY = "already-complete"
FAILED = "failed"

#: The phases whose real work is a **batch loop an invoker runs**, mapped to the
#: directory each leaves that loop's instruction document in. They are the one
#: place ``state.json`` cannot answer "did it happen?" on its own:
#: ``apply-suggestions`` (``marks_phase_completed = True``) and ``implement``
#: record themselves completed the moment the document is written — before a
#: single edit exists — so a recorded completion here is a *claim about work
#: handed out*, not about work done. ``apply-code-fixes``'s mode word is not its
#: phase dir: the orchestrator's own ``phase_name`` is ``apply-fixes``.
WRITE_PHASE_DIRS: Dict[str, str] = {
    "apply-suggestions": "apply-suggestions",
    "apply-task-suggestions": "apply-task-suggestions",
    "apply-code-fixes": "apply-fixes",
}

#: The fourth write phase. Its document is ``--output``, not a plan-dir file.
IMPLEMENT_PHASE = "implement"


@dataclass
class StepReport:
    """What one step did, for the final report."""

    label: str
    phase: str
    outcome: str
    detail: str = ""
    exit_code: Optional[int] = None


@dataclass
class Step:
    """One entry in the sequence.

    Attributes:
        phase: The ``state.json`` phase this step must end up recording, or ``""``
            for an advisory step that records nothing (consolidation).
        label: How the report names it.
        run: ``() -> (outcome, detail, exit_code)``.
        skippable: Whether a recorded completion/skip lets the driver pass over it
            (every real phase; never the advisory ones, which are cheap and
            idempotent).
    """

    phase: str
    label: str
    run: Callable[[], Tuple[str, str, Optional[int]]]
    skippable: bool = True


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="full_workflow_orchestrator.py",
        description=(
            "Run the whole plan-to-code workflow as one resumable sequence: "
            "review-plan, apply-suggestions, generate-tasks, review-tasks, "
            "apply-task-suggestions, implement, review-code, apply-code-fixes."
        ),
        parents=[build_parent_parser()],
    )
    parser.add_argument(
        "--plan-file", required=True, help="Path to the plan markdown file"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="PROVIDER:MODEL",
        help="Models for the two fan-out phases (D6 row 2). Omit to use "
             "providers.yaml defaults; --quick is the other non-interactive answer.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the quick model set for the fan-out phases (D6 row 2).",
    )
    parser.add_argument(
        "--review-tasks",
        action="store_true",
        help="Opt in to the optional task-review phase (D6 row 12). Implied by "
             "--yes; without either, the phase is recorded skipped rather than "
             "left perpetually pending.",
    )
    parser.add_argument(
        "--consolidate",
        action="store_true",
        help="Force the post-review-plan consolidation step (D6 row 4). Under "
             "--yes it auto-runs when review-plan recommends it.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the first phase state.json does not record as "
             "completed or skipped (D6 row 14's named re-invocation). This is also "
             "the default behavior; the flag additionally forwards --resume to the "
             "phases that support it, so partially processed batches are not "
             "re-applied.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run every phase even when state.json records it completed or "
             "skipped. Not forwarded to the phases (there --force means "
             "'re-invoke the models'); it only clears this loop's skipping.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


# ---------------------------------------------------------------------------
# The workflow
# ---------------------------------------------------------------------------

class FullWorkflow:
    """The state machine. One instance per invocation."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.plan_path = Path(os.path.abspath(args.plan_file))
        self.unattended = bool(getattr(args, "yes", False))
        self.plan_dir = get_output_dir(self.plan_path)
        self.reports: List[StepReport] = []
        #: Phases whose apply step this run already auto-ran (D6a: at most once
        #: per phase per run — a second unmet re-check is exit 3, not a retry).
        self.auto_ran: List[str] = []
        self._state: Optional[StateManager] = None

    # -- state ----------------------------------------------------------
    @property
    def state(self) -> StateManager:
        """A freshly read ``StateManager``.

        Re-read rather than cached: each phase runs in this process and writes its
        own completion through its own manager instance, so a cached copy would
        answer "is this phase done?" from before the phase ran.
        """
        return StateManager(self.plan_path)

    def _recorded(self, phase: str) -> Optional[str]:
        """``"completed"``/``"skipped"`` when *phase* is already accounted for."""
        state = self.state
        if state.is_phase_completed(phase):
            return "completed"
        if state.is_phase_skipped(phase):
            return "skipped"
        return None

    def _record_skip(self, phase: str, reason: str) -> Tuple[str, str, Optional[int]]:
        """Record *phase* skipped, unless it is already recorded completed.

        The guard matters under ``--force``, where the driver re-enters a phase the
        conditional counters would report as "nothing to apply" *because* it already
        ran — recording a skip over a completed phase would make `status` show both.
        """
        state = self.state
        if state.is_phase_completed(phase):
            return ALREADY, "completed", None
        record_prerequisite_skip(state, phase, reason=reason)
        return SKIPPED, reason, None

    def _record_progress(self, *, current: str, failed: Optional[str] = None) -> None:
        state = self.state
        record = dict(state.get(STATE_KEY) or {})
        record.update({
            "current_phase": current,
            "failed_phase": failed,
            "steps": {r.phase or r.label: r.outcome for r in self.reports},
        })
        state.set(STATE_KEY, record)
        state.save()

    # -- in-process dispatch -------------------------------------------
    def run_mode(self, mode_word: str, extra: Sequence[str]) -> int:
        """Call another mode's ``main(argv)`` in-process (D1a's dispatch rule).

        Deliberately not ``multi_llm.main``: the dispatcher owns one
        ``MULTI_LLM_RESULT`` line and one result context per invocation, and
        nesting it would emit a second line and deactivate the outer context. The
        *target* is resolved through the same ``modes.py`` table the dispatcher
        uses, so there is still exactly one mode → module mapping.
        """
        mode = mode_table.resolve(mode_word)
        if mode is None or not mode.target_module:  # pragma: no cover - table-guarded
            raise RuntimeError(f"no dispatch target for mode {mode_word!r}")
        target = getattr(
            importlib.import_module(mode.target_module), mode.target_attr
        )
        argv = ["--plan-file", str(self.plan_path), *extra]
        print(f"\n$ {dispatcher_command(mode.name, self.plan_path, *extra)}")
        self._clear_next()
        try:
            if inspect.iscoroutinefunction(target):
                import asyncio

                result = asyncio.run(target(argv))
            else:
                result = target(argv)
        except SystemExit as exc:
            code = exc.code
            if code is None:
                return EXIT_OK
            if isinstance(code, int):
                return code
            print(str(code), file=sys.stderr)
            return EXIT_ERROR
        except NeedsHuman as exc:
            print(exc.message, file=sys.stderr)
            return exc.exit_code
        return EXIT_OK if result is None else int(result)

    @staticmethod
    def _clear_next() -> None:
        """Drop a previous phase's follow-up so a stale one is never re-reported."""
        ctx = active_result()
        if ctx is not None:
            ctx.next = None

    @staticmethod
    def _pending_next() -> Optional[str]:
        ctx = active_result()
        return ctx.next if ctx is not None else None

    # -- flag forwarding -----------------------------------------------
    def common_flags(self) -> List[str]:
        """Cross-cutting flags every phase gets, and only those.

        ``--detach``/``--if-not-running``/``--max-detached`` are deliberately
        absent: this run already crossed the detach boundary, and a phase that
        forked again would leave the workflow polling a child it does not own.
        The ``--internal-*`` re-entry flags are absent for the same reason.
        """
        flags: List[str] = []
        if getattr(self.args, "host", None):
            flags += ["--host", str(self.args.host)]
        if self.unattended:
            flags += ["--yes"]
        elif getattr(self.args, "interactive", False):
            flags += ["--interactive"]
        if getattr(self.args, "exec_subagents", False):
            flags += ["--exec-subagents"]
        if getattr(self.args, "dry_run_subagents", False):
            flags += ["--dry-run-subagents"]
        if getattr(self.args, "skip_prerequisites", False):
            flags += ["--skip-prerequisites"]
        return flags

    def model_flags(self) -> List[str]:
        """Row 2, resolved once and reused: the same models in both fan-out phases."""
        if self.args.models:
            return ["--models", *self.args.models]
        if self.args.quick:
            return ["--quick"]
        return []

    def apply_flags(self) -> List[str]:
        """The apply phases' unattended answers (rows 10/11).

        Both flags are gated the same way, because both *are* the unattended
        dispositions of those two rows — the core appends ``--claude-decide
        --no-confirm`` only under ``--yes``. Attended, the no-selection
        confirmation stays a real gate: the phase prints ``confirmation_needed``
        and records nothing, which this driver reads as "did not finish" and ends
        at exit 3 naming the phase (row 14). Silently applying every valid item
        nobody selected is the one thing an attended run must not do.
        """
        flags: List[str] = []
        if self.unattended:
            flags += ["--claude-decide", "--no-confirm"]
        if getattr(self.args, "resume", False):
            flags += ["--resume"]
        return flags

    # -- the prerequisite gates (D6a rows 5/6) --------------------------
    def apply_runner(self, site_id: str) -> Optional[Callable[[str], int]]:
        """The runner ``ensure_prerequisites`` auto-runs, bounded per phase.

        Supplied only under ``--yes``: the D6 table's auto-run is the *unattended*
        resolution of rows 5/6. Attended, the gate keeps its standalone ending and
        names the fix, which in the normal sequence never fires — Phase 2 runs
        before Phase 3, so the gate is satisfied rather than repaired.
        """
        if not self.unattended:
            return None

        def runner(phase: str) -> int:
            if phase in self.auto_ran:
                print(
                    f"[PREREQUISITE_UNMET] {phase} was already auto-run once in "
                    f"this run and the gate is still unmet; not retrying (D6a: at "
                    f"most once per phase per run).",
                    file=sys.stderr,
                )
                return EXIT_NEEDS_HUMAN
            self.auto_ran.append(phase)
            unattended_mod.record_resolution(
                str(self.run_dir),
                site_id,
                f"auto-ran {phase} once, then re-checked the gate",
            )
            return self.run_mode(phase, self.apply_flags() + self.common_flags())

        return runner

    def gate(self, mode_word: str, site_id: str) -> Optional[str]:
        """``None`` when *mode_word* may run, else the blocking message."""
        outcome = ensure_prerequisites(
            self.plan_path,
            mode_word,
            unattended=self.unattended,
            in_workflow=True,
            skip_prerequisites=bool(getattr(self.args, "skip_prerequisites", False)),
            apply_runner=self.apply_runner(site_id),
        )
        for notice in outcome.notices:
            print(notice)
        if outcome.met:
            return None
        return outcome.message or (
            f"[PREREQUISITE_UNMET] {mode_word}: prerequisites are not met."
        )

    # -- run dir --------------------------------------------------------
    @property
    def run_dir(self) -> Path:
        return get_phase_dir(self.plan_path, RUN_DIR_PHASE)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
    def steps(self) -> List[Step]:
        return [
            Step("review-plan", "Phase 1: Review Plan", self.step_review_plan),
            Step("", "Phase 1b: Consolidation", self.step_consolidate,
                 skippable=False),
            Step("apply-suggestions", "Phase 2: Apply Suggestions",
                 self.step_apply_suggestions),
            Step("generate-tasks", "Phase 3: Generate Tasks",
                 self.step_generate_tasks),
            Step("review-tasks", "Phase 3b: Review Tasks", self.step_review_tasks),
            Step("apply-task-suggestions", "Phase 3c: Apply Task Suggestions",
                 self.step_apply_task_suggestions),
            Step("implement", "Phase 4: Implement", self.step_implement),
            Step("code-review", "Phase 5: Review Code", self.step_review_code),
            Step("apply-code-fixes", "Phase 6: Apply Code Fixes",
                 self.step_apply_code_fixes),
        ]

    def step_review_plan(self) -> Tuple[str, str, Optional[int]]:
        code = self.run_mode(
            "review-plan", self.model_flags() + self.common_flags()
        )
        return self._verify("review-plan", code)

    def step_consolidate(self) -> Tuple[str, str, Optional[int]]:
        """Row 4: auto-run consolidation when review-plan recommended it.

        Advisory by construction — consolidation clusters the *report*, it does not
        gate anything — so a failure here is a warning and the workflow continues.
        Without ``--exec-subagents`` the consolidate pass prints its batch marker
        for a native invoker and returns; that is the same "not progress" state as
        any other pending batch, but it cannot block a workflow whose next phase
        does not read consolidated output.
        """
        if not self.args.consolidate:
            if not self.unattended:
                return SKIPPED, "not requested (--consolidate)", None
            if not self._consolidation_recommended():
                return SKIPPED, "not recommended by review-plan", None
            unattended_mod.record_resolution(
                str(self.run_dir),
                unattended_mod.row(4).site_id,
                "auto-ran consolidation once ([CONSOLIDATION_RECOMMENDED])",
            )
        code = self.run_mode(
            "review-plan", ["--consolidate", *self.common_flags()]
        )
        if code != EXIT_OK:
            print(
                f"multi-llm warning: consolidation exited {code}; continuing "
                f"(it clusters the review report and gates nothing).",
                file=sys.stderr,
            )
            return SKIPPED, f"consolidation exited {code} (non-fatal)", code
        return DONE, "", code

    def _consolidation_recommended(self) -> bool:
        """The same threshold ``review_plan_orchestrator`` prints its marker on."""
        from utils.consolidation import CONSOLIDATION_RECOMMENDED_THRESHOLD

        entries = _validation_entries(self.plan_dir / "review-plan" / "validation.json")
        actionable = sum(
            1 for entry in entries
            if entry.get("status") in ("valid", "needs-human-decision")
        )
        return actionable >= CONSOLIDATION_RECOMMENDED_THRESHOLD

    def step_apply_suggestions(self) -> Tuple[str, str, Optional[int]]:
        code = self.run_mode(
            "apply-suggestions", self.apply_flags() + self.common_flags()
        )
        return self._verify("apply-suggestions", code)

    def step_generate_tasks(self) -> Tuple[str, str, Optional[int]]:
        """Phase 3, through ``generate_tasks_orchestrator`` — never an inlined prompt.

        Called as a Python function rather than through :meth:`run_mode` for one
        reason: this is where ``ensure_prerequisites``' ``apply_runner`` is
        injected (row 5). Everything else about the phase — materialization,
        dispatch, validation, the in-process ``update_plan_tasks`` and the
        ``TASKS_FILE`` verification — is the standalone mode's code, unchanged.
        """
        import generate_tasks_orchestrator as generate_tasks

        child = argparse.Namespace(**vars(self.args))
        child.reaggregate = False
        # `full --force` re-runs phases; it never re-invokes the models — with one
        # exception, and it is the one the phase's own failure message names. A
        # `task_generation_result.json` a native invoker persisted can be
        # schema-invalid, and the resume branch re-consumes it forever: every
        # `--resume` *and* `--force` would fail identically at consume time while
        # the printed fix ("re-run with --force to re-dispatch") pointed at a flag
        # this driver swallowed. So when the phase has no recorded completion,
        # `--force` travels: there is no earlier progress a re-dispatch could
        # discard. Once it *is* recorded, force means "re-run the phase" again.
        child.force = bool(getattr(self.args, "force", False)) and (
            self._recorded("generate-tasks") is None
        )
        child.detach = False
        child.internal_worker = False
        extra = list(self.common_flags())
        if child.force:
            extra += ["--force"]
        print(f"\n$ {dispatcher_command('generate-tasks', self.plan_path, *extra)}")
        self._clear_next()
        try:
            code = generate_tasks.run_generate_tasks(
                child,
                in_workflow=True,
                apply_runner=self.apply_runner(unattended_mod.row(5).site_id),
            )
        except SystemExit as exc:  # pragma: no cover - defensive
            code = EXIT_OK if exc.code is None else int(exc.code)
        except NeedsHuman as exc:
            print(exc.message, file=sys.stderr)
            code = exc.exit_code
        # `--full` never proceeds past a failed generation: implementing against a
        # plan with no tasks is the error `--implement`'s auto-detection already
        # reports, and every later phase would treat an invented list as ground
        # truth.
        return self._verify("generate-tasks", code)

    def step_review_tasks(self) -> Tuple[str, str, Optional[int]]:
        """Row 12: run it under ``--yes``/``--review-tasks``, else record the skip."""
        if not (self.unattended or self.args.review_tasks):
            return self._record_skip(
                "review-tasks", "not requested (--review-tasks not passed)"
            )
        if self.unattended and not self.args.review_tasks:
            unattended_mod.record_resolution(
                str(self.run_dir),
                unattended_mod.row(12).site_id,
                "ran review-tasks automatically",
            )
        code = self.run_mode(
            "review-tasks", self.model_flags() + self.common_flags()
        )
        return self._verify("review-tasks", code)

    def step_apply_task_suggestions(self) -> Tuple[str, str, Optional[int]]:
        """The three-case conditional, answered by the existing counter.

        (a) review-tasks was skipped, (b) it ran and found nothing, (c) it found
        findings. Cases (a) and (b) are recorded skips — ``check_workflow_``
        ``prerequisites.check_apply_task_suggestions_prerequisite`` already
        distinguishes them, so the decision is not re-derived here.
        """
        state = self.state
        if state.is_phase_skipped("review-tasks"):
            return self._record_skip(
                "apply-task-suggestions", "Skipped: review-tasks was not executed"
            )
        check = check_apply_task_suggestions_prerequisite(state, self.plan_dir)
        if check.get("met"):
            return self._record_skip(
                "apply-task-suggestions",
                f"Skipped: {check.get('reason', 'nothing to apply')}",
            )
        code = self.run_mode(
            "apply-task-suggestions", self.apply_flags() + self.common_flags()
        )
        return self._verify("apply-task-suggestions", code)

    def step_implement(self) -> Tuple[str, str, Optional[int]]:
        blocked = self.gate("implement", unattended_mod.row(6).site_id)
        if blocked:
            print(blocked, file=sys.stderr)
            return FAILED, "prerequisites unmet (D6 row 6)", EXIT_NEEDS_HUMAN
        extra = list(self.common_flags())
        if getattr(self.args, "resume", False):
            extra += ["--resume"]
        code = self.run_mode("implement", extra)
        return self._verify("implement", code)

    def step_review_code(self) -> Tuple[str, str, Optional[int]]:
        code = self.run_mode(
            "review-code", self.model_flags() + self.common_flags()
        )
        return self._verify("code-review", code)

    def step_apply_code_fixes(self) -> Tuple[str, str, Optional[int]]:
        if not _grouped_has_suggestions(self.plan_dir / "code-review" / "grouped.json"):
            return self._record_skip(
                "apply-code-fixes",
                "Skipped: code review produced zero actionable findings",
            )
        code = self.run_mode(
            "apply-code-fixes", self.apply_flags() + self.common_flags()
        )
        return self._verify("apply-code-fixes", code)

    # ------------------------------------------------------------------
    # The driver
    # ------------------------------------------------------------------
    def _write_phase_document(self, phase: str) -> Optional[Path]:
        """Where *phase* left its batches for an invoker, or ``None``.

        ``None`` for every read/analyze phase: they have no batch loop, so there is
        nothing for this check to look at.
        """
        subdir = WRITE_PHASE_DIRS.get(phase)
        if subdir:
            return self.plan_dir / subdir / "orchestrator_output.json"
        if phase != IMPLEMENT_PHASE:
            return None
        # `implement` writes its batches to `--output`, which this driver never
        # passes — so the phase used its own default. Recomputed through the
        # phase's own function (never a second spelling of that path) and only
        # once its git-root precondition holds, so the probe cannot exit for it.
        try:
            import implement_orchestrator

            if not implement_orchestrator.get_project_root(str(self.plan_path)):
                return None
            return Path(
                implement_orchestrator.resolve_default_output(
                    Path(self.plan_path).resolve()
                )
            )
        except (ImportError, SystemExit):  # pragma: no cover - defensive
            return None

    def _unexecuted_batches(self, phase: str) -> int:
        """How many batches *phase* emitted that nothing in this run could apply.

        ``full`` runs every phase **inside this one process**: there is no invoker
        between a phase and the next one, so the only thing that can execute a
        write phase's batches here is the write scheduler under
        ``--exec-subagents`` (which finalizes — and therefore completes the phase —
        only after every batch verified, D3n). Without it the batches are handed to
        an invoker that never arrives, and the two phases that mark themselves
        completed at emit time would otherwise report that as progress.

        Zero is the answer whenever there is no evidence of a pending loop — no
        document, an unreadable one, one belonging to another plan, or one
        declaring no batches (the honest "nothing to apply" completion this must
        not turn into a failure).
        """
        if getattr(self.args, "exec_subagents", False):
            return 0
        document = self._write_phase_document(phase)
        if document is None or not document.is_file():
            return 0
        try:
            payload = json.loads(document.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return 0
        if not isinstance(payload, dict):
            return 0
        declared = payload.get("plan_file")
        if declared and os.path.realpath(str(declared)) != os.path.realpath(
            str(self.plan_path)
        ):
            # `implement`'s default output lives in a shared `.multi-llm/tmp/`;
            # a document for another plan says nothing about this phase.
            return 0
        batches = payload.get("batches")
        return len(batches) if isinstance(batches, list) else 0

    def _unexecuted_write_phase(self, phase: str, pending: int) -> str:
        """Print the row-14 explanation for *phase*, and return the report detail."""
        follow_up = dispatcher_command(
            phase, self.plan_path, *(["--yes"] if self.unattended else []),
        )
        document = self._write_phase_document(phase)
        print(
            "\n".join([
                f"[WRITE_PHASE_NOT_EXECUTED] {phase} recorded itself completed "
                f"after writing {pending} batch(es) to",
                f"       {document}",
                "       — but nothing applied them: `full` runs every phase "
                "inside this one process,",
                "       so with --exec-subagents unset there is no invoker "
                "between this phase and the",
                "       next. Run the phase on its own, so your host's subagents "
                "execute its batches,",
                "       then resume the workflow — or re-run the whole workflow "
                "with --exec-subagents,",
                "       which applies the batches in process.",
            ]),
            file=sys.stderr,
        )
        record_next(follow_up)
        return (
            f"exited 0 and recorded itself completed, but the {pending} batch(es) "
            f"it emitted were never executed"
        )

    def _verify(self, phase: str, code: int) -> Tuple[str, str, Optional[int]]:
        """A phase counts as progress only if *it* recorded itself (D2i's rule).

        Exit 0 is the phase's claim; ``state.json`` is the artifact. The common way
        the two disagree is the native-invoker path — a fan-out that stopped at
        ``[VALIDATION_PENDING]`` exits 0 with the phase deliberately incomplete —
        and marching on from there would review, implement and code-review against
        output nobody produced.

        A write phase needs the *second* artifact check
        (:meth:`_unexecuted_batches`), because its record is not that evidence: two
        of the four mark themselves completed when they emit their batch document,
        which is before any edit exists. Trusting it there would generate tasks
        from a plan whose suggestions were never applied and code-review an
        unimplemented tree — the same claim-vs-artifact failure, one phase further
        on.
        """
        recorded = self._recorded(phase)
        if code in (EXIT_OK, EXIT_NOTHING_TO_DO) and recorded:
            pending = self._unexecuted_batches(phase)
            if pending:
                return FAILED, self._unexecuted_write_phase(phase, pending), code
            return DONE, recorded, code
        if code in (EXIT_OK, EXIT_NOTHING_TO_DO):
            return (
                FAILED,
                f"exited {code} but state.json records no completion for "
                f"'{phase}', so the phase did not finish",
                code,
            )
        return FAILED, f"exited {code}", code

    def _phase_failed(self, step: Step, detail: str, code: Optional[int]) -> int:
        """D6 row 14: exit 3, name the phase and the ``--resume`` re-invocation."""
        resume = dispatcher_command(
            MODE_NAME, self.plan_path, *(["--yes"] if self.unattended else []),
            "--resume",
        )
        pending = self._pending_next()
        lines = [
            "",
            f"ERROR: {step.label} did not complete ({detail}). The workflow "
            f"stopped here; no later phase ran.",
        ]
        if pending:
            lines += [
                "       That phase printed a follow-up of its own — run it first:",
                f"         {pending}",
            ]
        lines += [
            "       Then continue the workflow from this phase:",
            f"         {resume}",
            "       A failed phase is never a question, even attended: nothing "
            "makes a hard failure proceed (D6 row 14).",
        ]
        print("\n".join(lines), file=sys.stderr)
        self._record_progress(current=step.phase or step.label,
                              failed=step.phase or step.label)
        unattended_mod.mark_needs_human(MODE_NAME)
        # The machine-readable follow-up is the *next* command to run, and when the
        # phase printed a prerequisite of its own that is that one — resuming
        # before it re-materializes the same pending state and points back here, so
        # a host that follows `next` mechanically would loop. The prose above still
        # names both, in order.
        record_next(pending or resume)
        return EXIT_NEEDS_HUMAN

    def run(self) -> int:
        for step in self.steps():
            if step.skippable and not self.args.force:
                recorded = self._recorded(step.phase)
                if recorded:
                    print(f"\n=== {step.label}: already {recorded}, skipping ===")
                    self.reports.append(
                        StepReport(step.label, step.phase, ALREADY, recorded)
                    )
                    continue
            print(f"\n=== {step.label} ===")
            self._record_progress(current=step.phase or step.label)
            outcome, detail, code = step.run()
            self.reports.append(
                StepReport(step.label, step.phase, outcome, detail, code)
            )
            if outcome == FAILED:
                self.print_report()
                return self._phase_failed(step, detail, code)
            if outcome == SKIPPED and detail:
                print(f"[SKIPPED] {step.label}: {detail}")
        self._record_progress(current="complete")
        self.print_report()
        record_next(None)
        return EXIT_OK

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def print_report(self) -> None:
        print("\n## Full Workflow Summary\n")
        print(f"**Plan**: {self.plan_path}")
        print("")
        for report in self.reports:
            suffix = f" — {report.detail}" if report.detail else ""
            print(f"- {report.label}: {report.outcome}{suffix}")
        artifacts = self._artifacts()
        if artifacts:
            print("\n### Output files\n")
            for path in artifacts:
                print(f"- {path}")
                record_artifact(path)
        print("")
        self._print_metrics()

    def _artifacts(self) -> List[str]:
        """The per-phase artifacts a reader is most likely to open next."""
        candidates = [
            self.plan_dir / "review-plan" / "report.html",
            self.plan_dir / "review-plan" / "consolidated-report.md",
            self.plan_dir / "tasks" / "tasks.md",
            self.plan_dir / "review-tasks" / "report.html",
            self.plan_dir / "apply-task-suggestions" / "orchestrator_output.json",
            self.plan_dir / "code-review" / "report.html",
        ]
        return [str(path) for path in candidates if path.exists()]

    def _print_metrics(self) -> None:
        """Combined resource usage, when the run recorded any."""
        try:
            from utils import metrics

            state_file = self.plan_dir / "state.json"
            if not state_file.exists():
                return
            report = metrics.generate_all_phases_report(state_file)
        except Exception:  # noqa: BLE001 - a report must never fail a green run
            return
        if report:
            print(report)


# ---------------------------------------------------------------------------
# Small readers (shapes the existing counters already tolerate)
# ---------------------------------------------------------------------------

def _validation_entries(path: Path) -> List[Dict[str, Any]]:
    """``validation.json`` entries, v1 flat dict or v2 envelope."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []
    if isinstance(data, dict) and "groups" in data:
        entries = data["groups"]
    elif isinstance(data, dict):
        entries = list(data.values())
    elif isinstance(data, list):
        entries = data
    else:
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _grouped_has_suggestions(path: Path) -> bool:
    """Whether a ``grouped.json`` carries anything to apply (v1 list or v2 envelope)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    if isinstance(data, dict) and "groups" in data:
        groups = data["groups"]
    elif isinstance(data, list):
        groups = data
    else:
        return False
    return any(
        isinstance(group, dict) and group.get("suggestions") for group in groups
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    bootstrap_streams()
    args = parse_args(argv)

    plan_path = Path(os.path.abspath(args.plan_file))
    if not plan_path.is_file():
        print(f"ERROR: Plan file not found: {args.plan_file}", file=sys.stderr)
        print(f"       Resolved path: {plan_path}", file=sys.stderr)
        return EXIT_ERROR

    workflow = FullWorkflow(args)

    # The detach boundary: after pre-flight (plan readable, flags parsed, run dir
    # created) and before the first phase runs, so a bad flag fails on the
    # caller's terminal with no child and no log.
    from utils.review_orchestrator_base import maybe_detach

    detached = maybe_detach(
        args=args,
        mode=MODE_NAME,
        phase_dir=str(workflow.run_dir),
        plan_file=str(plan_path),
        phase=RUN_DIR_PHASE,
    )
    if detached is not None:
        return detached

    return workflow.run()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
