"""A write phase's own check of the repository, and its tracked-files census (D3n).

:class:`VerifyStep` is the part of a code phase's ending that asks the
repository rather than the batches: whether a task of the phase applied an edit
still owed a check (:meth:`VerifyStep.phase_has_applied_batches`), running
``driver.verify_command`` once after the last batch (:meth:`VerifyStep.run`)
and saying so when nothing ran the tests (:meth:`VerifyStep.note_unverified_on_stop`),
and recording what the phase changed in ``tracked_files`` after each batch and
at the end (:meth:`VerifyStep.record_batch_tracking`,
:meth:`VerifyStep.finalize_tracking`). It holds no state of its own.

``WriteBatchScheduler`` composes it with the run's ``RunContext`` and hands it
the few things it calls back: the command runner, which stays on the scheduler
(``_run_verify_command``), the tests line worded from the batches' checks, the
stop's headline and the task lookup
(docs/development/write-scheduler.md#where-the-schedulers-helpers-live).
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import batch_prompt as _batch_prompt
from . import check_records as _check_records
from . import write_context as _write_context
from .batch_ledger import touched_files
from .cli_common import EXIT_NEEDS_HUMAN
from .provider_registry import get_verify_command, get_verify_settings
from .subagent_executor import WRITE_APPLIED, WRITE_PARTIAL
from .subagent_tasks import PHASE_BY_WRITE_KIND

#: How much of the verification command's output travels in the result payload,
#: the summary and `.status.json`. A tail rather than the whole capture: a failing
#: build prints megabytes and what names the failure is the end of it.
VERIFY_TAIL_CHARS = 2000


def _redacted_text(text: str) -> str:
    """``write_safety.redact_text``, imported on first use."""
    from .write_safety import redact_text
    return redact_text(text)


def load_finalize_tracking():
    """The top-level ``finalize_tracking`` module, or ``None`` if unimportable."""
    try:
        import finalize_tracking  # type: ignore
    except ImportError:  # pragma: no cover - direct-script import shim
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        try:
            import finalize_tracking  # type: ignore
        except ImportError:
            return None
    return finalize_tracking


class VerifyStep:
    """The repository check and the tracked-files census, over one run's result.

    Built once per run by the scheduler. *effects_lock* is the scheduler's
    ``_effects_lock``, which every in-memory ``state.json`` write of the run
    takes; *finalize* is the scheduler's ``finalize``. The four callables
    reach the scheduler's ``_task_by_id``, ``_unverified_text``,
    ``_set_stop_headline`` and ``_run_verify_command``, looked up at each call.
    """

    def __init__(
        self,
        ctx: _write_context.RunContext,
        *,
        effects_lock: threading.Lock,
        finalize: bool,
        task_by_id: Callable[[str], Optional[Dict[str, Any]]],
        unverified_text: Callable[..., str],
        set_stop_headline: Callable[..., Any],
        run_verify_command: Callable[[str, int], Any],
    ) -> None:
        self._ctx = ctx
        self._effects_lock = effects_lock
        self._finalize = finalize
        self._task_by_id = task_by_id
        self._unverified_text = unverified_text
        self._set_stop_headline = set_stop_headline
        self._run_verify_command = run_verify_command

    def note_unverified_on_stop(self, result: Any) -> List[str]:
        """The "nothing ran this repository's tests" line on a stop (fix
        round 16); the lines to print, ``[]`` when none is owed.

        Run 6 (Codex and OpenCode stop 01): a code phase that stopped after
        code batches ran said nothing about the repository's tests — the line
        :meth:`run` adds reached only a finished run's result,
        notice and summary. Set on ``result.unverified`` (the notice, the
        Verification line, the report, the mirror) when this is a code phase,
        a task of it applied an edit (this run or an earlier one,
        :meth:`phase_has_applied_batches` — the rule
        :meth:`run` keeps too) and no ``driver.verify_command``
        is set. Never runs anything. Fix round 17: it counted any completed
        task, so a stop after a check alone (which edits nothing, and may have
        run the tests itself) said that nothing had run them.

        Fix round 18 (live run 7, Codex and OpenCode stop 01): the line was
        in the result's ``notice``, and neither host relayed ``notice`` at a
        stop — both relay the stop's headline, the result line's ``error``
        (the one line the needs-human ``message`` quotes). So the line also
        ends the headline of a stop already recorded (the scheduler's ``_record_stop``
        and ``_record_check_block`` run first): the tool routes carry it where
        they carry the stop — the result line's ``error`` in the tool output
        — and ``run.error`` and ``status`` with it. Not the OpenCode stop
        title (fix round 19, review of round 18, #5): that cuts the error
        line at ``modes.STOP_TITLE["max_line"]`` (300) characters, and a
        real stop headline runs past that before this line (live run 7:
        644), so the title shows what stopped and the resume command. The
        line stays in the ``notice`` too, the field a finished run carries
        it in, so a tool result shows it in both.
        """
        if self._ctx.kind not in _batch_prompt.CODE_WRITE_KINDS or self._ctx.dry_run or result.unverified:
            return []
        if not self.phase_has_applied_batches(result):
            return []
        try:
            command = get_verify_command(anchor=self._ctx.anchor)
        except Exception:  # noqa: BLE001 - advisory on a stop
            return []
        if command:
            return []
        result.unverified = self._unverified_text(stop=True)
        headline = (result.stop or {}).get("headline")
        if headline and result.unverified not in headline:
            # The result line's `error` is bounded: the headline is fitted
            # again with this line as its whole end (fix round 19: the
            # model's quoted words give way first, never the remedy).
            self._set_stop_headline(result.stop, tail=result.unverified)
        return [result.unverified]

    def run(self, result: Any) -> bool:
        """Run ``driver.verify_command`` once, after the last batch (F4).

        Returns whether the phase may complete. Three ways that is ``True``
        without a command ever running, and only the first is announced:

        * **No command configured.** Logged, because "this repo declares no
          check" and "the check passed" are different facts and a silent skip
          spells them the same way. `instructions/apply-code-fixes.md` used to
          hard-code `pnpm typecheck && pnpm lint:fix`, which is right for one
          repo shape and wrong for the rest — the audited run's target had no
          `package.json` at all.
        * **A markdown phase.** `apply_suggestions`/`apply_task_suggestions`
          edit the plan and the tasks file; a typecheck has nothing to say about
          them, so they skip silently rather than nagging about a command that
          was never going to apply.
        * **Nothing applied.** No task of this run **and** none recorded by an
          earlier run of this phase applied an edit
          (:meth:`phase_has_applied_batches`) means no edit to check. Fix
          round 17: a dispatched batch used to be enough, so a run whose only
          batches were checks or no-ops ran the command (or said nothing had
          run the tests) about edits nobody made; the stop's line
          (:meth:`note_unverified_on_stop`) asks the same question.

        The command is a **shell string** (`&&` is in its documented shape), run
        with `cwd` at the pinned repo root — the same root every containment
        check resolves against — and bounded by `driver.verify_timeout_s`, which
        is a bound on the whole process group and not just on `/bin/sh`
        (the scheduler's ``_run_verify_command``). It is logged before it starts and again
        with its exit code once it ends.
        """
        if self._ctx.kind not in _batch_prompt.CODE_WRITE_KINDS:
            return True
        if not self.phase_has_applied_batches(result):
            return True
        # One load for the command, its timeout and the layer named in the log
        # (docs/development/providers.md#configuration-and-runner-resolution).
        command, timeout, source = get_verify_settings(anchor=self._ctx.anchor)
        if not command:
            self._ctx.log(
                "[write-subagents] no driver.verify_command configured — "
                f"{len(result.batches_dispatched)} batch(es) were verified "
                "against their own snapshots, but nothing checked the "
                "repository as a whole."
            )
            # Fix round 14: said on the result too, not only in the log —
            # codex racing run 5's apply-code-fixes left the repository's own
            # test failing and every surface a host relays read clean.
            # Fix round 20: and which checks the batches were asked to run,
            # with what each batch reported (apply-code-fixes); fix round 24:
            # led by what ran, implement's tasks' checks too.
            result.unverified = self._unverified_text()
            result.messages.append(result.unverified)
            return True
        # Announced BEFORE it runs, which is what `docs/configuration.md`'s
        # safety section already promises of the one config key whose value is
        # executed: the only line used to be written once the command returned,
        # so a 600 s check left a detached run's log silent about what it was
        # waiting on — and naming it is the whole mitigation for an untrusted
        # repo's override (F6/F3a, 2026-09-16). The closing line below still
        # reports the exit code and the duration.
        self._ctx.log(
            f"[write-subagents] verification ({source}): running "
            f"`{_redacted_text(command)}` (timeout {timeout}s)"
        )
        started = time.monotonic()
        exit_code, output = self._run_verify_command(command, timeout)
        duration = time.monotonic() - started
        # Redact BEFORE the tail cap, the same order the audit log uses: a build
        # that echoes a token into its log would otherwise put it in
        # `.status.json`, `write_batch_report.json` and `summary.md` at once.
        result.verification = {
            "command": _redacted_text(command),
            "source": source,
            "exit_code": exit_code,
            "duration_seconds": round(duration, 1),
            "output_tail": _redacted_text(output).strip()[-VERIFY_TAIL_CHARS:],
        }
        self._ctx.log(
            f"[write-subagents] verification ({source}): "
            f"`{result.verification['command']}` exited {exit_code} "
            f"in {duration:.1f}s"
        )
        if exit_code == 0:
            return True
        result.exit_code = EXIT_NEEDS_HUMAN
        result.messages.extend(self.verification_message(result))
        return False

    def phase_has_applied_batches(
        self, result: Optional[Any] = None,
    ) -> bool:
        """Whether a task of this phase applied an edit that is still owed a
        check: one of *result*'s outcomes, or one an earlier run left (F3).

        The check used to gate on ``result.batches_dispatched`` alone. The
        apply kinds re-materialize their full batch list on ``--resume`` and a
        resume skip counts as dispatched, so their check re-ran; ``implement``
        materializes only the tasks still owed, so after a run in which every
        task applied and only the check failed, the resumed list was EMPTY,
        nothing dispatched, the check never re-ran and ``_finalize`` stamped
        the phase completed against a repository still failing its own check
        — the exact ``--resume`` the exit-3 message had told the user to run
        (2026-09-16). ``state.json``'s task statuses are the record that
        survives that filter, so they answer here.

        Fix round 17: only an edit counts. This run's ``ok`` outcomes count
        when they ``applied`` (a resume skip carries the earlier verdict); a
        task ``state.json`` records ``completed`` counts when the phase's
        audit log's last completion of it applied or partly applied an edit
        (a failed batch the user accepted keeps what it moved). A check task
        (``no-op``, ``verification_only``), a ``no-op`` batch and a manual
        task answered ``done`` (no dispatch at all) edit nothing. A completed
        task with no completion in the audit log (the native leg records
        none) still counts unless it is a manual or a check task this run
        knows of: the conservative reading, which runs the check.
        """
        if result is not None and any(
            outcome.ok and outcome.write_outcome == WRITE_APPLIED
            for outcome in result.outcomes
        ):
            return True
        if self._ctx.state is None or not hasattr(self._ctx.state, "get_all_task_statuses"):
            return False
        try:
            statuses = self._ctx.state.get_all_task_statuses()
        except Exception:  # noqa: BLE001 - advisory, never fatal
            return False
        candidates = {str(task.get("task_id") or "") for task in self._ctx.tasks}
        if self._ctx.kind in _write_context._ID_KEYED_WRITE_KINDS:
            # The plan's own task ids, materialized this run or not — the flat
            # map also holds the apply kinds' batch ids, which are not this
            # phase's.
            candidates |= {
                task_id for task_id in statuses if _check_records._IMPLEMENT_TASK_ID.match(task_id)
            }
        candidates.discard("")
        completed = sorted(
            task_id for task_id in candidates if statuses.get(task_id) == "completed"
        )
        if not completed:
            return False
        record = getattr(self._ctx.state, "state", None)
        human = {
            str(task_id) for task_id in (
                (record.get("implement_human_task_ids") if isinstance(record, dict)
                 else None) or []
            )
        }
        for task_id in completed:
            task = self._task_by_id(task_id)
            expected = [str(i) for i in ((task or {}).get("expected_ids") or [])]
            try:
                done = self._ctx.audit.completion_for(
                    task_id, expected_ids=expected or None,
                )
            except Exception:  # noqa: BLE001 - an unreadable log is no record
                done = None
            if done is not None:
                if done.get("write_outcome") in (WRITE_APPLIED, WRITE_PARTIAL):
                    return True
                continue
            if task_id in human:
                continue
            if task is not None and (
                _batch_prompt.is_check_task(self._ctx.kind, task)
                or task.get("executor_role") == "human" or task.get("is_human")
            ):
                continue
            return True
        return False

    def verification_message(self, result: Any) -> List[str]:
        """The exit-3 block for a failed `driver.verify_command` (F4).

        Worded like the scheduler's ``_stop_message``: what happened, what was NOT undone,
        and the one command that moves it forward. Deliberately *not* a rollback
        — every batch was verified against its own snapshot and the half-applied
        state a revert would destroy is exactly what a human needs to read.
        """
        record = result.verification or {}
        lines = [
            "",
            f"ERROR: every batch applied and verified, but the repository's own "
            f"check failed, so the {self._ctx.phase} phase is NOT marked completed.",
            f"  - driver.verify_command ({record.get('source', 'unknown')}): "
            f"`{record.get('command', '')}` exited {record.get('exit_code')} "
            f"after {record.get('duration_seconds')}s",
        ]
        tail = str(record.get("output_tail") or "").strip()
        if tail:
            lines.append("Last output:")
            lines += [f"  {line}" for line in tail.splitlines()[-20:]]
        applied = (
            f"batches {', '.join(str(i) for i in result.batches_dispatched)} "
            f"are applied"
            if result.batches_dispatched
            else "every batch an earlier run of this phase applied is still applied"
        )
        lines += [
            f"Nothing was rolled back: {applied} and the pre-batch snapshots "
            f"are intact under {os.path.join(self._ctx.safety_dir, 'pre_batch')}/.",
            "Fix the repository (or the fix the batches applied), then re-invoke "
            "with --resume: every batch is already done, so nothing is "
            "re-dispatched and only the check runs again.",
        ]
        return lines

    def finalize_tracking(self, result: Optional[Any] = None) -> None:
        """``finalize_tracking.py --state-file`` in process (D3n).

        Every file a batch verifiably changed travels as ``--touched``.
        ``finalize_tracking`` otherwise subtracts ``pre_existing_changes`` from
        the working tree's current diff, which is right for a file the run
        never touched and wrong for one it did: a file already modified when the
        phase started (measured 2026-09-16, ``index.html`` under ``AM``) is
        subtracted along with the rest, ``tracked_files`` ends up naming only
        the noise, and ``review-code`` reviews the noise.
        """
        touched = self.touched_files(result)
        finalize_tracking = load_finalize_tracking()
        if finalize_tracking is None:
            return
        argv = ["--state-file", str(self._ctx.state.state_file)]
        for path in touched:
            argv += ["--touched", path]
        argv += ["--phase", PHASE_BY_WRITE_KIND[self._ctx.kind]]
        try:
            code = finalize_tracking.main(argv)
        except SystemExit:  # pragma: no cover - argparse-style exit
            return
        except Exception as exc:  # noqa: BLE001
            self._ctx.log(f"[write-subagents] warning: finalize_tracking failed: {exc}")
            return
        # F4 (2026-09-17): `main` *returns* 1 for an unreadable or unparseable
        # state file and for a failed write — it does not raise — and the return
        # value used to be dropped. `_reload_state` then silently kept whatever
        # it could read, the phase finalized, and `tracked_files` stayed whatever
        # an earlier phase had left for `review-code` to review. A census that
        # did not happen has to be visible in both places a reader looks.
        if code:
            message = (
                f"WARNING: finalize_tracking exited {code}; tracked_files was "
                f"NOT updated — `review-code` will read whatever the previous "
                f"phase left. Re-run: util finalize-tracking --state-file "
                f"{self._ctx.state.state_file}"
            )
            self._ctx.log(f"[write-subagents] {message}")
            if result is not None:
                result.messages.append(message)

    def touched_files(self, result: Optional[Any]) -> List[str]:
        """Every file an ``ok`` batch of this phase verifiably changed.

        This run's outcomes, and on a ``--resume`` the earlier runs' ``ok``
        rows too: a resumed implement materializes only the unfinished tasks,
        so without those rows a file an earlier run edited while it was
        already dirty (the baseline subtracts it) left ``tracked_files`` the
        moment the resumed run recorded its own.
        """
        if result is None:
            return []
        return touched_files(
            [*result.earlier_rows, *result.outcomes], include_partial=False)

    def record_batch_tracking(self, result: Any) -> None:
        """Record what this code phase has changed so far, after each batch.

        ``tracked_files`` used to be written only by ``_finalize``, which a
        run that stops part-way never reaches: the Codex racing run 3
        (2026-10-05) stopped implement at its ninth task with ``index.html``
        created and unrecorded, and review-code reviewed the plan and
        ``tasks.md`` an earlier phase had left there instead. The same census
        ``_finalize`` runs, in memory through this phase's own
        ``StateManager`` (the one writer), quietly, and saved only when the
        list moved.
        """
        if (self._ctx.dry_run or not self._finalize or self._ctx.state is None
                or self._ctx.kind not in _batch_prompt.CODE_WRITE_KINDS):
            return
        recorded = getattr(self._ctx.state, "state", None)
        state_file = getattr(self._ctx.state, "state_file", None)
        if not isinstance(recorded, dict) or not state_file:
            return
        finalize_tracking = load_finalize_tracking()
        if finalize_tracking is None:
            return
        try:
            entries = finalize_tracking.phase_tracked_files(
                recorded, self.touched_files(result),
                state_path=Path(str(state_file)),
                phase=PHASE_BY_WRITE_KIND[self._ctx.kind],
                say=lambda line: None,
            )
        except Exception as exc:  # noqa: BLE001 - bookkeeping must not fail a phase
            self._ctx.log(
                f"[write-subagents] warning: could not record the files this "
                f"batch changed in tracked_files: {exc}"
            )
            return
        if entries == recorded.get("tracked_files"):
            return
        with self._effects_lock:
            self._ctx.state.set("tracked_files", entries)
            self._ctx.state.save()
        named = ", ".join(
            f"{entry['path']} ({entry['action']})" for entry in entries[:5]
        )
        more = f", and {len(entries) - 5} more" if len(entries) > 5 else ""
        self._ctx.log(
            f"[write-subagents] tracked_files now names {len(entries)} "
            f"file(s): {named}{more}"
        )
