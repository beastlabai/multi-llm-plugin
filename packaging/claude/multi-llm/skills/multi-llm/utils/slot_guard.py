"""One fan-out slot's sandbox gate and write audit: :class:`SlotGuard`.

Every lane that dispatches a fan-out slot brackets the dispatch with the same
sequence: ask's single-shot slot (``AskRun.ask_one``), review-code's
``run_single_review``, and ``run_single_model`` for review-plan and
review-tasks, on both of its lanes (the file-output ladder and the single
stdout dispatch). This module is that sequence, written once.

* :meth:`SlotGuard.begin`, before anything is dispatched. It runs the
  pre-dispatch sandbox gate (``sandbox_refusal``) and prints the one line a
  refusal or a fallback rung gets. Then it takes the before-snapshots: the
  porcelain baseline on the uncontained rung, the file modes on the plugin's
  own jail. A refused slot comes back :class:`Refused` and must not be
  dispatched.
* :meth:`SlotGuard.finish`, once, after the slot's last attempt. It runs the
  porcelain delta and its verdict (a not-attributed warning when the slot's
  own audited stream clears it, otherwise the fail-closed violation naming the
  boundary-less siblings), the jail's own lines when the lane asks for them,
  and the mode and home checks. It returns the one reason the slot fails
  with, or ``None``.

What a lane does with that reason stays the lane's: ask sets the answer file
aside as ``.violated`` and keeps its stdout text as a partial, review-code
records the tracker's ``failed`` and returns its tuple, and the review base
drops ``output`` so ``save_model_result`` never parses it.

The checks themselves are ``review_orchestrator_base``'s and are looked up on
that module at call time. They stay there because the sibling counts there
ask the same gate, and the tests that pin a probe verdict or a snapshot patch
them on that module.

The rules are in docs/development/providers.md (the fallback ladder, the
porcelain check and its stream-audit exemption, the plugin's jail and its mode
and home checks) and docs/development/write-safety.md.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Sequence, Union

from . import readonly_audit as _readonly_audit
from . import review_orchestrator_base as _base

#: Every attempt's ``details["home_check"]``, or, for a lane whose single call
#: raised, a coroutine function that compares the homes that call armed
#: (``raised_attempt_home_records``). The second form runs only when the
#: write check passed, which is where the stdout lane has always run it.
HomeChecks = Union[Sequence[Any], Callable[[], Awaitable[Sequence[Any]]]]


@dataclass(frozen=True)
class Refused:
    """The gate refused the slot: dispatch nothing. ``reason`` is the
    ``[SANDBOX_UNAVAILABLE]`` line, already printed under the slot's name."""

    reason: str


@dataclass(frozen=True)
class Ready:
    """The slot may be dispatched with ``sandbox_fallback_kwargs(sandbox)``."""

    sandbox: _base.SandboxVerdict


class SlotGuard:
    """The gate and the write audit around one fan-out slot's dispatch.

    ``display_name`` prefixes every line the guard prints (ask's slots print
    under the model spec, the review lanes under ``provider:model``).
    ``tracker`` is the fan-out's ``FanoutProgress`` or ``None``; the guard
    records the audit verdicts and the not-attributed warnings on it and
    leaves every slot transition to the lane.

    ``recompute_ignore_prefixes`` is the lane's own rule for the porcelain
    check's ignore list after the dispatch: ``True`` resolves it again for the
    after snapshot (ask and review-code), ``False`` reuses the one ``begin()``
    took (the review base's two lanes). The list resolves the output folder
    through symlinks, so the two can differ when the window re-points it
    (docs/development/providers.md, the porcelain check).
    """

    def __init__(
        self,
        *,
        provider_name: str,
        display_name: str,
        model_spec: str,
        project_root: Optional[str],
        plan_path: Union[str, Path],
        recompute_ignore_prefixes: bool,
        tracker: Optional[Any] = None,
    ) -> None:
        self.provider_name = provider_name
        self.display_name = display_name
        self.model_spec = model_spec
        self.project_root = project_root
        self.plan_path = plan_path
        self.recompute_ignore_prefixes = recompute_ignore_prefixes
        self.tracker = tracker
        self.sandbox: Optional[_base.SandboxVerdict] = None
        self._ignore_prefixes: Sequence[str] = ()
        self._tree_before: Optional[_base.PorcelainSnapshot] = None
        self._mode_before: Optional[Dict[str, str]] = None

    async def begin(self) -> Union[Refused, Ready]:
        """The gate, then the before-snapshots; call it before the dispatch.

        The gate runs in a thread: it may spawn the sandbox probe, and the
        sibling slots share this event loop. So do the snapshots, since
        ``git status`` is not instant on a large repository. The porcelain
        baseline is taken for the uncontained rung alone (the rung id decides,
        ``porcelain_check_applies``), so a sandboxed or contained-rung slot
        makes no git call. Its ignore list is the roster-aware one, computed
        here in the slot's task where the fan-out roster lives; the after
        snapshot reuses it or resolves it again, as the lane chose
        (``recompute_ignore_prefixes``). The mode baseline is taken for the
        plugin's own jail alone (``mode_check_applies``).
        """
        sandbox = await asyncio.to_thread(
            _base.sandbox_refusal, self.provider_name, self.project_root
        )
        self.sandbox = sandbox
        if sandbox.refused:
            print(f"[{self.display_name}] {sandbox.reason}")
            return Refused(sandbox.reason)
        # Any fallback rung prints its one line: `[SANDBOX_FALLBACK]` for a
        # contained rung, `[SANDBOX_INCIDENTAL]` for the plugin's jail,
        # `[SANDBOX_BYPASSED]` for the uncontained one.
        if sandbox.fallback:
            print(f"[{self.display_name}] {sandbox.reason}")
        if _base.porcelain_check_applies(sandbox):
            self._ignore_prefixes = _base.fanout_ignore_prefixes(
                self.plan_path, self.project_root, anchor=self.plan_path
            )
            self._tree_before = await asyncio.to_thread(
                _base.porcelain_snapshot, self.project_root,
                ignore_prefixes=self._ignore_prefixes,
            )
        if _base.mode_check_applies(sandbox):
            self._mode_before = await asyncio.to_thread(
                _base.mode_snapshot, self.project_root
            )
        return Ready(sandbox)

    async def finish(
        self,
        audits: Sequence[Any],
        home_checks: HomeChecks,
        *,
        summary_key: Any,
        jail_details: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """The one reason this slot fails with, or ``None``; once, after the
        slot's last attempt and before the lane reads anything it returned.

        ``audits`` is every attempt's ``details["readonly_audit"]`` in the
        slot's window, ``None`` for an attempt that raised (plan §4.3: ANDed,
        so a clean retry never excuses an earlier attempt). ``summary_key``
        keys the run's closing summary (the phase or question directory).

        The write check runs first and its violation ends the sequence. Then
        ``jail_details``, when given, gets its ``[JAIL_DENIED]`` line and
        in-stream denial count (``report_jail_outcome``). Ask's single-shot
        slot passes its call's ``details``; the ladder lanes leave it out,
        because the ladder loop reports the jail per attempt. Then the mode
        and home checks for a slot on the plugin's jail.
        """
        violation = await self._write_check(audits)
        if violation is not None:
            return violation
        if jail_details is not None:
            _base.report_jail_outcome(self.display_name, jail_details, summary_key)
        if callable(home_checks):
            home_checks = await home_checks()
        return await _base.run_incidental_checks(
            self._verdict(), provider_name=self.provider_name,
            display_name=self.display_name, model_spec=self.model_spec,
            project_root=self.project_root, plan_path=self.plan_path,
            mode_before=self._mode_before, home_checks=home_checks, audits=audits,
            tracker=self.tracker, summary_key=summary_key,
        )

    async def _write_check(self, audits: Sequence[Any]) -> Optional[str]:
        """The reason to fail an uncontained slot whose window changed the tree, or ``None``.

        A delta fails the slot whatever the attempt reported. A review that
        also edited the repository is not a review the phase may count. Nothing
        is restored: no snapshot of the files was taken, so the operator is
        told which paths and left to decide. The reason line
        (``bypass_write_violation``) states the change without naming an
        author, and counts the fan-out's other boundary-less slots. ``None``
        for every slot not on the uncontained rung: the rung id gates it, so a
        contained-rung or sandboxed slot makes no git call at all. The
        ``git status`` and the sibling count run in threads, like the gate.

        When the AND of ``audits`` is clean and option B is switched on
        (``bypass_stream_audit_clears``), the delta is reported as a
        ``not attributed`` warning on ``progress.models[spec].warning`` and
        the slot keeps its output.
        """
        if not _base.porcelain_check_applies(self._verdict()):
            return None
        # Resolved again here, in the slot's task like begin()'s, when the
        # lane asks for it (``recompute_ignore_prefixes``).
        ignore_prefixes = (
            _base.fanout_ignore_prefixes(
                self.plan_path, self.project_root, anchor=self.plan_path
            )
            if self.recompute_ignore_prefixes
            else self._ignore_prefixes
        )
        if self.tracker is not None:
            self.tracker.record_readonly_audit(
                self.model_spec, _readonly_audit.combine_audits(audits).to_dict()
            )
        tree_after = await asyncio.to_thread(
            _base.porcelain_snapshot, self.project_root,
            ignore_prefixes=ignore_prefixes,
        )
        touched = _base.unexpected_writes(
            self._tree_before, tree_after, ignore_prefixes=ignore_prefixes,
        )
        if not touched:
            return None
        if _base.bypass_stream_audit_clears(audits):
            warning = _base.bypass_write_not_attributed(self.provider_name, touched)
            print(f"[{self.display_name}] {warning}")
            if self.tracker is not None:
                self.tracker.warn(self.model_spec, warning)
            return None
        siblings = await asyncio.to_thread(
            _base.boundaryless_sibling_count,
            self.model_spec, self.project_root, self.plan_path,
        )
        violation = _base.bypass_write_violation(
            self.provider_name, touched, boundaryless_siblings=siblings
        )
        print(f"[{self.display_name}] {violation}")
        return violation

    def _verdict(self) -> _base.SandboxVerdict:
        if self.sandbox is None:
            raise RuntimeError("SlotGuard: the checks ran before begin() took the gate")
        return self.sandbox
