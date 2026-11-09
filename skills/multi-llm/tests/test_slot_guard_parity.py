"""The fan-out lanes' sandbox gate and write audit, pinned lane by lane.

Four lanes bracket a slot's dispatch with the same sequence: ask's
``AskRun.ask_one``, review-code's ``run_single_review``, and both lanes of
``run_single_model`` (the file-output ladder and the single stdout dispatch).
The sequence is the pre-dispatch sandbox gate (refuse, or print the rung's
one line), the porcelain and mode before-snapshots, and after the dispatch the
porcelain delta and its verdict, the jail's own lines, and the mode and home
checks. ``utils/slot_guard.py`` runs it for all four; each lane keeps only
what it does with the one reason it gets back.

This file drives every lane through every outcome of that sequence and pins
what is observable: the failure reason (literal text other tests also assert),
the slot's marker lines in order, the order of the gate, the snapshots and the
dispatch (and that each ran off the event loop's thread), the tracker calls,
the closing-summary lines, and each lane's own disposition. It was written
against the three hand-written copies and passed there before they were
merged, so a difference between the lanes is either listed here as one lane's
own or is a regression.

Tier 1, zero spend. The probe is pinned at ``_fanout_sandbox_verdict`` and the
real gate decides from it; the dispatch is a stub at each lane's invoker name;
the tree, its snapshots and the mode check are a real git repo under
``tmp_path``. See docs/development/providers.md (the fallback ladder, the
porcelain check, the plugin's jail) and docs/development/write-safety.md.
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import ask_orchestrator  # noqa: E402
import code_review_orchestrator as cro  # noqa: E402
from ask_orchestrator import ASK_PROMPT_FILE, AskRun  # noqa: E402
from utils import landlock_dispatch  # noqa: E402
from utils import review_orchestrator_base as base  # noqa: E402
from utils import slot_guard  # noqa: E402
from utils.json_extractor import sanitize_model_name  # noqa: E402
from utils.prompt_loader import load_prompt  # noqa: E402
from utils.providers.base import (  # noqa: E402
    CODEX_HOME_CHECK_MARKER,
    JAIL_DENIED_MARKER,
    MODE_CHECK_MARKER,
    SANDBOX_BYPASSED_MARKER,
    SANDBOX_FALLBACK_MARKER,
    SANDBOX_INCIDENTAL_MARKER,
)
from utils.subagent_executor import SandboxProbeResult  # noqa: E402
from tests.support.gitrepo import git  # noqa: E402

#: Hermetic git (tests/support/gitrepo.py), returning the CompletedProcess.
_git = partial(git, text_out=False)

# POSIX file modes and a real git tree; Windows runs no plugin jail.
pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="POSIX file modes (the mode check); Windows runs no plugin jail",
)

SPEC = "codex:gpt-5.5"
STEM = sanitize_model_name(SPEC)
#: Every slot runs in a fan-out of three: a boundary-less sibling by design
#: (cursor-agent's probe answers ``not-required``) and a second codex slot,
#: which takes whatever verdict this slot takes. Both count on the uncontained
#: rung and on the jail, so the sibling counts read 2.
ROSTER = (SPEC, "cursor-agent:auto", "codex:gpt-5.4")
PREFIX = f"[{SPEC}] "

BWRAP_LINE = "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted"
PROBES = {
    "refuse": SandboxProbeResult("unavailable", BWRAP_LINE),
    "plain": SandboxProbeResult("ok", ""),
    "contained": SandboxProbeResult(
        "unavailable", BWRAP_LINE, "landlock", (("landlock", "taken"),), (), "0.155.1",
    ),
    "bypass": SandboxProbeResult(
        "unavailable", BWRAP_LINE, "unsandboxed",
        (("landlock", "skipped"), ("plugin-landlock", "absent"), ("unsandboxed", "taken")),
        (), "0.163.0",
    ),
    "jail": SandboxProbeResult(
        "unavailable", BWRAP_LINE, "plugin-landlock",
        (("landlock", "skipped"), ("plugin-landlock", "taken")), (), "0.163.0",
    ),
}

CLEAN = {"clean": True, "reason": "", "actions": 1}
NOT_CLEAN = {"clean": False, "reason": "command_execution: heredoc", "actions": 1}
HOME_CHANGE = {
    "home": "~/codex-home", "changed": ["config.toml"], "config_keys": ["mcp_servers.x"],
    "foreign_trust": [], "other_live_children": 1,
}
JAIL_NOTICE = f"{JAIL_DENIED_MARKER} codex: simulated denial of /elsewhere/x"
#: ``details["jail_denials"]`` as ``llm_client`` records it: the denials'
#: ``to_dict()`` plus their total and the paths they name.
_DENIED = landlock_dispatch.JailDenials(
    out_of_scope_count=2, out_of_scope_paths=["/elsewhere/a", "/elsewhere/b"],
)
JAIL_DENIALS = {**_DENIED.to_dict(), "total": _DENIED.total, "paths": _DENIED.named_paths()}
ANSWER = "An answer."
FINDINGS = [{
    "title": "Add input validation",
    "desc": "The plan does not validate the new email field.",
    "importance": "HIGH",
    "reference": "Overview",
    "type": "addition",
}]

# The reasons, literally: the lanes print them, record them and return them.
BYPASS_VIOLATION = (
    f"{SANDBOX_BYPASSED_MARKER} codex: the working tree changed during this "
    f"unsandboxed slot's window (src/app.py) while 2 other boundary-less slots "
    f"in this fan-out, or any other process writing to this repository, could "
    f"also have written it; the check cannot tell who made the change, so this "
    f"slot fails closed and its output is discarded"
)
BYPASS_WARNING = (
    f"{SANDBOX_BYPASSED_MARKER} codex: the working tree changed during this "
    f"unsandboxed slot's window (src/app.py), changed during the window by "
    f"something outside this slot — not attributed: this slot's own action "
    f"stream shows only read-only actions, so its output is kept"
)
MODE_VIOLATION = (
    f"{MODE_CHECK_MARKER} codex: a tracked file's executable bit changed during "
    f"this slot on the plugin's Landlock jail, which does not stop chmod "
    f"(tests/run.sh 100644 → 100755), while 2 other slots in this fan-out that "
    f"could also change a file mode, or any other process writing to this "
    f"repository, could also have made the change; the mode check cannot tell "
    f"who made it, so this slot fails closed and its output is discarded (the "
    f"mode is left as found)"
)
MODE_WARNING = (
    f"{MODE_CHECK_MARKER} codex: a tracked file's executable bit changed during "
    f"this slot's window (tests/run.sh 100644 → 100755) — not attributed: this "
    f"slot's own action stream shows only read-only actions, so its output is "
    f"kept (the mode is left as found)"
)
HOME_LINE = (
    f"{CODEX_HOME_CHECK_MARKER} codex: ~/codex-home changed during this slot on "
    f"the plugin's Landlock jail, which does not stop writes there: config.toml "
    f"(mcp_servers.x); 1 other codex child of this run was live in the window; "
    f"the change persists into later codex sessions and was not reverted — "
    f"review those files; this slot fails closed and its output is discarded"
)
DENIALS_LINE = (
    "the plugin's Landlock jail denied 2 writes inside the model's own commands "
    "(not a failure): /elsewhere/a, /elsewhere/b"
)

LANES = ("ask", "review-code", "review-plan", "review-plan-stdout")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """One outcome of the sequence, and what every lane does with it."""

    probe: str
    #: What the stub does in the slot's window before it answers or raises.
    write: bool = False
    chmod: bool = False
    raises: bool = False
    audit: Optional[Dict[str, Any]] = None
    home_check: Optional[Dict[str, Any]] = None
    jail: bool = False
    #: The one failure reason, or ``None`` when the slot keeps its answer;
    #: ``"gate"`` means the gate's own reason (the refusal line).
    reason: Optional[str] = None
    #: The slot's marker lines in order; ``"gate"`` is the gate's one line.
    lines: Tuple[str, ...] = ()
    #: Event names in order (see :class:`Recorder`).
    events: Tuple[str, ...] = ()
    #: ``(method, text)`` of every tracker ``warn``.
    warnings: Tuple[str, ...] = ()
    #: The ``clean`` field of every ``record_readonly_audit`` call.
    audits_recorded: Tuple[bool, ...] = ()
    #: The closing summary's lines for this run.
    summary: Tuple[str, ...] = ()
    #: Lane name -> overrides of ``lines`` and ``summary`` (the one lane
    #: whose order differs).
    lane_lines: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    lane_summary: Dict[str, Tuple[str, ...]] = field(default_factory=dict)


SCENARIOS: Dict[str, Scenario] = {
    "refused": Scenario(
        probe="refuse", reason="gate", lines=("gate",), events=("gate",),
    ),
    "plain-dispatch": Scenario(
        probe="plain", write=True, audit=NOT_CLEAN,
        events=("gate", "dispatch"),
    ),
    # A contained rung is fenced by the kernel: no snapshot, no git call, so a
    # write the stub makes is not looked for.
    "contained-rung": Scenario(
        probe="contained", write=True, audit=NOT_CLEAN, lines=("gate",),
        events=("gate", "dispatch"),
    ),
    "bypass-no-write": Scenario(
        probe="bypass", audit=NOT_CLEAN, lines=("gate",),
        events=("gate", "porcelain", "dispatch", "porcelain"),
        audits_recorded=(False,),
    ),
    "bypass-write-cleared": Scenario(
        probe="bypass", write=True, audit=CLEAN, lines=("gate", BYPASS_WARNING),
        events=("gate", "porcelain", "dispatch", "porcelain"),
        warnings=(BYPASS_WARNING,), audits_recorded=(True,),
    ),
    "bypass-write-violation": Scenario(
        probe="bypass", write=True, audit=NOT_CLEAN, reason=BYPASS_VIOLATION,
        lines=("gate", BYPASS_VIOLATION),
        events=("gate", "porcelain", "dispatch", "porcelain", "siblings"),
        audits_recorded=(False,),
    ),
    # A call that raised read nothing back: its audit is `None`, which never
    # clears a change.
    "bypass-write-raised": Scenario(
        probe="bypass", write=True, raises=True, reason=BYPASS_VIOLATION,
        lines=("gate", BYPASS_VIOLATION),
        events=("gate", "porcelain", "dispatch", "porcelain", "siblings"),
        audits_recorded=(False,),
    ),
    # Jail details beside a porcelain violation: the lanes report the jail at
    # different points. Ask reports it after the porcelain verdict, so not at
    # all once that failed; the stdout lane reports it first; the ladder lanes
    # report it per attempt, inside the ladder loop, before any verdict.
    "bypass-violation-with-jail-lines": Scenario(
        probe="bypass", write=True, audit=NOT_CLEAN, jail=True, reason=BYPASS_VIOLATION,
        lines=("gate", JAIL_NOTICE, DENIALS_LINE, BYPASS_VIOLATION),
        events=("gate", "porcelain", "dispatch", "porcelain", "siblings"),
        audits_recorded=(False,), summary=(JAIL_NOTICE,),
        lane_lines={"ask": ("gate", BYPASS_VIOLATION)}, lane_summary={"ask": ()},
    ),
    "jail-clean": Scenario(
        probe="jail", audit=NOT_CLEAN, lines=("gate",),
        events=("gate", "mode", "dispatch", "mode"), audits_recorded=(False,),
    ),
    "jail-lines": Scenario(
        probe="jail", audit=NOT_CLEAN, jail=True,
        lines=("gate", JAIL_NOTICE, DENIALS_LINE),
        events=("gate", "mode", "dispatch", "mode"), audits_recorded=(False,),
        summary=(JAIL_NOTICE,),
    ),
    "jail-mode-violation": Scenario(
        probe="jail", chmod=True, audit=NOT_CLEAN, reason=MODE_VIOLATION,
        lines=("gate", MODE_VIOLATION),
        events=("gate", "mode", "dispatch", "mode", "mode-siblings"),
        audits_recorded=(False,),
    ),
    "jail-mode-cleared": Scenario(
        probe="jail", chmod=True, audit=CLEAN, lines=("gate", MODE_WARNING),
        events=("gate", "mode", "dispatch", "mode"),
        warnings=(MODE_WARNING,), audits_recorded=(True,),
    ),
    "jail-home-violation": Scenario(
        probe="jail", audit=NOT_CLEAN, home_check=HOME_CHANGE, reason=HOME_LINE,
        lines=("gate", HOME_LINE),
        events=("gate", "mode", "dispatch", "mode"), audits_recorded=(False,),
        summary=(HOME_LINE,),
    ),
    "jail-mode-and-home": Scenario(
        probe="jail", chmod=True, audit=NOT_CLEAN, home_check=HOME_CHANGE,
        reason=f"{MODE_VIOLATION}; {HOME_LINE}",
        lines=("gate", MODE_VIOLATION, HOME_LINE),
        events=("gate", "mode", "dispatch", "mode", "mode-siblings"),
        audits_recorded=(False,), summary=(HOME_LINE,),
    ),
    "jail-mode-raised": Scenario(
        probe="jail", chmod=True, raises=True, reason=MODE_VIOLATION,
        lines=("gate", MODE_VIOLATION),
        events=("gate", "mode", "dispatch", "mode", "mode-siblings"),
        audits_recorded=(False,),
    ),
}


# ---------------------------------------------------------------------------
# The rig: a scratch repo, the pinned probe, the spies and the stub
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A committed repo: a source file, a 0644 script, and the plan."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "plans").mkdir()
    (root / "src" / "app.py").write_text("print('app')\n", encoding="utf-8")
    run_sh = root / "tests" / "run.sh"
    run_sh.write_text("echo run\n", encoding="utf-8")
    run_sh.chmod(0o644)
    (root / "plans" / "feature.md").write_text(
        "# Plan\n\nAdd an email field with validation.\n", encoding="utf-8",
    )
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "init")
    home = tmp_path / "codex-home"
    home.mkdir()
    slot_tmp = tmp_path / "slot-tmp"
    slot_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(slot_tmp))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(home))
    monkeypatch.setenv("MULTI_LLM_TEST_FAST_BACKOFF", "1")
    return root


class Recorder:
    """The fan-out tracker's stand-in: every method call, recorded."""

    def __init__(self):
        self.calls: List[Tuple[str, tuple, dict]] = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return record

    def named(self, method):
        return [(args, kwargs) for name, args, kwargs in self.calls if name == method]


def _off_loop() -> bool:
    return threading.current_thread() is not threading.main_thread()


#: The names the sequence calls, spied wherever a lane module holds the same
#: function: the lanes imported them by name before ``SlotGuard``, so a spy on
#: the base module alone would miss two of the four lanes there.
_SPIED = {
    "sandbox_refusal": "gate",
    "porcelain_snapshot": "porcelain",
    "mode_snapshot": "mode",
    "boundaryless_sibling_count": "siblings",
    "mode_check_sibling_count": "mode-siblings",
}


def _lane_modules():
    modules = [base, cro, ask_orchestrator]
    try:
        from utils import slot_guard
    except ImportError:
        pass
    else:
        modules.append(slot_guard)
    return modules


@pytest.fixture
def rig(repo, monkeypatch):
    """Pin the probe per scenario, spy on the sequence, stub the dispatch."""
    events: List[Tuple[str, Any, bool]] = []

    for name, event in _SPIED.items():
        real = getattr(base, name)

        def spy(*args, _real=real, _event=event, **kwargs):
            # The gate's own calls are keyed by provider, so the sibling
            # counts' gate calls for the other two slots can be told apart.
            detail = args[0] if _event == "gate" else None
            events.append((_event, detail, _off_loop()))
            return _real(*args, **kwargs)

        for module in _lane_modules():
            if getattr(module, name, None) is real:
                monkeypatch.setattr(module, name, spy)

    state: Dict[str, Any] = {"events": events, "scenario": None, "answer_path": None}

    def probe(provider_name, _cwd):
        if provider_name == "codex":
            return PROBES[state["scenario"].probe]
        return SandboxProbeResult("not-required", "")

    monkeypatch.setattr(base, "_fanout_sandbox_verdict", probe)

    def details_for(scenario, rung):
        details: Dict[str, Any] = {"stderr": "", "readonly_audit": scenario.audit}
        if rung:
            details["sandbox"] = rung
        if scenario.home_check is not None:
            details["home_check"] = dict(scenario.home_check)
        if scenario.jail:
            details["jail_failure"] = {"notice": JAIL_NOTICE}
            details["jail_denials"] = dict(JAIL_DENIALS)
        return details

    def stub(lane):
        def invoke(**kwargs):
            scenario = state["scenario"]
            events.append(("dispatch", kwargs.get("sandbox_fallback"), _off_loop()))
            if scenario.write:
                (repo / "src" / "app.py").write_text("print('changed')\n", encoding="utf-8")
            if scenario.chmod:
                os.chmod(repo / "tests" / "run.sh", 0o755)
            if lane == "ask":
                # The answer file the model was asked for, and the same text
                # on stdout: a failed slot sets the one aside and keeps the
                # other as its partial.
                Path(state["answer_path"]).write_text(ANSWER + "\n", encoding="utf-8")
            if scenario.raises:
                raise ValueError("parser exploded")
            details = details_for(scenario, kwargs.get("sandbox_fallback"))
            if lane == "ask":
                details["answer_text"] = ANSWER
                return {"success": True, "error": None, "details": details}
            if lane == "review-plan-stdout":
                return {"success": True, "data": FINDINGS, "details": details}
            return {
                "success": True, "data": FINDINGS, "source": "file",
                "output_file": "", "details": details,
            }

        return invoke

    monkeypatch.setattr(ask_orchestrator, "invoke_with_provider", stub("ask"))
    monkeypatch.setattr(cro, "invoke_with_file_output", stub("review-code"))
    monkeypatch.setattr(base, "invoke_with_file_output", stub("review-plan"))
    monkeypatch.setattr(base, "invoke_with_provider", stub("review-plan-stdout"))
    state["repo"] = repo
    return state


# ---------------------------------------------------------------------------
# Driving one lane
# ---------------------------------------------------------------------------


@dataclass
class Outcome:
    success: bool
    reason: Optional[str]
    #: ask's and the base lanes' `source`; review-code's tracker `failed` source.
    source: Optional[str]
    result: Any
    tracker: Recorder
    summary_key: Any


def _in_fanout(coro_factory: Callable[[], Any]):
    async def main():
        return await base.spawn_fanout_slot(ROSTER, coro_factory())

    return asyncio.run(main())


def _drive(lane: str, rig) -> Outcome:
    repo = rig["repo"]
    plan = rig.get("plan") or repo / "plans" / "feature.md"
    tracker = Recorder()
    if lane == "ask":
        question_dir = Path(base.get_phase_dir(plan, "ask")) / "q-12345678"
        question_dir.mkdir(parents=True, exist_ok=True)
        rig["answer_path"] = str(question_dir / f"answer_{STEM}.md")
        run = AskRun(
            args=argparse.Namespace(timeout=60, max_parallel=1, rerun_all=False),
            question="Is it sound?", plan_path=str(plan),
            question_dir=str(question_dir), project_root=str(repo),
            template=load_prompt(ASK_PROMPT_FILE), inline_plan=plan.read_text(),
            model_specs=list(ROSTER), unattended=True, skip_existing=True,
            tracker=tracker,
        )
        result = _in_fanout(lambda: run.ask_one(SPEC, 0))
        rig["ask_run"] = run
        return Outcome(
            success=result["success"], reason=result["error"],
            source=result["source"], result=result, tracker=tracker,
            summary_key=str(question_dir),
        )
    if lane == "review-code":
        out_dir = plan.parent / plan.stem
        result = _in_fanout(lambda: cro.run_single_review(
            SPEC, plan, out_dir, timeout=60, tracker=tracker,
        ))
        success, _issues, error = result
        failed = tracker.named("failed")
        return Outcome(
            success=success, reason=error,
            source=failed[-1][1].get("source") if failed else None,
            result=result, tracker=tracker,
            summary_key=cro.get_phase_dir(plan, "code-review"),
        )
    out_dir = (
        str(base.get_phase_dir(plan, "review-plan")) if lane == "review-plan" else None
    )
    result = _in_fanout(lambda: base.run_single_model(
        model_spec=SPEC, plan_path=str(plan),
        prompt_template="{output_json_path}{output_location}", prompt_context={},
        phase_name="review-plan", invoke_phase="plan_review", timeout=60,
        out_dir=out_dir, prefix="feature" if out_dir else None, tracker=tracker,
    ))
    return Outcome(
        success=result["success"], reason=result["error"], source=result["source"],
        result=result, tracker=tracker,
        summary_key=base.phase_summary_key(str(plan), "review-plan"),
    )


def _marker_lines(out: str) -> List[str]:
    """The slot's own marker lines, prefix stripped, in the order printed."""
    lines = []
    for line in out.splitlines():
        if not line.startswith(PREFIX):
            continue
        text = line[len(PREFIX):]
        if (text.startswith("[") and text[1:2].isupper()) or text.startswith(
            "the plugin's Landlock jail denied"
        ):
            lines.append(text)
    return lines


def _expected_source(lane: str, scenario: Scenario) -> Optional[str]:
    """Each lane's own record of where a slot's result came from."""
    if scenario.reason == "gate":
        return "not-dispatched"
    if lane == "ask":
        if scenario.reason is None:
            return "file"
        return "exception" if scenario.raises else "failed"
    if lane == "review-code":
        if scenario.reason is None:
            return None
        return "exception" if scenario.raises else "file"
    if lane == "review-plan-stdout":
        return "stdout"
    return "exception" if scenario.raises else "file"


# ---------------------------------------------------------------------------
# The pins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lane", LANES)
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_each_lane_runs_the_same_gate_and_audit(name, lane, rig, capsys):
    scenario = SCENARIOS[name]
    rig["scenario"] = scenario
    gate = base.sandbox_refusal("codex", str(rig["repo"]))
    rig["events"].clear()
    capsys.readouterr()

    outcome = _drive(lane, rig)
    out = capsys.readouterr().out

    # The reason, literally, and where the lane says the result came from.
    expected_reason = gate.reason if scenario.reason == "gate" else scenario.reason
    assert outcome.reason == expected_reason, out
    assert outcome.success is (scenario.reason is None)
    assert outcome.source == _expected_source(lane, scenario)

    # The slot's marker lines, in order: the gate's one line first.
    expected_lines = [
        gate.reason if line == "gate" else line
        for line in scenario.lane_lines.get(lane, scenario.lines)
    ]
    assert _marker_lines(out) == expected_lines, out
    if scenario.probe == "contained":
        assert gate.reason.startswith(SANDBOX_FALLBACK_MARKER)
    elif scenario.probe == "bypass":
        assert gate.reason.startswith(SANDBOX_BYPASSED_MARKER)
    elif scenario.probe == "jail":
        assert gate.reason.startswith(SANDBOX_INCIDENTAL_MARKER)

    # The order: gate, before-snapshots, dispatch, after-snapshot, sibling
    # count. Each sibling count asks the gate once more, for the other codex
    # slot (cursor-agent's `not-required` probe answers without it). Every one
    # of these calls ran in a worker thread, never on the event loop.
    events = rig["events"]
    gates = [index for index, (event, _d, _o) in enumerate(events) if event == "gate"]
    own_events = [
        event for index, (event, _d, _o) in enumerate(events) if index not in gates[1:]
    ]
    assert own_events == list(scenario.events), events
    counts = sum(event in ("siblings", "mode-siblings") for event in scenario.events)
    assert [events[i][1] for i in gates] == ["codex"] * (1 + counts), events
    assert all(off for _event, _detail, off in events), events
    expected_rung = {
        "contained": "landlock", "bypass": "unsandboxed", "jail": "plugin-landlock",
    }.get(scenario.probe)
    dispatches = [detail for event, detail, _off in events if event == "dispatch"]
    assert dispatches == ([] if scenario.reason == "gate" else [expected_rung])

    # The tracker: the warnings and the recorded audit verdicts.
    warnings = [args[1] for args, _kw in outcome.tracker.named("warn")]
    assert warnings == list(scenario.warnings)
    audits = [args[1]["clean"] for args, _kw in outcome.tracker.named("record_readonly_audit")]
    assert audits == list(scenario.audits_recorded)

    # The closing summary keeps the jail's line and the home line for the run.
    assert base.print_closing_summary(outcome.summary_key) == list(
        scenario.lane_summary.get(lane, scenario.summary)
    )

    _assert_lane_disposition(lane, scenario, outcome, rig)


def _assert_lane_disposition(lane, scenario, outcome, rig):
    """What each lane does with the one reason: its own, kept apart."""
    if lane == "ask":
        run = rig["ask_run"]
        answer = Path(rig["answer_path"])
        question_dir = answer.parent
        partial = question_dir / f"partial_{STEM}.md"
        if scenario.reason == "gate":
            # Refused ahead of the stale-file sweep: nothing was dispatched.
            assert run.dispatched_specs == set()
            assert not answer.exists() and not partial.exists()
        elif scenario.reason is None:
            assert run.dispatched_specs == {SPEC}
            assert answer.read_text(encoding="utf-8").strip() == ANSWER
            assert not partial.exists()
        else:
            # Failed closed: the answer file set aside as `.violated` (never
            # deleted), the stdout text kept as a partial (none for a call
            # that raised: it read nothing back), and the error logged.
            assert not answer.exists()
            assert Path(str(answer) + ".violated").exists()
            assert run.answers_left_in_place == set()
            if scenario.raises:
                assert not partial.exists()
            else:
                assert ANSWER in partial.read_text(encoding="utf-8")
            assert scenario.reason in (
                question_dir / f"error_{STEM}.log"
            ).read_text(encoding="utf-8")
        return
    if lane == "review-code":
        success, issues, error = outcome.result
        failed = outcome.tracker.named("failed")
        if scenario.reason is None:
            assert success is True and failed == []
            assert [args[0] for args, _kw in outcome.tracker.named("done")] == [SPEC]
        else:
            assert issues is None
            assert [(args[0], kw["error"]) for args, kw in failed] == [(SPEC, error)]
        return
    result = outcome.result
    if scenario.reason is None:
        assert result["output"] == json.dumps(FINDINGS)
    else:
        # The output dropped, so `save_model_result` never parses it.
        assert result["output"] is None


# ---------------------------------------------------------------------------
# The output folder re-pointed during the window
# ---------------------------------------------------------------------------

#: The stub's invoker name on each lane's module (the rig's stubs).
_INVOKERS = {
    "ask": (ask_orchestrator, "invoke_with_provider"),
    "review-code": (cro, "invoke_with_file_output"),
    "review-plan": (base, "invoke_with_file_output"),
    "review-plan-stdout": (base, "invoke_with_provider"),
}

#: The folders the lanes write into under the output folder during the window.
_PHASE_DIRS = ("ask/q-12345678", "code-review", "review-plan")


#: The folders only one ignore list covers, per direction. The output folder
#: is followed through symlinks (``bypass_ignore_prefixes``); its own entry
#: under ``.multi-llm/`` is covered by ``.multi-llm`` on both sides anyway.
_DRIFT = {
    "symlink-replaced": ("src", "none"),
    "symlink-planted": ("none", "src"),
}


def _drift_line(before_only: str, after_only: str) -> str:
    return (
        f"{SANDBOX_BYPASSED_MARKER} codex: the plugin's own folders, which the "
        f"write check leaves out, resolved differently before and after this "
        f"unsandboxed slot's window (the plan's output folder is followed "
        f"through symlinks; before only: {before_only}; after only: "
        f"{after_only}); the check leaves out only what both cover, so a change "
        f"under the others is reported"
    )


@pytest.mark.parametrize("before, after, common, before_only, after_only", [
    # Nobody re-pointed anything: the list comes back as is, and no line.
    ([".multi-llm", "plans/feature"], [".multi-llm", "plans/feature/"],
     [".multi-llm", "plans/feature/"], "", ""),
    # Symlink replaced by a directory: `src` was only ever the symlink's target.
    ([".multi-llm", "src"], [".multi-llm", ".multi-llm/feature"],
     [".multi-llm", ".multi-llm/feature"], "src", ""),
    # Symlink planted: the reverse.
    ([".multi-llm", ".multi-llm/feature"], [".multi-llm", "src"],
     [".multi-llm", ".multi-llm/feature"], "", "src"),
    # Both sides own a folder the other lacks: only the shared tree is left.
    ([".multi-llm", "src", ".aider.tags.cache.v4"],
     [".multi-llm", "plans/feature", ".aider.tags.cache.v4"],
     [".multi-llm", ".aider.tags.cache.v4"], "src", "plans/feature"),
    # Nested on one side only: the deeper prefix is what both cover.
    (["plans"], ["plans/feature"], ["plans/feature"], "plans", ""),
])
def test_the_exemption_is_what_both_ignore_lists_cover(
    before, after, common, before_only, after_only
):
    assert base.common_ignore_prefixes(before, after) == common
    line = base.bypass_ignore_drift("codex", before, after)
    if not before_only and not after_only:
        assert line is None
    else:
        assert line == _drift_line(before_only or "none", after_only or "none")


@pytest.mark.parametrize("direction", ["symlink-replaced", "symlink-planted"])
@pytest.mark.parametrize("lane", LANES)
def test_a_reported_edit_survives_a_repointed_output_folder(
    lane, direction, rig, monkeypatch, capsys
):
    """The plan in ``.multi-llm/``, so re-pointing its output folder is a change
    inside the plugin's own, always-exempt tree, and an uncontained slot whose
    stream audit did not clear it edits tracked ``src/app.py`` in the window.

    ``symlink-replaced``: the output folder is a symlink to ``src/`` before the
    window and a directory after it. ``symlink-planted``: the reverse. Either
    ignore list alone covers ``src/``; every lane leaves out only what both
    cover (``SlotGuard``), so the edit is named, and one line names the drift.
    """
    root = rig["repo"]
    (root / "plans").rename(root / ".multi-llm")
    rig["plan"] = root / ".multi-llm" / "feature.md"
    output = root / ".multi-llm" / "feature"
    if direction == "symlink-replaced":
        output.symlink_to("../src", target_is_directory=True)
    rig["scenario"] = SCENARIOS["bypass-write-violation"]
    module, name = _INVOKERS[lane]
    invoke = getattr(module, name)

    def repoint_then_answer(**kwargs):
        if direction == "symlink-replaced":
            output.unlink()
            for sub in _PHASE_DIRS:
                (output / sub).mkdir(parents=True)
        else:
            subprocess.run(["rm", "-rf", str(output)], check=True)
            output.symlink_to("../src", target_is_directory=True)
            for sub in _PHASE_DIRS:
                (output / sub).mkdir(parents=True, exist_ok=True)
        return invoke(**kwargs)

    monkeypatch.setattr(module, name, repoint_then_answer)
    outcome = _drive(lane, rig)

    assert not outcome.success, outcome.result
    assert outcome.reason.startswith(
        f"{SANDBOX_BYPASSED_MARKER} codex: the working tree changed during this "
        f"unsandboxed slot's window ("
    ), outcome.reason
    assert "src/app.py" in outcome.reason.split("(", 1)[1].split(")", 1)[0]
    drift = _drift_line(*_DRIFT[direction])
    lines = _marker_lines(capsys.readouterr().out)
    assert drift in lines, lines
    assert lines.index(drift) < lines.index(outcome.reason), lines
    assert drift in base.print_closing_summary(outcome.summary_key)


# ---------------------------------------------------------------------------
# A normal run keeps its answer: both ignore lists cover the output folder
# ---------------------------------------------------------------------------


def _write_plugin_files(output: Path) -> None:
    """What the plugin writes into the output folder during a slot: the lanes'
    folders and a log in each (the ask lane's answer comes on top)."""
    for sub in _PHASE_DIRS:
        folder = output / sub
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "slot.log").write_text("log\n", encoding="utf-8")


def _grow_past_the_cap(path: Path) -> None:
    """A sparse file one byte over the hashing cap: its size counts, and a
    snapshot over the cap never reads it."""
    with path.open("wb") as handle:
        handle.truncate(base.PORCELAIN_HASH_CAP_BYTES + 1)


@pytest.mark.parametrize("write", [False, True], ids=["no-edit", "edit"])
@pytest.mark.parametrize(
    "case", ["created-in-window", "unchanged", "stable-symlink", "dirty-rerun"]
)
@pytest.mark.parametrize("lane", LANES)
def test_an_output_folder_nobody_re_points_stays_exempt(
    lane, case, write, rig, monkeypatch, capsys
):
    """No drift line and no false positive on a normal run, on every lane.

    ``created-in-window``: the output folder does not exist when the gate runs
    (removed there, ahead of ``begin()``'s ignore list) and the plugin creates
    it during the run (review-code's phase folder after the gate, the logs in
    the window); the list resolved before already names the path it gets.
    ``unchanged``: an existing folder nobody touches. ``stable-symlink``:
    the output folder is a symlink to an untracked ``out/`` that stays put
    through the window. ``dirty-rerun``: the folder holds an earlier run's
    files, one of them past the hashing cap, and ``src/app.py`` is already
    dirty; the output folder's entries are kept in the baseline apart from the
    rest, so they never switch off the content comparison (no
    ``[PORCELAIN_CHECK]`` note) and the edit below, which keeps the file's
    `` M`` status, is still seen. With ``edit`` the slot also changes tracked
    ``src/app.py``, which must then be the one path named: the plugin's own
    files stay left out.
    """
    root = rig["repo"]
    output = root / "plans" / "feature"
    if case == "stable-symlink":
        (root / "out").mkdir()
        output.symlink_to("../out", target_is_directory=True)
    elif case == "unchanged":
        output.mkdir(exist_ok=True)
    elif case == "dirty-rerun":
        (output / "review-plan").mkdir(parents=True, exist_ok=True)
        (output / "review-plan" / "earlier.json").write_text("[]\n", encoding="utf-8")
        _grow_past_the_cap(output / "review-plan" / "earlier.log")
        (root / "src" / "app.py").write_text("print('dirty before')\n", encoding="utf-8")
    scenario = SCENARIOS["bypass-write-violation" if write else "bypass-no-write"]
    rig["scenario"] = scenario
    gate = base.sandbox_refusal("codex", str(root))

    removed: List[bool] = []
    if case == "created-in-window":
        probe = base._fanout_sandbox_verdict

        def remove_then_probe(provider_name, cwd):
            if provider_name == "codex" and not removed:
                subprocess.run(["rm", "-rf", str(output)], check=True)
                removed.append(True)
            return probe(provider_name, cwd)

        monkeypatch.setattr(base, "_fanout_sandbox_verdict", remove_then_probe)

    module, name = _INVOKERS[lane]
    invoke = getattr(module, name)

    def plugin_writes_then_answer(**kwargs):
        _write_plugin_files(output)
        return invoke(**kwargs)

    monkeypatch.setattr(module, name, plugin_writes_then_answer)
    capsys.readouterr()

    outcome = _drive(lane, rig)
    out = capsys.readouterr().out

    if case == "created-in-window":
        assert removed == [True]
    assert (output / "review-plan" / "slot.log").is_file()
    assert outcome.reason == scenario.reason, out
    assert outcome.success is (scenario.reason is None)
    assert _marker_lines(out) == [gate.reason, *scenario.lines[1:]], out
    assert base.print_closing_summary(outcome.summary_key) == []
    assert "[PORCELAIN_CHECK]" not in out, out
    if case == "stable-symlink":
        assert output.is_symlink()
        assert (root / "out" / "review-plan" / "slot.log").is_file()


def test_two_plans_sharing_an_output_folder_stay_exempt(rig):
    """Two plans whose output folders are symlinks to one ``out/``: each
    slot's lists cover it before and after, so the plugin's write there is
    left out on both, with no drift line."""
    root = rig["repo"]
    (root / "out").mkdir()
    plans = []
    for name in ("one", "two"):
        plan = root / "plans" / f"{name}.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        (root / "plans" / name).symlink_to("../out", target_is_directory=True)
        plans.append(plan)
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "two plans")
    rig["scenario"] = SCENARIOS["bypass-no-write"]

    async def both():
        guards = [_guard(root, plan) for plan in plans]
        for guard in guards:
            assert isinstance(await guard.begin(), slot_guard.Ready)
        (root / "out" / "slot.log").write_text("log\n", encoding="utf-8")
        return [
            await guard.finish([NOT_CLEAN], [], summary_key=str(root)) for guard in guards
        ]

    assert _in_fanout(both) == [None, None]
    assert base.print_closing_summary(str(root)) == []


# ---------------------------------------------------------------------------
# The folders only one ignore list covers are hashed apart from the rest
# ---------------------------------------------------------------------------


def _guard(root: Path, plan: Path) -> "slot_guard.SlotGuard":
    return slot_guard.SlotGuard(
        provider_name="codex", display_name=SPEC, model_spec=SPEC,
        project_root=str(root), plan_path=plan,
    )


def _plan_in_the_plugin_folder(rig) -> Path:
    """Move the plan into ``.multi-llm/`` (so re-pointing its output folder is
    a change inside the always-exempt tree) and return the output folder."""
    root = rig["repo"]
    (root / "plans").rename(root / ".multi-llm")
    rig["plan"] = root / ".multi-llm" / "feature.md"
    return root / ".multi-llm" / "feature"


def _repoint(output: Path, target: Optional[str]) -> None:
    """Make the output folder a symlink to ``target``, or a directory when
    ``None``, with the lanes' phase folders in it."""
    if output.is_symlink():
        output.unlink()
    elif output.exists():
        shutil.rmtree(output)
    if target is None:
        output.mkdir()
    else:
        output.symlink_to(target, target_is_directory=True)
    for sub in _PHASE_DIRS:
        (output / sub).mkdir(parents=True, exist_ok=True)


def _named(reason: str) -> List[str]:
    return reason.split("(", 1)[1].split(")", 1)[0].split(", ")


def test_a_zone_is_hashed_under_its_own_cap(repo, capsys):
    """``porcelain_snapshot``'s zones: their entries are kept apart and
    hashed under a cap of their own, so a zone past it leaves the rest
    hashed, and its note is kept, not printed. Without the zone, the same
    tree is one snapshot past the cap."""
    (repo / "src" / "app.py").write_text("print('dirty')\n", encoding="utf-8")
    (repo / "out").mkdir()
    _grow_past_the_cap(repo / "out" / "big.log")
    snapshot = base.porcelain_snapshot(
        str(repo), ignore_prefixes=[".multi-llm"], zones=(["out"],)
    )
    assert capsys.readouterr().out == ""
    assert snapshot.hashed and set(snapshot.entries) == {"src/app.py"}
    assert snapshot.entries["src/app.py"].blob
    (zone,) = snapshot.zones
    assert set(zone.entries) == {"out/big.log"}
    assert not zone.hashed and "hashing cap" in zone.note
    whole = base.porcelain_snapshot(str(repo), ignore_prefixes=[".multi-llm"])
    assert not whole.hashed and whole.zones == ()
    assert "hashing cap" in capsys.readouterr().out


@pytest.mark.parametrize("direction", ["reverse", "forward"])
@pytest.mark.parametrize("lane", LANES)
def test_a_folder_past_the_cap_under_one_list_keeps_the_content_check(
    lane, direction, rig, monkeypatch, capsys
):
    """Tracked ``src/app.py`` is already dirty, so the slot's edit keeps its
    `` M`` status and only its content tells. The output folder is a directory
    before the window and a symlink to untracked ``payload/`` after it
    (``reverse``), or the other way round (``forward``), and ``payload/blob``
    grows past the hashing cap in the window.

    ``payload/`` is under one list only. Hashed with the rest, it would switch
    off the content comparison for every path and the edit would go unseen;
    the list that leaves ``payload/`` out (the after list for ``reverse``, the
    before list for ``forward``) sees it alone. Hashed apart, every lane names
    it, with no ``[PORCELAIN_CHECK]`` note. (The drift line is pinned above.)
    """
    root = rig["repo"]
    output = _plan_in_the_plugin_folder(rig)
    (root / "payload").mkdir()
    (root / "payload" / "blob").write_text("small\n", encoding="utf-8")
    (root / "src" / "app.py").write_text("print('dirty before')\n", encoding="utf-8")
    _repoint(output, "../payload" if direction == "forward" else None)
    rig["scenario"] = SCENARIOS["bypass-write-violation"]
    module, name = _INVOKERS[lane]
    invoke = getattr(module, name)

    def repoint_grow_then_answer(**kwargs):
        _repoint(output, None if direction == "forward" else "../payload")
        _grow_past_the_cap(root / "payload" / "blob")
        return invoke(**kwargs)  # the stub's write edits src/app.py

    monkeypatch.setattr(module, name, repoint_grow_then_answer)
    outcome = _drive(lane, rig)
    out = capsys.readouterr().out

    assert not outcome.success, out
    assert "src/app.py" in _named(outcome.reason), outcome.reason
    assert "[PORCELAIN_CHECK]" not in out, out


@pytest.mark.parametrize("lane", LANES)
def test_under_drift_what_the_after_list_saw_change_is_still_named(
    lane, rig, monkeypatch
):
    """The output folder is a symlink to untracked ``payload/`` before the
    window and a directory after it; in the window ``payload/blob`` grows past
    the hashing cap, its status unchanged, and untracked ``src/new.py`` is
    staged, its bytes unchanged.

    The list taken after the window, alone, hashes ``payload/`` with the rest,
    goes past the cap and compares by status, so it names the staged file; it
    has no baseline under ``payload/``, so it names the blob. Outside both
    lists a status change always counts, and under one list alone a path whose
    content cannot be compared counts, so both stay named.
    """
    root = rig["repo"]
    output = _plan_in_the_plugin_folder(rig)
    (root / "payload").mkdir()
    (root / "payload" / "blob").write_text("small\n", encoding="utf-8")
    (root / "src" / "new.py").write_text("print('new')\n", encoding="utf-8")
    _repoint(output, "../payload")
    rig["scenario"] = SCENARIOS["bypass-no-write"]
    module, name = _INVOKERS[lane]
    invoke = getattr(module, name)

    def repoint_grow_stage_then_answer(**kwargs):
        _repoint(output, None)
        _grow_past_the_cap(root / "payload" / "blob")
        _git(root, "add", "src/new.py")
        return invoke(**kwargs)

    monkeypatch.setattr(module, name, repoint_grow_stage_then_answer)
    outcome = _drive(lane, rig)

    assert not outcome.success, outcome.result
    assert {"payload/blob", "src/new.py"} <= set(_named(outcome.reason)), outcome.reason


def test_the_guard_keeps_the_content_check_when_one_list_tips_the_cap(rig):
    """The reverse case above, at the guard itself: the output folder becomes
    a symlink to ``payload/`` in the window and ``payload/blob`` grows past the
    cap; the edit to already-dirty ``src/app.py`` is named."""
    root = rig["repo"]
    output = _plan_in_the_plugin_folder(rig)
    output.mkdir()
    (root / "payload").mkdir()
    (root / "payload" / "blob").write_text("small\n", encoding="utf-8")
    app = root / "src" / "app.py"
    app.write_text("print('dirty before')\n", encoding="utf-8")
    rig["scenario"] = SCENARIOS["bypass-write-violation"]

    async def slot():
        guard = _guard(root, rig["plan"])
        assert isinstance(await guard.begin(), slot_guard.Ready)
        shutil.rmtree(output)
        output.symlink_to("../payload", target_is_directory=True)
        _grow_past_the_cap(root / "payload" / "blob")
        app.write_text("print('edited in the window')\n", encoding="utf-8")
        return await guard.finish([NOT_CLEAN], [], summary_key=str(root))

    reason = _in_fanout(slot)
    assert reason is not None and "src/app.py" in _named(reason), reason


@pytest.mark.parametrize("change", ["restored", "edited", "unchanged"])
@pytest.mark.parametrize("lane", LANES)
def test_a_folder_only_the_before_list_covers_keeps_its_baseline(
    lane, change, rig, monkeypatch, capsys
):
    """The output folder is a symlink to ``src/`` before the window and a
    directory after it, so ``src/`` is under the before list only, and
    ``src/app.py`` is already dirty. The baseline kept the output folder's
    entries apart, so they are compared like any other path:

    ``restored``: put back to clean in the window, named. ``edited``: edited
    again, same `` M`` status, named. ``unchanged``: left alone, not named,
    and the slot keeps its answer. (The drift line is pinned above.)
    """
    root = rig["repo"]
    output = _plan_in_the_plugin_folder(rig)
    app = root / "src" / "app.py"
    app.write_text("print('dirty before')\n", encoding="utf-8")
    _repoint(output, "../src")
    rig["scenario"] = SCENARIOS["bypass-no-write"]
    module, name = _INVOKERS[lane]
    invoke = getattr(module, name)

    def repoint_change_then_answer(**kwargs):
        _repoint(output, None)
        if change == "restored":
            app.write_text("print('app')\n", encoding="utf-8")
        elif change == "edited":
            app.write_text("print('edited in the window')\n", encoding="utf-8")
        return invoke(**kwargs)

    monkeypatch.setattr(module, name, repoint_change_then_answer)
    outcome = _drive(lane, rig)
    out = capsys.readouterr().out

    if change == "unchanged":
        assert outcome.success and outcome.reason is None, out
    else:
        assert not outcome.success, out
        assert _named(outcome.reason) == ["src/app.py"], outcome.reason
