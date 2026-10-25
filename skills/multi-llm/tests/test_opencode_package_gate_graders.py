"""Row 8's graders against synthetic bundles: every check, and a mutation that fails it.

The five `opencode-package-gate-*` rows (`test_container_opencode_package_gate.py`)
pass because the product is correct, and so would a grader whose check had
quietly stopped measuring anything. This module is the other half: no Docker,
no image, in the plain suite. For each scenario it writes a minimal bundle in
the shape `container/opencode_package_gate.py` writes — modelled on the
2026-10-25 full-gate bundles, pids and argv shapes included — checks that every
check passes on it, and then, per check, applies ONE mutation that breaks
exactly the property the check names and asserts that check fails. A mutation
may only take down the other checks it declares as `collateral` (a launch that
did not happen leaves no banner to claim), so a mutation that breaks something
else, or a check that a mutation no longer reaches, is a failure here.

`test_every_check_has_a_mutation` is the guard: the check names each grader
emits on its good bundle must equal the names the mutation table covers, so a
new check without a negative case fails the suite.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Callable, NamedTuple

import pytest

from .container.opencode_package_gate_graders import (
    ABORTED_BY_DISPATCH,
    CHECKOUT_SKILL_DIR,
    INSTALLED_ROOT,
    PAYLOAD_SKILL_DIR,
    SCENARIOS,
    Bundle,
    grade_bundle,
    redacted,
    registry_entry,
    result_from_bundle,
    row_id,
    write_grader,
)

RUN_ID = "a16a079f-2c79-4787-9b08-aabbc076b1e0"
OTHER_RUN_ID = "58bb12ba-1eac-416d-86d3-5dda8c5dfb94"
WORKSPACE = "/work/sample-project"
PLAN = "plans/auth-feature.md"
PLAN_DIR = "plans/auth-feature"
RUN_DIR = f"{PLAN_DIR}/ask/what-does-this-plan-change-8a3c7cf7"
LOG = f"{WORKSPACE}/{RUN_DIR}/orchestrator-run.log"
ANSWER = "answer_opencode_stub_stub-model.md"
UV = "/opt/mllm/bin/uv"
QUESTION = "What does this plan change?"

#: The pids of the real detach bundle: the host `opencode run`, the worker's
#: wrapper / uv / python and their group, and its three polls.
HOST, WRAPPER, UV_PID, WORKER, GROUP = 39, 109, 110, 113, 108
POLL_PIDS = (126, 155, 165)
#: The detach launch's own chain under the host, for the handshake race: the
#: plugin's uv wrapper, the real uv, the launcher's python, and the double
#: spawn's intermediate, a fork of the launcher whose `setsid` made it the
#: worker group's leader (so its pid IS the group's).
L_WRAPPER, L_UV, LAUNCHER, INTERMEDIATE = 101, 102, 104, GROUP
#: The cancellation bundle's: `opencode serve`, the attached client, the
#: worker's three and its group, the held provider, and the tool's abort.
SERVE, CLIENT, C_WRAPPER, C_UV, C_WORKER, C_GROUP = 39, 51, 118, 119, 122, 117
C_PROVIDER, C_SLEEP, C_ABORT = 159, 161, 171
FINAL = 4.832


# --------------------------------------------------------------------------- #
# Building a bundle
# --------------------------------------------------------------------------- #

def tool_argv(*args: str) -> list[str]:
    """The tool's own uv argv, as its result reports it."""
    return [UV, "run", "--frozen", "--no-dev", "--project", PAYLOAD_SKILL_DIR, "--",
            "python", f"{PAYLOAD_SKILL_DIR}/multi_llm.py", *args]


def receipt(seq: int, *args: str, exit: int = 0, result=None, run=None,
            banner=None, project: str = PAYLOAD_SKILL_DIR) -> dict:
    """One uv-wrapper receipt, as `wrapper.py` writes it."""
    return {"seq": seq, "exit": exit, "result": result, "run": run, "banner": banner,
            "argv": ["run", "--no-sync", "--frozen", "--no-dev", "--project", project,
                     "--", "python", f"{project}/multi_llm.py", *args]}


def watch(touched: dict | None = None, censuses: int = 24) -> dict:
    """A `WorkspaceWatch.summary()`."""
    return {"scope": "workspace", "root": WORKSPACE, "excluded": [".git/"],
            "interval_s": 0.25, "censuses": censuses, "files_at_start": 9,
            "truncated": False, "touched": touched or {}}


def touched(kinds=("created",), present: bool = True, at: float = 1.0) -> dict:
    return {"first_seen": at, "kinds": list(kinds), "present_at_end": present}


def proc(pid: int, ppid: int, pgid: int, args: str, *, sid: int | None = None,
         first_ppid: int | None = None, state: str = "S", first_seen: float = 1.0,
         last_seen: float = FINAL, last_alive: float | None = None, **extra) -> dict:
    entry = {"pid": pid, "ppid": ppid, "first_ppid": ppid if first_ppid is None
             else first_ppid, "pgid": pgid, "sid": pgid if sid is None else sid,
             "state": state, "args": args, "start": 26842000 + pid,
             "first_seen": first_seen, "last_seen": last_seen,
             "last_alive": (last_seen if state != "Z" else None)
             if last_alive is None else last_alive}
    entry.update(extra)
    return entry


def common_spec(scenario: str) -> dict:
    """What every row's bundle carries, whatever its scenario."""
    return {
        "scenario": scenario,
        "row_result": {"exit_code": 0, "timed_out": False, "cap_reason": None,
                       "wall_clock_s": 17.7},
        "posture_inside": {"network": {"declared": "none", "default_route": False,
                                       "routed_via": []}},
        "posture_host": {"captured": True, "mismatches": []},
        "stub": [{"path": "/v1/chat/completions", "matched": "ask"}],
        "host_stubs": {},
        "receipts": [],
        "processes": None,
        "tap": None,
        "messages": None,
        "artifacts": {},
        "gate": {
            "probe": f"gate-{scenario}",
            "scenario": scenario,
            "installed_root": INSTALLED_ROOT,
            "opencode_version": "1.19.4",
            "file_shims": [],
            "wiring": {"plugin": [INSTALLED_ROOT], "provider": "hoststub"},
            "debug_config": {
                "parsed": True, "command_keys": ["multi-llm"],
                "plugin": [f"file://{INSTALLED_ROOT}"],
                "plugin_origins": [{"scope": "global", "spec": f"file://{INSTALLED_ROOT}",
                                    "source": "/home/e2e/.config/opencode"}],
                "template_matches_payload": True,
                "description_matches_payload": True,
            },
            "plan_output_dir": PLAN_DIR,
            "run_records": [],
            "legs": [],
        },
    }


def scripted(label: str, mode: str, extra_flags=(), facts=()) -> list[dict]:
    """The host stub's log for one leg: the one scripted `multi_llm` call."""
    return [{"scripted": {"phase": "call", "name": "multi_llm",
                          "arguments": {"mode": mode, "extra_flags": list(extra_flags)}},
             "facts": list(facts)},
            {"scripted": {"phase": "terminal"}, "facts": []}]


def init_status_spec() -> dict:
    spec = common_spec("init-status")
    init_payload = {"mode": "init", "exit": 0, "state": "done"}
    status_payload = {"mode": "status", "exit": 2, "state": "nothing-to-do"}
    spec["gate"]["workspace_set_aside"] = [".multi-llm/providers.yaml"]
    spec["gate"]["legs"] = [
        {"label": "init", "exit": 0, "timed_out": False, "stub_log": "gate-stub-init.jsonl",
         "error_events": [],
         "tool_parts": [{"status": "completed", "result": {
             "ok": True, "exit": 0, "state": "done", "result": init_payload,
             "argv": tool_argv("init", "--host", "opencode", "--yes", "--exec-subagents")}}],
         "census_delta": {".multi-llm/providers.yaml": [None, "cb517ffe"]},
         "workspace_watch": watch({".multi-llm/providers.yaml": touched()})},
        {"label": "status", "exit": 0, "timed_out": False,
         "stub_log": "gate-stub-status.jsonl", "error_events": [],
         "tool_parts": [{"status": "completed", "result": {
             "ok": True, "exit": 2, "state": "nothing-to-do", "result": status_payload,
             "argv": tool_argv("status", PLAN, "--host", "opencode", "--yes",
                               "--exec-subagents")}}],
         "census_delta": {},
         "workspace_watch": watch()},
    ]
    spec["host_stubs"] = {
        "init": scripted("init", "init", facts=["multi_llm_command"]),
        "status": scripted("status", "status", facts=["multi_llm_command"]),
    }
    spec["receipts"] = [
        receipt(1, "init", "--host", "opencode", "--yes", "--exec-subagents",
                result=init_payload),
        receipt(2, "status", PLAN, "--host", "opencode", "--yes", "--exec-subagents",
                exit=2, result=status_payload),
    ]
    return spec


def _worker_args(run_id: str = RUN_ID) -> str:
    return (f"{PAYLOAD_SKILL_DIR}/multi_llm.py ask --plan-file {WORKSPACE}/{PLAN} "
            f"--quick --internal-worker --internal-run-id {run_id}")


def _poll_args(run_id: str = RUN_ID) -> str:
    return (f"python3 {UV} run --frozen --no-dev --project {PAYLOAD_SKILL_DIR} -- "
            f"python {PAYLOAD_SKILL_DIR}/multi_llm.py status {WORKSPACE}/{PLAN} "
            f"--host opencode --yes --exec-subagents --view run --wait --run-id {run_id}")


def detached_spec(scenario: str) -> dict:
    """Rows 2-4: one `ask --quick` the tool detaches, polls three times, returns."""
    spec = common_spec(scenario)
    banner = {"run_id": RUN_ID, "log": LOG, "mode": "ask", "pid": str(WORKER)}
    run = {"run_id": RUN_ID, "state": "done", "exit_code": 0, "error": None,
           "pid": WORKER, "pgid": GROUP, "detached": True, "host": "opencode",
           "mode": "ask", "log_path": LOG, "route": "package"}
    ask = ("ask", PLAN, QUESTION, "--host", "opencode", "--yes", "--exec-subagents")
    poll = ("status", f"{WORKSPACE}/{PLAN}", "--host", "opencode", "--yes",
            "--exec-subagents", "--view", "run", "--wait", "--run-id", redacted(RUN_ID))
    running = {"run_id": RUN_ID, "state": "running", "exit_code": None}
    spec["receipts"] = [
        receipt(1, *ask, "--detach", "--quick", result={"mode": "ask", "exit": 0,
                                                        "state": "done"},
                run=running, banner=banner),
        receipt(2, *poll, exit=2, result={"exit": 2, "state": "running"}, run=running),
        receipt(3, *poll, exit=2, result={"exit": 2, "state": "running"}, run=running),
        receipt(4, "ask", "--plan-file", f"{WORKSPACE}/{PLAN}", "--quick",
                "--internal-worker", "--internal-run-id", redacted(RUN_ID)),
        receipt(5, *poll, exit=0, result={"exit": 0, "state": "done"},
                run={"run_id": RUN_ID, "state": "done", "exit_code": 0}),
    ]
    files = [".lock", ".status.json", ".status.lock", ANSWER, "answers.md",
             "log_opencode_stub_stub-model.txt", "orchestrator-run.log"]
    spec["gate"].update({
        "host_pid": HOST,
        "poll_s": 5, "slow_provider_s": 12,
        "run_records": [{"dir": RUN_DIR, "run": run, "files": files,
                         "models": [["opencode:stub/stub-model", {"state": "done"}]]}],
    })
    delta = {f"{RUN_DIR}/{name}": [None, "0" * 8] for name in files}
    spec["gate"]["legs"] = [{
        "label": "ask", "pid": HOST, "exit": 0, "timed_out": False,
        "stub_log": "gate-stub-ask.jsonl", "error_events": [],
        "tool_parts": [{"status": "completed", "error": None, "title": "", "result": {
            "ok": True, "exit": 0, "state": "done", "run_id": RUN_ID, "log": LOG,
            "polls": 3, "launch_state": "done", "artifacts": [LOG],
            "argv": tool_argv(*ask, "--detach", "--quick")}}],
        "census_delta": delta,
        "workspace_watch": watch({
            **{name: touched() for name in delta},
            registry_entry(RUN_ID): touched(present=False)}),
    }]
    spec["host_stubs"] = {"ask": scripted("ask", "ask", extra_flags=["--quick"])}
    host_args = ("/opt/mllm/real/opencode run --command multi-llm --format json "
                 "--model hoststub/stub-model -- --ask plans/auth-feature.md")
    worker = dict(samples_alive=140, first_chain=[UV_PID, WRAPPER, 1],
                  ancestors_seen=[1, WRAPPER, UV_PID])
    processes = [
        proc(1, 0, 1, "/usr/bin/python3 /opt/mllm/rig/entrypoint.py", first_seen=0.0),
        proc(HOST, 1, 1, host_args, sid=1, first_seen=0.2, last_seen=17.63),
        proc(WRAPPER, 1, GROUP, f"python3 {UV} run --frozen -- python {_worker_args()}",
             first_seen=1.607, state="Z", last_alive=16.3, samples_alive=140,
             first_chain=[1], ancestors_seen=[1]),
        proc(UV_PID, WRAPPER, GROUP, f"/opt/mllm/real/uv run -- python {_worker_args()}",
             first_seen=1.607, last_seen=16.2, samples_alive=140,
             first_chain=[WRAPPER, 1], ancestors_seen=[1, WRAPPER]),
        proc(WORKER, UV_PID, GROUP, f"/opt/mllm/skill/.venv/bin/python {_worker_args()}",
             first_seen=1.607, last_seen=16.2, **worker),
        proc(150, WORKER, GROUP, "/bin/sh /work/out/gate-bin/opencode run --format json",
             first_seen=2.8, last_seen=16.2),
    ]
    for index, pid in enumerate(POLL_PIDS):
        processes.append(proc(pid, HOST, pid, _poll_args(), first_seen=1.9 + 5 * index,
                              last_seen=7.0 + 5 * index))
    spec["processes"] = {"interval_s": 0.1, "samples": 178,
                         "snapshots": {"worker-live": "gate-ps-worker-live.txt",
                                       "after": "gate-ps-after.txt"},
                         "processes": processes}
    spec["artifacts"] = {
        f"{Path(PLAN_DIR).name}/ask/what-does-this-plan-change-8a3c7cf7/{name}":
        "# answer\nThe plan adds session auth.\n" for name in (ANSWER, "answers.md")}
    return spec


def _launcher_args() -> str:
    return (f"/opt/mllm/skill/.venv/bin/python {PAYLOAD_SKILL_DIR}/multi_llm.py ask {PLAN} "
            f"--host opencode --yes --exec-subagents --detach --quick")


def handshake_race(spec: dict) -> dict:
    """The detach bundle with one 0.1 s sample landing inside the detach
    handshake: the worker's wrapper first seen under the intermediate (a fork of
    the launcher, gone by the next sample), its chain then still running through
    the launcher and the plugin's uv into opencode; reparented to init after.
    The wrapper's child is caught in that sample too. Only the samples after
    the handshake count, so this grades green."""
    launcher = _launcher_args()
    spec["processes"]["processes"] += [
        proc(L_WRAPPER, HOST, L_WRAPPER, f"python3 {UV} run --frozen -- python {launcher}",
             first_seen=1.4, last_seen=1.9),
        proc(L_UV, L_WRAPPER, L_WRAPPER, f"/opt/mllm/real/uv run -- python {launcher}",
             first_seen=1.4, last_seen=1.9),
        proc(LAUNCHER, L_UV, L_WRAPPER, launcher, first_seen=1.5, last_seen=1.9),
        proc(INTERMEDIATE, LAUNCHER, GROUP, launcher, first_seen=1.607, last_seen=1.607),
    ]
    before = [INTERMEDIATE, LAUNCHER, L_UV, L_WRAPPER, HOST, 1]
    process(spec, WRAPPER).update(first_ppid=INTERMEDIATE, first_chain=before,
                                  ancestors_seen=sorted(set(before)))
    process(spec, UV_PID).update(first_chain=[WRAPPER, *before],
                                 ancestors_seen=sorted({WRAPPER, *before}))
    return spec


def cancellation_spec() -> dict:
    spec = common_spec("cancellation")
    banner = {"run_id": RUN_ID, "log": LOG, "mode": "ask", "pid": str(C_WORKER)}
    running = {"run_id": RUN_ID, "state": "running", "exit_code": None, "error": None,
               "pid": C_WORKER, "pgid": C_GROUP, "detached": True, "log_path": LOG}
    aborted = dict(running, state="aborted", exit_code=3, error=ABORTED_BY_DISPATCH)
    abort_args = (f"python3 {UV} run --frozen --no-dev --project {PAYLOAD_SKILL_DIR} -- "
                  f"python {PAYLOAD_SKILL_DIR}/multi_llm.py status {WORKSPACE}/{PLAN} "
                  f"--host opencode --yes --exec-subagents --abort --run-id {RUN_ID}")
    spec["receipts"] = [
        receipt(1, "ask", PLAN, QUESTION, "--host", "opencode", "--yes",
                "--exec-subagents", "--detach", "--quick",
                result={"mode": "ask", "exit": 0, "state": "done"},
                run={"run_id": RUN_ID, "state": "running"}, banner=banner),
        receipt(2, "status", f"{WORKSPACE}/{PLAN}", "--host", "opencode", "--yes",
                "--exec-subagents", "--abort", "--run-id", redacted(RUN_ID),
                result={"mode": "status", "exit": 0, "state": "done", "phase": None,
                        "next": None, "artifacts": []}),
    ]
    poll = {"pid": 135, "run_id": RUN_ID, "args": _poll_args()}
    spec["gate"].update({
        "serve_pid": SERVE,
        "tap": "gate-tool-tap.jsonl",
        "poll_s": 5, "slow_provider_s": 900,
        "run_records": [{"dir": RUN_DIR, "run": aborted, "files": [],
                         "models": []}],
        "abort": {
            "method": "POST /session/{id}/abort",
            "via": "opencode serve on http://127.0.0.1:4196",
            "sent": True, "http_status": 200, "response": "true",
            "session_id": "ses_e8c8dfb6dffeNcEbw7uhV0qLxw",
            "sessions_listed": ["ses_e8c8dfb6dffeNcEbw7uhV0qLxw"],
            "sent_at_s": 2.666, "tap_seen": True,
            "supervising_poll": dict(poll),
            "supervising_at_abort": dict(poll),
            "provider_at_abort": {"pid": C_PROVIDER, "pgid": C_GROUP,
                                  "args": "/bin/sh /work/out/gate-bin/opencode run"},
            "run_records_at_abort": [{"dir": RUN_DIR, "run": dict(running)}],
            "dispatch": {"interval_s": 0.02, "scans": 140, "seen": True, "at_s": 2.81,
                         "abort": {"pid": C_ABORT, "ppid": SERVE, "args": abort_args},
                         "workers": [{"pid": pid, "ppid": ppid, "pgid": C_GROUP,
                                      "sid": C_GROUP, "state": "S"}
                                     for pid, ppid in ((C_WRAPPER, 1), (C_UV, C_WRAPPER),
                                                       (C_WORKER, C_UV))],
                         "run": dict(running)},
        },
    })
    spec["gate"]["legs"] = [{
        "label": "ask", "pid": CLIENT, "exit": 0, "stub_log": "gate-stub-ask.jsonl",
        "error_events": [{"name": "MessageAbortedError", "data": {"message": "Aborted"}}],
        "tool_parts": [], "census_delta": {}}]
    spec["host_stubs"] = {"ask": scripted("ask", "ask", extra_flags=["--quick"])}
    report = {"state": "needs-human", "ok": False, "reason": "the call was cancelled",
              "run_id": RUN_ID, "log": LOG, "abort_dispatched": True,
              "message": f"the call was cancelled. The detached run {RUN_ID} was "
                         f"confirmed aborted. Log: {LOG}."}
    spec["tap"] = [{"tool": "multi_llm", "output": json.dumps(report)}]
    spec["messages"] = [{"parts": [{"type": "tool", "tool": "multi_llm",
                                    "state": {"status": "error",
                                              "error": "Tool execution aborted"}}]}]
    worker = _worker_args()
    spec["processes"] = {"interval_s": 0.1, "samples": 50, "snapshots": {}, "processes": [
        proc(1, 0, 1, "/usr/bin/python3 /opt/mllm/rig/entrypoint.py", first_seen=0.0),
        proc(SERVE, 1, 1, "/opt/mllm/real/opencode serve --port 4196", sid=1,
             first_seen=0.1, last_seen=4.734),
        proc(CLIENT, 1, 1, "/opt/mllm/real/opencode run --command multi-llm --attach "
             "http://127.0.0.1:4196", sid=1, first_seen=0.7, last_seen=3.526),
        proc(C_WRAPPER, 1, C_GROUP, f"python3 {UV} run -- python {worker}", state="Z",
             first_seen=1.911, last_alive=3.425, samples_alive=16, first_chain=[1],
             ancestors_seen=[1]),
        proc(C_UV, 1, C_GROUP, f"/opt/mllm/real/uv run -- python {worker}",
             first_ppid=C_WRAPPER, state="Z", first_seen=1.911, last_alive=3.425),
        proc(C_WORKER, C_UV, C_GROUP, f"/opt/mllm/skill/.venv/bin/python {worker}",
             first_seen=1.911, last_seen=3.425),
        proc(C_PROVIDER, C_WORKER, C_GROUP, "/bin/sh /work/out/gate-bin/opencode run",
             first_seen=3.122, last_seen=3.425),
        proc(C_SLEEP, 1, C_GROUP, "sleep 900", first_ppid=C_PROVIDER, state="Z",
             first_seen=3.122, last_alive=3.425),
        proc(135, SERVE, 135, _poll_args(), first_seen=2.217, last_seen=3.223),
        proc(C_ABORT, SERVE, C_ABORT, abort_args, first_seen=3.425, last_seen=3.626),
    ]}
    return spec


def good_spec(scenario: str) -> dict:
    """A fresh good bundle spec, through JSON so no two readings share an object
    and a mutation of one cannot move another."""
    if scenario == "init-status":
        spec = init_status_spec()
    elif scenario == "cancellation":
        spec = cancellation_spec()
    else:
        spec = detached_spec(scenario)
    return json.loads(json.dumps(spec))


def _dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _dump_lines(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_bundle(spec: dict, out: Path) -> Path:
    """Lay `spec` out as `runs/<utc>/<row-id>/` is laid out."""
    out.mkdir(parents=True, exist_ok=True)
    rid = row_id(spec["scenario"])
    _dump(out / "row-result.json", dict(spec["row_result"], row_id=rid))
    _dump(out / "posture-inside.json", spec["posture_inside"])
    _dump(out / "posture-host.json", spec["posture_host"])
    _dump(out / "opencode-gate.json", spec["gate"])
    _dump_lines(out / "stub.jsonl", spec["stub"])
    for label, rows in spec["host_stubs"].items():
        _dump_lines(out / f"gate-stub-{label}.jsonl", rows)
    for item in spec["receipts"]:
        _dump(out / "receipts" / rid / f"{item['seq']:03d}.receipt.json",
              dict(item, row_id=rid))
    if spec["processes"] is not None:
        _dump(out / "gate-processes.json", spec["processes"])
    if spec["tap"] is not None:
        _dump_lines(out / "gate-tool-tap.jsonl", spec["tap"])
    if spec["messages"] is not None:
        _dump(out / "gate-session-messages.json", spec["messages"])
    for name, text in spec["artifacts"].items():
        path = out / "artifacts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return out


def grade(spec: dict, tmp_path: Path) -> dict:
    out = write_bundle(spec, tmp_path / row_id(spec["scenario"]))
    checks, _observed, _method = grade_bundle(out, spec["scenario"])
    return checks


# --------------------------------------------------------------------------- #
# Reaching into a spec
# --------------------------------------------------------------------------- #

def leg(spec: dict, label: str) -> dict:
    return next(item for item in spec["gate"]["legs"] if item["label"] == label)


def tool(spec: dict, label: str) -> dict:
    return leg(spec, label)["tool_parts"][0]["result"]


def process(spec: dict, pid: int) -> dict:
    return next(p for p in spec["processes"]["processes"] if p["pid"] == pid)


def receipt_with(spec: dict, *words: str) -> dict:
    return next(r for r in spec["receipts"] if all(w in r["argv"] for w in words))


def run_record(spec: dict) -> dict:
    return spec["gate"]["run_records"][0]["run"]


def abort(spec: dict) -> dict:
    return spec["gate"]["abort"]


def tap_report(spec: dict, **changes) -> None:
    report = json.loads(spec["tap"][0]["output"])
    report.update(changes)
    spec["tap"][0]["output"] = json.dumps(report)


def set_project(item: dict, project: str) -> None:
    argv = item["argv"]
    argv[argv.index("--project") + 1] = project


# --------------------------------------------------------------------------- #
# The mutations: per check, what breaks exactly its property
# --------------------------------------------------------------------------- #

class Mutation(NamedTuple):
    name: str
    apply: Callable[[dict], None]
    #: Checks the broken property necessarily takes down with it.
    collateral: frozenset = frozenset()


def _two_scripted_calls(spec):
    label = spec["gate"]["legs"][-1]["label"]
    spec["host_stubs"][label].insert(0, copy.deepcopy(spec["host_stubs"][label][0]))


def _extra_checkout_argument(spec):
    item = spec["receipts"][0]
    item["argv"] += ["--config", f"{CHECKOUT_SKILL_DIR}/providers.yaml"]


def common_mutations() -> dict[str, list[Mutation]]:
    """The nine checks every grader runs, and what fails each on any scenario."""
    return {
        "row_completed": [
            Mutation("container exited non-zero",
                     lambda s: s["row_result"].update(exit_code=1)),
            Mutation("row timed out", lambda s: s["row_result"].update(timed_out=True)),
        ],
        "offline_no_credentials": [
            Mutation("a default route inside the container", lambda s: s[
                "posture_inside"]["network"].update(default_route=True)),
            Mutation("docker inspect disagreed with posture.yaml", lambda s: s[
                "posture_host"].update(mismatches=["NetworkMode: bridge != none"])),
        ],
        "package_declared": [Mutation(
            "opencode.json declares another plugin",
            lambda s: s["gate"]["wiring"].update(plugin=["opencode-multi-llm@0.1.1"]))],
        "plugin_loaded_from_installed_root": [Mutation(
            "opencode loaded the plugin from a checkout",
            lambda s: s["gate"]["debug_config"]["plugin_origins"][0].update(
                spec="file:///home/someone/checkout/packaging/opencode"))],
        "no_file_shim": [Mutation(
            "a file-route /multi-llm beside the plugin",
            lambda s: s["gate"].update(
                file_shims=["/home/e2e/.config/opencode/command/multi-llm.md"]))],
        "dispatcher_ran_from_installed_payload": [Mutation(
            "a dispatcher invocation ran another install's skill dir",
            lambda s: set_project(s["receipts"][0],
                                  "/home/e2e/.cache/opencode/node_modules/"
                                  "opencode-multi-llm/skills/multi-llm"))],
        "checkout_skill_never_dispatched": [Mutation(
            "a dispatcher argv names the checkout copy", _extra_checkout_argument)],
        "every_leg_made_one_scripted_call": [Mutation(
            "the host stub answered a second multi_llm call", _two_scripted_calls)],
        "stub_answered_every_request": [Mutation(
            "the main stub saw a request it had no answer for",
            lambda s: s["stub"].append({"path": "/v1/chat/completions",
                                        "unmatched": True}))],
    }


def init_status_mutations() -> dict[str, list[Mutation]]:
    def _no_tool_argv(spec):
        for label in ("init", "status"):
            tool(spec, label).pop("argv")

    def _status_terminal_zero(spec):
        result = tool(spec, "status")
        result.update(exit=0, state="done")
        result["result"].update(exit=0, state="done")

    return {
        "command_registered": [Mutation(
            "no command['multi-llm'] in debug config",
            lambda s: s["gate"]["debug_config"].update(command_keys=["review"]))],
        "command_is_the_payloads_own": [
            Mutation("a template that is not the payload's",
                     lambda s: s["gate"]["debug_config"].update(
                         template_matches_payload=False)),
            Mutation("a description that is not the payload's",
                     lambda s: s["gate"]["debug_config"].update(
                         description_matches_payload=False)),
        ],
        # The tool's argv is where the skill dir is read; with none to read,
        # the field must not fall back to a constant.
        "command_ran_the_payload_skill_dir": [Mutation(
            "the tool's results carry no argv to read a skill dir off", _no_tool_argv)],
        "command_expanded_into_the_turn": [Mutation(
            "the host never sent the payload's /multi-llm text",
            lambda s: s["host_stubs"]["init"][0].update(facts=[]))],
        "opencode_runs_exited_zero": [Mutation(
            "the status leg's opencode run exited 1",
            lambda s: leg(s, "status").update(exit=1))],
        "init_returned_success": [Mutation(
            "init reported failure",
            lambda s: tool(s, "init").update(ok=False, state="failed", exit=1))],
        "init_created_project_files": [
            Mutation("init wrote a file outside .multi-llm/ and removed it, seen "
                     "only by the watch",
                     lambda s: leg(s, "init")["workspace_watch"]["touched"].update(
                         {"README.md.tmp": touched(present=False)})),
            Mutation("init's census delta has no providers.yaml",
                     lambda s: leg(s, "init").update(census_delta={})),
            Mutation("init's watch took a single census",
                     lambda s: leg(s, "init")["workspace_watch"].update(censuses=1)),
        ],
        "status_returned_documented_result": [Mutation(
            "status answered exit 0 `done` for a plan with no run",
            _status_terminal_zero)],
        "status_wrote_nothing": [
            Mutation("status created and removed a file, seen only by the watch",
                     lambda s: leg(s, "status")["workspace_watch"]["touched"].update(
                         {".multi-llm/status.lock": touched(present=False)})),
            Mutation("status's watch hit its file bound",
                     lambda s: leg(s, "status")["workspace_watch"].update(
                         truncated=True)),
        ],
    }


def detach_mutations() -> dict[str, list[Mutation]]:
    launch_gone = {"banner_names_a_run", "tool_claimed_the_banner_run",
                   "run_record_is_the_claimed_detached_run"}

    def _second_launch(spec):
        spec["receipts"].append(dict(copy.deepcopy(spec["receipts"][0]), seq=9))

    def _workers(spec, change):
        for pid in (WRAPPER, UV_PID, WORKER):
            change(process(spec, pid))

    return {
        "tool_composed_detach": [
            Mutation("the tool's argv has no --detach",
                     lambda s: tool(s, "ask")["argv"].remove("--detach")),
            Mutation("the model typed --detach itself",
                     lambda s: s["host_stubs"]["ask"][0]["scripted"]["arguments"][
                         "extra_flags"].append("--detach")),
        ],
        "one_detached_launch": [Mutation("two detached launches", _second_launch,
                                         launch_gone)],
        "banner_names_a_run": [Mutation(
            "the launch banner names no run",
            lambda s: s["receipts"][0]["banner"].update(run_id=""),
            {"tool_claimed_the_banner_run"})],
        "tool_claimed_the_banner_run": [Mutation(
            "the banner printed a different run than the tool claimed",
            lambda s: s["receipts"][0]["banner"].update(run_id=OTHER_RUN_ID))],
        "run_record_is_the_claimed_detached_run": [Mutation(
            "the claimed run's record is not detached",
            lambda s: run_record(s).update(detached=False))],
        "worker_observed_running": [Mutation(
            "no process sample while the worker ran",
            lambda s: s["processes"]["snapshots"].pop("worker-live"))],
        "worker_in_its_own_process_group": [
            Mutation("the worker shares opencode's process group",
                     lambda s: _workers(s, lambda p: p.update(pgid=1, sid=1))),
            Mutation("the worker leads a group but not a session",
                     lambda s: _workers(s, lambda p: p.update(sid=1))),
        ],
        "worker_outside_opencode_tree_in_every_sample": [
            Mutation("a sample saw opencode on the worker's chain",
                     lambda s: process(s, WORKER)["ancestors_seen"].append(HOST)),
            Mutation("a sample saw the worker a direct child of opencode",
                     lambda s: process(s, WRAPPER).update(
                         first_chain=[HOST, 1], ancestors_seen=[1, HOST])),
            Mutation("the samples never measured the worker's chain",
                     lambda s: [process(s, pid).pop("ancestors_seen")
                                for pid in (WRAPPER, UV_PID, WORKER)]),
            # The handshake race is excused only for the double spawn's own
            # intermediate, briefly, with the worker then outside opencode.
            Mutation("the first parent outlived the handshake: the worker sat in "
                     "opencode's tree for seconds",
                     lambda s: process(handshake_race(s), INTERMEDIATE).update(
                         last_seen=4.1, last_alive=4.1)),
            Mutation("the first parent was the launcher itself, not a fork of it",
                     lambda s: process(handshake_race(s), WRAPPER).update(
                         first_ppid=LAUNCHER,
                         first_chain=[LAUNCHER, L_UV, L_WRAPPER, HOST, 1],
                         ancestors_seen=sorted({LAUNCHER, L_UV, L_WRAPPER, HOST, 1}))),
            Mutation("after the handshake the worker's parent is opencode itself",
                     lambda s: process(handshake_race(s), WRAPPER).update(ppid=HOST)),
            Mutation("a worker's child ends up reparented to opencode",
                     lambda s: process(handshake_race(s), UV_PID).update(ppid=HOST)),
            Mutation("a worker process first seen under the launcher's uv, not the worker",
                     lambda s: process(handshake_race(s), UV_PID).update(
                         first_ppid=L_UV, first_chain=[L_UV, L_WRAPPER, HOST, 1],
                         ancestors_seen=sorted({L_UV, L_WRAPPER, HOST, 1, WRAPPER}))),
        ],
    }


def supervision_mutations() -> dict[str, list[Mutation]]:
    polls_gone = {"polled_the_claimed_run", "non_terminal_poll_before_terminal",
                  "terminal_poll_reports_the_run_done",
                  "reported_poll_count_matches_observed",
                  "every_poll_spawned_by_the_tool"}

    def _polls(spec):
        return [r for r in spec["receipts"] if "--wait" in r["argv"]]

    def _polls_other_run(spec):
        for item in _polls(spec):
            item["argv"][-1] = redacted(OTHER_RUN_ID)

    return {
        "tool_claimed_a_run": [Mutation(
            "the tool's result names no run",
            lambda s: tool(s, "ask").update(run_id=None), polls_gone)],
        "polled_the_claimed_run": [Mutation(
            "every poll named another run", _polls_other_run,
            polls_gone - {"polled_the_claimed_run"})],
        "non_terminal_poll_before_terminal": [Mutation(
            "the first poll was already terminal",
            lambda s: _polls(s)[0].update(exit=0))],
        "terminal_poll_reports_the_run_done": [Mutation(
            "the last poll reports the run failed",
            lambda s: _polls(s)[-1]["run"].update(state="failed"))],
        "reported_poll_count_matches_observed": [Mutation(
            "the tool reports four polls, three were made",
            lambda s: tool(s, "ask").update(polls=4))],
        "every_poll_spawned_by_the_tool": [Mutation(
            "a poll process was not opencode's child",
            lambda s: process(s, POLL_PIDS[1]).update(ppid=1))],
    }


def completion_mutations() -> dict[str, list[Mutation]]:
    def _second_terminal(spec):
        parts = leg(spec, "ask")["tool_parts"]
        parts.append(copy.deepcopy(parts[0]))

    def _remove_answer(spec):
        spec["artifacts"] = {k: v for k, v in spec["artifacts"].items()
                             if not k.endswith(ANSWER)}

    def _touch(name, **kwargs):
        return lambda s: leg(s, "ask")["workspace_watch"]["touched"].update(
            {name: touched(**kwargs)})

    return {
        "exactly_one_terminal_result": [
            Mutation("a second terminal result", _second_terminal),
            Mutation("the terminal result is not ok",
                     lambda s: tool(s, "ask").update(ok=False, state="failed")),
        ],
        "result_names_the_claimed_run": [Mutation(
            "the result names another run's log",
            lambda s: tool(s, "ask").update(log=f"{WORKSPACE}/{PLAN_DIR}/other.log"))],
        "run_finished_done": [Mutation(
            "the run record ended failed",
            lambda s: run_record(s).update(state="failed", exit_code=1))],
        "per_model_outputs_exist": [
            Mutation("an answer file missing from the exported run", _remove_answer),
            Mutation("a model that did not finish",
                     lambda s: s["gate"]["run_records"][0]["models"][0][1].update(
                         state="failed")),
        ],
        "outputs_under_the_plan_dir": [Mutation(
            "the run dir is outside the plan's dir",
            lambda s: s["gate"]["run_records"][0].update(dir="elsewhere/ask/x"),
            {"per_model_outputs_exist"})],
        "no_stray_writes": [
            Mutation("a stray file created and removed outside the plan dir, seen "
                     "only by the watch", _touch("src/.auth.py.swp", present=False)),
            Mutation("a stray file outside the plan dir in the census delta",
                     lambda s: leg(s, "ask")["census_delta"].update(
                         {"src/auth.py": ["16d4f894", "00000000"]})),
            Mutation("the run's registry entry still there at the end",
                     _touch(registry_entry(RUN_ID), present=True)),
            Mutation("another run's registry entry",
                     _touch(registry_entry(OTHER_RUN_ID), present=False)),
            Mutation("the watch took a single census",
                     lambda s: leg(s, "ask")["workspace_watch"].update(censuses=1)),
        ],
        "opencode_run_exited_zero": [Mutation(
            "the opencode run exited 1", lambda s: leg(s, "ask").update(exit=1))],
    }


def cancellation_mutations() -> dict[str, list[Mutation]]:
    def _dispatch(**changes):
        return lambda s: abort(s)["dispatch"].update(**changes)

    def _abort_exit_two(spec):
        item = receipt_with(spec, "--abort")
        item["exit"] = 2
        item["result"].update(exit=2, state="nothing-to-do")

    return {
        "abort_sent_while_supervising_a_claimed_run": [
            Mutation("the run was already done when the abort was sent",
                     lambda s: abort(s)["run_records_at_abort"][0]["run"].update(
                         state="done")),
            Mutation("the session abort was never sent",
                     lambda s: abort(s).update(sent=False)),
        ],
        "abort_landed_mid_run": [Mutation(
            "no provider alive in the worker's group",
            lambda s: abort(s).update(provider_at_abort=None))],
        "session_abort_accepted": [Mutation(
            "opencode refused the abort",
            lambda s: abort(s).update(http_status=500, response="false"))],
        "session_reported_aborted": [Mutation(
            "the session never reported an abort",
            lambda s: (leg(s, "ask").update(error_events=[]),
                       s["messages"][0]["parts"][0]["state"].update(error=None)))],
        "tool_dispatched_abort_for_the_run": [
            Mutation("the abort was not the tool's (not opencode serve's child)",
                     lambda s: process(s, C_ABORT).update(ppid=CLIENT)),
            Mutation("two abort receipts for the run",
                     lambda s: s["receipts"].append(
                         dict(copy.deepcopy(receipt_with(s, "--abort")), seq=9)),
                     {"abort_stopped_a_live_run"}),
        ],
        "run_live_when_abort_dispatched": [
            Mutation("the run was already aborted when the tool dispatched",
                     lambda s: abort(s)["dispatch"]["run"].update(state="aborted")),
            Mutation("no worker process alive when the tool dispatched",
                     _dispatch(workers=[{"pid": C_WORKER, "state": "Z"}])),
            Mutation("the dispatch was never seen", _dispatch(seen=False)),
            Mutation("the dispatch seen came from another process",
                     lambda s: abort(s)["dispatch"]["abort"].update(ppid=CLIENT)),
        ],
        "abort_stopped_a_live_run": [
            Mutation("the dispatcher answered `nothing was running` (exit 2)",
                     _abort_exit_two),
            Mutation("the abort receipt exited 0 for some other mode",
                     lambda s: receipt_with(s, "--abort")["result"].update(
                         mode="ask")),
        ],
        "run_recorded_aborted": [Mutation(
            "the run recorded failed",
            lambda s: run_record(s).update(state="failed"))],
        "run_closed_by_the_dispatched_abort": [
            Mutation("the worker's own SIGTERM handler closed the run",
                     lambda s: run_record(s).update(
                         error="aborted_by_request (SIGTERM)")),
            Mutation("the run was closed out as already gone",
                     lambda s: run_record(s).update(
                         error="aborted_by_request (process already gone)")),
        ],
        "no_worker_group_process_survives": [Mutation(
            "a survivor in the worker's group",
            lambda s: process(s, C_WRAPPER).update(state="S", last_seen=FINAL,
                                                    last_alive=FINAL))],
        "no_worker_descendant_survives": [Mutation(
            "a survivor the worker started, in a session of its own",
            lambda s: s["processes"]["processes"].append(
                proc(300, 1, 300, "bash -c sleep 600", first_ppid=C_PROVIDER,
                     first_seen=3.2)))],
        "report_names_the_run_and_its_log": [
            Mutation("the report names another log",
                     lambda s: tap_report(s, log=f"{WORKSPACE}/other.log")),
            Mutation("the report says no abort was dispatched",
                     lambda s: tap_report(s, abort_dispatched=False)),
            Mutation("the report's message omits the run",
                     lambda s: tap_report(s, message=f"cancelled. Log: {LOG}.")),
        ],
    }


SCENARIO_MUTATIONS = {
    "init-status": init_status_mutations,
    "detach": detach_mutations,
    "supervision": supervision_mutations,
    "completion": completion_mutations,
    "cancellation": cancellation_mutations,
}


def mutations(scenario: str) -> dict[str, list[Mutation]]:
    return {**common_mutations(), **SCENARIO_MUTATIONS[scenario]()}


CASES = [pytest.param(scenario, check, index,
                      id=f"{scenario}:{check}:{mutation.name}")
         for scenario in SCENARIOS
         for check, cases in mutations(scenario).items()
         for index, mutation in enumerate(cases)]


# --------------------------------------------------------------------------- #
# The tests
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("scenario", SCENARIOS)
def test_good_bundle_passes_every_check(scenario, tmp_path):
    checks = grade(good_spec(scenario), tmp_path)
    assert checks and all(checks.values()), sorted(
        name for name, ok in checks.items() if not ok)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_check_has_a_mutation(scenario, tmp_path):
    """The guard: a check a grader emits with no negative case fails here."""
    emitted = set(grade(good_spec(scenario), tmp_path))
    covered = set(mutations(scenario))
    assert emitted - covered == set(), (
        f"{scenario}: checks with no mutation case: {sorted(emitted - covered)}")
    assert covered - emitted == set(), (
        f"{scenario}: mutation cases for checks the grader no longer emits: "
        f"{sorted(covered - emitted)}")
    assert all(mutations(scenario)[name] for name in covered)


@pytest.mark.parametrize("scenario,check,index", CASES)
def test_mutation_fails_its_check(scenario, check, index, tmp_path):
    mutation = mutations(scenario)[check][index]
    spec = good_spec(scenario)
    mutation.apply(spec)
    checks = grade(spec, tmp_path)
    failed = {name for name, ok in checks.items() if not ok}
    assert check in failed, f"{mutation.name!r} left {check} passing"
    assert failed - {check} <= mutation.collateral, (
        f"{mutation.name!r} also failed {sorted(failed - {check} - mutation.collateral)},"
        " which it does not declare")


def test_a_sample_inside_the_detach_handshake_is_not_opencodes_tree(tmp_path):
    """The false failure the round-2 review found: a 0.1 s sample landing in
    the intermediate's few-millisecond lifetime saw opencode on the worker's
    chain. Only samples from the worker's first sighting under its final parent
    count, so the row passes, and its observed ancestry says what was excused."""
    spec = handshake_race(good_spec("detach"))
    out = write_bundle(spec, tmp_path / row_id("detach"))
    checks, observed, _ = grade_bundle(out, "detach")
    assert checks["worker_outside_opencode_tree_in_every_sample"] is True
    assert all(checks.values()), sorted(name for name, ok in checks.items() if not ok)
    entries = {e["pid"]: e for e in observed["worker_ancestry"]}
    assert entries[WRAPPER]["handshake_excused"]["intermediate"] == INTERMEDIATE
    assert entries[WRAPPER]["handshake_excused"]["launcher"] == LAUNCHER
    assert {e["root"] for e in entries.values()} == {WRAPPER}
    assert all(e["final_chain"][-1] == 1 and not e["opencode_in_final_chain"]
               for e in entries.values())


def test_without_the_race_no_sample_is_excused(tmp_path):
    """The ordinary bundle (every worker first seen under its final parent)
    grades exactly as before: nothing excused, every sample counted."""
    out = write_bundle(good_spec("detach"), tmp_path / row_id("detach"))
    _checks, observed, _ = grade_bundle(out, "detach")
    assert all(e["handshake_excused"] is None for e in observed["worker_ancestry"])


def test_registry_entry_gone_by_the_end_is_not_stray(tmp_path):
    """The run's own registry entry is the one documented write outside the plan
    dir; the good completion bundle carries it, created and removed."""
    spec = good_spec("completion")
    assert leg(spec, "ask")["workspace_watch"]["touched"][registry_entry(RUN_ID)][
        "present_at_end"] is False
    assert grade(spec, tmp_path)["no_stray_writes"] is True


def test_grader_document_keeps_the_contract_shape(tmp_path):
    """`gate-grader.json` is what `release_gate.py check-row8` reads."""
    spec = good_spec("cancellation")
    out = write_bundle(spec, tmp_path / "bundle")
    result = result_from_bundle(out, "cancellation")
    checks, observed, method = grade_bundle(out, "cancellation")
    document = write_grader(Bundle(result), "cancellation", "grade_cancellation",
                            checks, observed, method)
    assert set(document) == {"scenario", "row", "grader", "pass", "route",
                             "installed_root", "dispatcher_skill_dir", "checks",
                             "abort_method"}
    assert document["pass"] is True and document["route"] == "package"
    assert document["dispatcher_skill_dir"] == PAYLOAD_SKILL_DIR
    assert json.loads((out / "gate-grader.json").read_text()) == document
    assert method and "POST /session/{id}/abort" in method
