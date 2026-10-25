"""Row 8 of plans/opencode-plugin-publish.md: the installed opencode package, offline.

The release gate's package container coverage. Five rows, one per scenario the
plan's step 0 makes the row-8 minimum, each its own `docker run`, its own run
record and its own grader:

====================  ======================================
scenario              row id
====================  ======================================
init and status       ``opencode-package-gate-init-status``
detached launch       ``opencode-package-gate-detach``
supervision           ``opencode-package-gate-supervision``
completion            ``opencode-package-gate-completion``
cancellation          ``opencode-package-gate-cancellation``
====================  ======================================

**Every row runs the INSTALLED package** — `route: package`, opencode's own
config declaring the payload Dockerfile §10b staged at
`/opt/mllm/opencode-plugin`, no file shim — **from a real `opencode run`**,
typed the way a user types it (`--command multi-llm -- <args>`), with the host's
model a loopback stub scripted to make ONE `multi_llm` call. No credential, no
route out (`--creds none` runs under `--network none`), $0. The in-container
driver is `container/opencode_package_gate.py`; what it measured lands in the
bundle as `opencode-gate.json` beside the process samples, the host's own
`--format json` streams, the uv wrapper's receipts and the exported plan dir.

**Each grader writes `gate-grader.json` into its bundle** — the shape
plans/opencode-plugin-publish.impl-contract.md fixes, which `release_gate.py
check-row8` reads — with every check it ran, and `gate-observed.json` beside it
with the values those checks read. The record is re-emitted (and its bundle
re-digested) at session end by `harness.GRADES`, so both files are inside the
digest `make e2e-verify-record` re-hashes. Every grader also checks that the
dispatcher ran from the installed payload's skill dir: a file-route shim, or the
checkout copy at `/opt/mllm/skill`, fails it.

Rows 2-4 run the same detached flow (`ask --quick`, the fan-out held so the run
outlives several `status --wait` polls) and each grades its own aspect of it; a
row is never credited with another row's run.

`container`-marked and NOT `live` or `live_provider`: `make e2e-container`'s
selection runs it, and nothing in it can spend.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from .container.docker_probe import requires_docker
from .container import harness

pytestmark = [pytest.mark.container, requires_docker()]

#: Where the image installs the package, and the payload's own skill tree — the
#: only place a row-8 dispatcher may run from.
INSTALLED_ROOT = "/opt/mllm/opencode-plugin"
PAYLOAD_SKILL_DIR = f"{INSTALLED_ROOT}/skills/multi-llm"
#: The checkout copy every other row dispatches from. Seeing it here means the
#: row graded the wrong object.
CHECKOUT_SKILL_DIR = "/opt/mllm/skill"

#: The fan-out the dispatcher runs: the provider lane's own opencode slug, read
#: from redirect.yaml rather than typed (a bare `stub-model` does not resolve).
_OPENCODE_ROW = next(r for r in harness.load_redirect()["clis"]
                     if r.get("cli") == "opencode")
FAN_OUT = f"opencode:{_OPENCODE_ROW.get('model_arg', 'stub/stub-model')}"
#: Answers the fan-out's `ask` prompt (and every other phase's) on the row's
#: main stub. The host's own turn is answered by a second, scripted stub.
CHAIN_SCENARIO = ("/opt/mllm/skill/tests/fixtures/e2e/scenarios/"
                  "container_full_chain.yaml")

SCENARIOS = ("init-status", "detach", "supervision", "completion",
             "cancellation")
ROW_TIMEOUT_S = {"init-status": 600, "detach": 600, "supervision": 600,
                 "completion": 600, "cancellation": 900}

#: The dispatcher's documented exit codes and the states they report
#: (`cli_common`'s EXIT_* and the surface's `dispatcher.states`).
EXIT_OK, EXIT_NOTHING_TO_DO = 0, 2
STATE_FOR_EXIT = {EXIT_OK: "done", EXIT_NOTHING_TO_DO: "nothing-to-do"}


def row_id(scenario: str) -> str:
    return f"opencode-package-gate-{scenario}"


@pytest.fixture(scope="module")
def posture():
    return harness.load_posture()


@pytest.fixture(scope="module")
def utc():
    import time

    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _drive(scenario: str, posture, utc) -> harness.RowResult:
    row = harness.Row(
        row_id=row_id(scenario),
        lane="host",
        host="opencode",
        creds="none",
        route="package",
        # A probe row: `promotable_fields` promotes nothing from it, because
        # the host's "turn" is a script, not a model's choice.
        probe=f"gate-{scenario}",
        spec=[FAN_OUT],
        runner=FAN_OUT,
        scenario=CHAIN_SCENARIO,
        row_timeout_s=ROW_TIMEOUT_S[scenario],
    )
    return harness.run_row(row, posture=posture, utc=utc)


@pytest.fixture(scope="module")
def init_status(posture, utc):
    return _drive("init-status", posture, utc)


@pytest.fixture(scope="module")
def detach(posture, utc):
    return _drive("detach", posture, utc)


@pytest.fixture(scope="module")
def supervision(posture, utc):
    return _drive("supervision", posture, utc)


@pytest.fixture(scope="module")
def completion(posture, utc):
    return _drive("completion", posture, utc)


@pytest.fixture(scope="module")
def cancellation(posture, utc):
    return _drive("cancellation", posture, utc)


# --------------------------------------------------------------------------- #
# Reading the bundle
# --------------------------------------------------------------------------- #

def _json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def _jsonl(path: Path) -> list[dict]:
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def redacted(value: str) -> str:
    """How the uv wrapper's receipts spell a run id (`wrapper._redact`)."""
    return f"<redacted:sha256:{hashlib.sha256(value.encode()).hexdigest()[:8]}>"


def _option(argv: list, name: str):
    if name in argv:
        index = argv.index(name)
        if index + 1 < len(argv):
            return argv[index + 1]
    return None


class Bundle:
    """One row's evidence, read back from `runs/<utc>/<row-id>/`."""

    def __init__(self, result: harness.RowResult):
        self.result = result
        self.out = result.out_dir
        self.gate = _json(self.out / "opencode-gate.json") or {}
        self.receipts = sorted(result.receipts, key=lambda r: r.get("seq") or 0)
        self.processes = (_json(self.out / "gate-processes.json") or {}).get(
            "processes") or []
        self.final_sample = max((p.get("last_seen") or 0 for p in self.processes),
                                default=0)

    # -- the legs and the tool's results ----------------------------------- #
    def legs(self) -> list[dict]:
        return list(self.gate.get("legs") or [])

    def leg(self, label: str) -> dict:
        return next((leg for leg in self.legs() if leg.get("label") == label), {})

    def stub_entries(self, leg: dict) -> list[dict]:
        return _jsonl(self.out / str(leg.get("stub_log") or "-"))

    def scripted_calls(self, leg: dict) -> list[dict]:
        return [e["scripted"] for e in self.stub_entries(leg)
                if (e.get("scripted") or {}).get("phase") == "call"]

    def tap(self) -> list[dict]:
        return _jsonl(self.out / str(self.gate.get("tap") or "-"))

    # -- the dispatcher ---------------------------------------------------- #
    def dispatcher_dirs(self) -> list[str]:
        """Every `--project` a dispatcher invocation of this row named."""
        return sorted({str(_option(r.get("argv") or [], "--project"))
                       for r in self.receipts})

    def dispatcher_scripts(self) -> list[str]:
        scripts = set()
        for receipt in self.receipts:
            argv = receipt.get("argv") or []
            if "python" in argv and argv.index("python") + 1 < len(argv):
                scripts.add(str(argv[argv.index("python") + 1]))
        return sorted(scripts)

    def receipts_for(self, *words: str) -> list[dict]:
        return [r for r in self.receipts
                if all(word in (r.get("argv") or []) for word in words)]

    # -- the run ----------------------------------------------------------- #
    def run_record(self, run_id: str | None) -> dict:
        for record in self.gate.get("run_records") or []:
            if run_id and (record.get("run") or {}).get("run_id") == run_id:
                return record
        return {}

    # -- the process table ------------------------------------------------- #
    def alive_at_end(self, process: dict) -> bool:
        return (process.get("last_seen") == self.final_sample
                and process.get("state") != "Z")

    def ancestors(self, pid: int, first: bool = False) -> list[int]:
        """The ppid chain the samples observed for `pid`, nearest first.

        `first=True` follows the parent each process was first seen under, so
        an orphan reparented to PID 1 when its parent died still traces back to
        the tree that started it — the question "did anything the worker
        started survive" needs that, and the latest ppid would hide it.
        """
        key = "first_ppid" if first else "ppid"
        parent = {p["pid"]: p.get(key, p["ppid"]) for p in self.processes}
        chain, seen = [], set()
        current = parent.get(pid)
        while current and current not in seen:
            seen.add(current)
            chain.append(current)
            current = parent.get(current)
        return chain

    def workers(self, run_id: str | None) -> list[dict]:
        return [p for p in self.processes if run_id
                and "--internal-worker" in p.get("args", "")
                and f"--internal-run-id {run_id}" in p.get("args", "")]


# --------------------------------------------------------------------------- #
# The checks every grader runs
# --------------------------------------------------------------------------- #

def common_checks(bundle: Bundle) -> tuple[dict, dict]:
    """What all five rows must show, whatever their scenario.

    The dispatcher check is the one the plan names for every grader: each
    dispatcher invocation of the row ran `uv run --project <payload skill dir>`
    on `<payload skill dir>/multi_llm.py`. A file shim dispatches from wherever
    `install-host` pointed it, and every other container row from the checkout
    copy, so either fails it.
    """
    result = bundle.result
    gate = bundle.gate
    debug = gate.get("debug_config") or {}
    origins = [str(o.get("spec")) for o in debug.get("plugin_origins") or []]
    dirs = bundle.dispatcher_dirs()
    scripts = bundle.dispatcher_scripts()
    tool_argvs = [part["result"].get("argv") for leg in bundle.legs()
                  for part in leg.get("tool_parts") or []
                  if isinstance(part.get("result"), dict)
                  and part["result"].get("argv")]
    tool_dirs = sorted({str(_option(argv, "--project")) for argv in tool_argvs})
    main_stub = result.stub_log
    host_stub = [e for leg in bundle.legs() for e in bundle.stub_entries(leg)]
    observed = {
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "cap_reason": result.cap_reason,
        "wall_clock_s": result.wall_clock_s,
        "plugin": (gate.get("wiring") or {}).get("plugin"),
        "plugin_origins": origins,
        "file_shims": gate.get("file_shims"),
        "receipts": len(bundle.receipts),
        "dispatcher_dirs": dirs,
        "dispatcher_scripts": scripts,
        "tool_argv_project_dirs": tool_dirs,
        "opencode_version": gate.get("opencode_version"),
        "network_inside": result.container_posture.get("network"),
        "posture_mismatches": result.host_posture.get("mismatches"),
    }
    checks = {
        "row_completed": (result.exit_code == 0 and not result.timed_out
                          and bool(gate)),
        # Read off the container's own inside check, not the row's request:
        # `--creds none` asserted no credential variable before anything ran,
        # and this is the route table it then measured.
        "offline_no_credentials": (
            result.row.creds == "none"
            and (result.container_posture.get("network") or {}).get("declared")
            == "none"
            and (result.container_posture.get("network") or {}).get(
                "default_route") is False
            and not (result.host_posture.get("mismatches") or [])),
        "package_declared": (gate.get("wiring") or {}).get("plugin")
        == [INSTALLED_ROOT],
        "plugin_loaded_from_installed_root": f"file://{INSTALLED_ROOT}" in origins,
        "no_file_shim": gate.get("file_shims") == [],
        "dispatcher_ran_from_installed_payload": (
            bool(bundle.receipts) and dirs == [PAYLOAD_SKILL_DIR]
            and scripts == [f"{PAYLOAD_SKILL_DIR}/multi_llm.py"]
            and tool_dirs in ([], [PAYLOAD_SKILL_DIR])),
        "checkout_skill_never_dispatched": not any(
            CHECKOUT_SKILL_DIR in json.dumps(r.get("argv") or [])
            for r in bundle.receipts),
        "every_leg_made_one_scripted_call": bool(bundle.legs()) and all(
            len(bundle.scripted_calls(leg)) == 1 for leg in bundle.legs()),
        "stub_answered_every_request": not any(
            e.get("unhandled") or e.get("unmatched")
            for e in [*main_stub, *host_stub]),
    }
    return checks, observed


def dispatcher_skill_dir(bundle: Bundle) -> str | None:
    """The one skill dir this row's dispatcher ran from, or `None` if not one."""
    dirs = bundle.dispatcher_dirs()
    return dirs[0] if len(dirs) == 1 else None


def write_grader(bundle: Bundle, scenario: str, grader: str, checks: dict,
                 observed: dict, abort_method: str | None = None) -> dict:
    """`gate-grader.json` (the contract's shape) and `gate-observed.json`."""
    document = {
        "scenario": scenario,
        "row": row_id(scenario),
        "grader": grader,
        "pass": bool(checks) and all(checks.values()),
        "route": bundle.result.row.route,
        "installed_root": INSTALLED_ROOT,
        "dispatcher_skill_dir": dispatcher_skill_dir(bundle),
        "checks": {name: bool(value) for name, value in checks.items()},
    }
    if abort_method is not None:
        document["abort_method"] = abort_method
    (bundle.out / "gate-grader.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (bundle.out / "gate-observed.json").write_text(
        json.dumps(observed, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    return document


def assert_graded(document: dict, observed: dict) -> None:
    failed = sorted(name for name, ok in document["checks"].items() if not ok)
    assert document["pass"], (
        f"{document['row']}: {document['grader']} failed {failed}\n"
        f"observed: {json.dumps(observed, indent=2, sort_keys=True, default=str)[:6000]}")


# --------------------------------------------------------------------------- #
# Scenario 1 — init and status
# --------------------------------------------------------------------------- #

def _tool_result(leg: dict) -> dict:
    parts = leg.get("tool_parts") or []
    return (parts[0].get("result") or {}) if len(parts) == 1 else {}


def grade_init_status(result: harness.RowResult) -> tuple[dict, dict, None]:
    bundle = Bundle(result)
    checks, observed = common_checks(bundle)
    debug = bundle.gate.get("debug_config") or {}
    init, status = bundle.leg("init"), bundle.leg("status")
    init_result, status_result = _tool_result(init), _tool_result(status)
    init_added = sorted(name for name, (before, after)
                        in (init.get("census_delta") or {}).items()
                        if before is None and after)
    facts = {fact for e in bundle.stub_entries(init) for fact in e.get("facts") or []}
    status_payload = status_result.get("result") or {}
    observed.update({
        "command_keys": debug.get("command_keys"),
        "command_skill_dir": debug.get("command_skill_dir"),
        "template_matches_payload": debug.get("template_matches_payload"),
        "description_matches_payload": debug.get("description_matches_payload"),
        "init_facts": sorted(facts),
        "workspace_set_aside": bundle.gate.get("workspace_set_aside"),
        "init": {k: init_result.get(k) for k in ("ok", "state", "exit", "error",
                                                  "artifacts")},
        "init_mode": (init_result.get("result") or {}).get("mode"),
        "init_added": init_added,
        "status": {k: status_result.get(k) for k in ("ok", "state", "exit",
                                                      "error")},
        "status_payload": status_payload,
        "status_census_delta": status.get("census_delta"),
        "legs": [{k: leg.get(k) for k in ("label", "exit", "timed_out",
                                          "wall_clock_s", "error_events")}
                 for leg in bundle.legs()],
    })
    command_dir = debug.get("command_skill_dir")
    checks.update({
        "command_registered": "multi-llm" in (debug.get("command_keys") or []),
        "command_resolves_inside_payload": (
            bool(debug.get("template_matches_payload"))
            and bool(debug.get("description_matches_payload"))
            and command_dir == PAYLOAD_SKILL_DIR
            and str(command_dir).startswith(INSTALLED_ROOT + "/")),
        "command_expanded_into_the_turn": "multi_llm_command" in facts,
        "opencode_runs_exited_zero": all(
            leg.get("exit") == 0 and not leg.get("timed_out")
            for leg in (init, status)),
        "init_returned_success": (
            init_result.get("ok") is True and init_result.get("exit") == EXIT_OK
            and init_result.get("state") == "done"
            and (init_result.get("result") or {}).get("mode") == "init"),
        "init_created_project_files": (
            ".multi-llm/providers.yaml" in init_added
            and all(name.startswith(".multi-llm/") for name in init_added)),
        # The plan has no run and no workflow state, and for that `status`
        # documents exit 2, `nothing-to-do` — never 0, which a poll loop would
        # read as finished (`run_status.exit_code_for`). The tool must hand it
        # back as a terminal, ok answer rather than as an error.
        "status_returned_documented_result": (
            status_result.get("ok") is True
            and status_result.get("exit") == EXIT_NOTHING_TO_DO
            and status_result.get("state") == STATE_FOR_EXIT[EXIT_NOTHING_TO_DO]
            and status_payload.get("mode") == "status"
            and status_payload.get("exit") == EXIT_NOTHING_TO_DO
            and status_payload.get("state") == STATE_FOR_EXIT[EXIT_NOTHING_TO_DO]),
        "status_wrote_nothing": not status.get("census_delta"),
    })
    return checks, observed, None


# --------------------------------------------------------------------------- #
# Scenarios 2-4 — the detached flow
# --------------------------------------------------------------------------- #

def _claim(bundle: Bundle) -> dict:
    """The detached call's readings, shared by the three rows that grade it."""
    leg = bundle.leg("ask")
    part = (leg.get("tool_parts") or [{}])[0] if leg.get("tool_parts") else {}
    terminal = part.get("result") or {}
    launches = [r for r in bundle.receipts_for("ask", "--detach")
                if "--internal-worker" not in (r.get("argv") or [])]
    launch = launches[0] if len(launches) == 1 else {}
    banner = launch.get("banner") or {}
    run_id = terminal.get("run_id")
    return {"leg": leg, "part": part, "terminal": terminal, "launches": launches,
            "launch": launch, "banner": banner, "run_id": run_id,
            "record": bundle.run_record(run_id)}


def grade_detach(result: harness.RowResult) -> tuple[dict, dict, None]:
    bundle = Bundle(result)
    checks, observed = common_checks(bundle)
    claim = _claim(bundle)
    terminal, banner, run_id = claim["terminal"], claim["banner"], claim["run_id"]
    record = claim["record"]
    run = record.get("run") or {}
    workers = bundle.workers(run_id)
    host_pid = bundle.gate.get("host_pid")
    host = next((p for p in bundle.processes if p.get("pid") == host_pid), {})
    scripted = bundle.scripted_calls(claim["leg"])
    observed.update({
        "scripted_extra_flags": [(c.get("arguments") or {}).get("extra_flags")
                                 for c in scripted],
        "tool_argv": terminal.get("argv"),
        "launch_receipts": [r.get("argv") for r in claim["launches"]],
        "banner": banner,
        "claimed_run_id": run_id,
        "claimed_log": terminal.get("log"),
        "run_record": run,
        "host": {k: host.get(k) for k in ("pid", "pgid", "sid", "args")},
        "workers": [{k: w.get(k) for k in ("pid", "ppid", "first_ppid", "pgid",
                                            "sid", "state", "args", "first_seen",
                                            "last_alive")}
                    for w in workers],
        "worker_ancestry": {w["pid"]: bundle.ancestors(w["pid"]) for w in workers},
        "ps_snapshots": ((_json(bundle.out / "gate-processes.json") or {})
                         .get("snapshots")),
    })
    checks.update({
        "tool_composed_detach": (
            "--detach" in (terminal.get("argv") or [])
            and all("--detach" not in ((c.get("arguments") or {}).get("extra_flags")
                                       or []) for c in scripted)),
        "one_detached_launch": len(claim["launches"]) == 1,
        "banner_names_a_run": bool(banner.get("run_id")),
        "tool_claimed_the_banner_run": (
            bool(run_id) and run_id == banner.get("run_id")
            and terminal.get("log") == banner.get("log")),
        "run_record_is_the_claimed_detached_run": (
            run.get("run_id") == run_id and run.get("detached") is True
            and run.get("route") == "package"
            and (claim["launch"].get("run") or {}).get("run_id") == run_id),
        "worker_observed_running": any(w.get("last_alive") is not None
                                       for w in workers)
        and "worker-live" in (observed["ps_snapshots"] or {}),
        "worker_in_its_own_process_group": bool(workers) and bool(host) and all(
            w.get("pgid") == run.get("pgid") and w.get("pgid") != host.get("pgid")
            and w.get("sid") == w.get("pgid") for w in workers),
        # Never parented by opencode, at first sight or later, and not inside
        # its tree at the end: the plugin's own children (the launch, the
        # polls) ARE opencode's, which is exactly what the worker must not be.
        "worker_not_a_child_of_opencode": bool(workers) and bool(host_pid) and all(
            w.get("ppid") != host_pid and w.get("first_ppid") != host_pid
            and host_pid not in bundle.ancestors(w["pid"])
            for w in workers),
    })
    return checks, observed, None


def _poll_receipts(bundle: Bundle, run_id: str | None) -> list[dict]:
    if not run_id:
        return []
    return [r for r in bundle.receipts_for("status", "--wait", "--run-id")
            if _option(r.get("argv") or [], "--run-id") == redacted(run_id)]


def grade_supervision(result: harness.RowResult) -> tuple[dict, dict, None]:
    bundle = Bundle(result)
    checks, observed = common_checks(bundle)
    claim = _claim(bundle)
    terminal, run_id = claim["terminal"], claim["run_id"]
    polls = _poll_receipts(bundle, run_id)
    exits = [p.get("exit") for p in polls]
    host_pid = bundle.gate.get("host_pid")
    sampled = [p for p in bundle.processes if run_id
               and p.get("ppid") == host_pid and "/opt/mllm/bin/uv" in p.get("args", "")
               and "multi_llm.py status" in p.get("args", "")
               and "--wait" in p.get("args", "")
               and f"--run-id {run_id}" in p.get("args", "")]
    observed.update({
        "claimed_run_id": run_id,
        "reported_polls": terminal.get("polls"),
        "poll_receipts": [{"seq": p.get("seq"), "exit": p.get("exit"),
                           "state": (p.get("result") or {}).get("state"),
                           "run": p.get("run")} for p in polls],
        "sampled_poll_processes": [{k: p.get(k) for k in ("pid", "ppid", "first_seen",
                                                           "last_seen")}
                                   for p in sampled],
    })
    checks.update({
        "tool_claimed_a_run": bool(run_id),
        "polled_the_claimed_run": bool(polls),
        "non_terminal_poll_before_terminal": (
            len(exits) >= 2 and exits[-1] == EXIT_OK
            and all(code == EXIT_NOTHING_TO_DO for code in exits[:-1])),
        "terminal_poll_reports_the_run_done": bool(polls) and (
            (polls[-1].get("run") or {}).get("run_id") == run_id
            and (polls[-1].get("run") or {}).get("state") == "done"),
        "reported_poll_count_matches_observed": (
            isinstance(terminal.get("polls"), int)
            and terminal.get("polls") == len(polls)),
        "every_poll_spawned_by_the_tool": bool(polls) and len(sampled) == len(polls),
    })
    return checks, observed, None


def grade_completion(result: harness.RowResult) -> tuple[dict, dict, None]:
    bundle = Bundle(result)
    checks, observed = common_checks(bundle)
    claim = _claim(bundle)
    leg, terminal, run_id = claim["leg"], claim["terminal"], claim["run_id"]
    record = claim["record"]
    run = record.get("run") or {}
    plan_dir = str(bundle.gate.get("plan_output_dir") or "")
    models = dict(record.get("models") or [])
    files = record.get("files") or []
    answers = sorted(name for name in files
                     if name.startswith("answer_") and name.endswith(".md"))
    run_dir = str(record.get("dir") or "")
    exported = bundle.out / "artifacts" / Path(plan_dir).name / Path(run_dir).relative_to(
        plan_dir) if run_dir.startswith(plan_dir + "/") else None
    delta = sorted((leg.get("census_delta") or {}))
    stray = [name for name in delta if not name.startswith(plan_dir + "/")]
    observed.update({
        "tool_parts": [{k: p.get(k) for k in ("status", "error", "title")}
                       for p in leg.get("tool_parts") or []],
        "terminal": {k: terminal.get(k) for k in ("ok", "state", "exit", "run_id",
                                                   "log", "polls", "launch_state",
                                                   "artifacts")},
        "run_record": run,
        "models": models,
        "answer_files": answers,
        "run_dir": run_dir,
        "plan_output_dir": plan_dir,
        "exported_run_dir": str(exported) if exported else None,
        "workspace_delta": delta,
        "stray_writes": stray,
        "opencode_exit": leg.get("exit"),
    })
    checks.update({
        "exactly_one_terminal_result": (
            len(leg.get("tool_parts") or []) == 1
            and claim["part"].get("status") == "completed"
            and isinstance(terminal, dict) and terminal.get("ok") is True
            and terminal.get("state") == "done" and terminal.get("exit") == EXIT_OK),
        "result_names_the_claimed_run": (
            bool(run_id) and run_id == claim["banner"].get("run_id")
            and terminal.get("log") == run.get("log_path")),
        "run_finished_done": run.get("state") == "done" and run.get("exit_code") == 0,
        "per_model_outputs_exist": (
            bool(models) and all((m or {}).get("state") == "done"
                                 for m in models.values())
            and len(answers) == len(models) and "answers.md" in files
            and exported is not None
            and all((exported / name).is_file() and (exported / name).stat().st_size
                    for name in [*answers, "answers.md"])),
        "outputs_under_the_plan_dir": (
            bool(plan_dir) and run_dir.startswith(plan_dir + "/")),
        "no_stray_writes": bool(delta) and not stray,
        "opencode_run_exited_zero": leg.get("exit") == 0 and not leg.get("timed_out"),
    })
    return checks, observed, None


# --------------------------------------------------------------------------- #
# Scenario 5 — cancellation
# --------------------------------------------------------------------------- #

def grade_cancellation(result: harness.RowResult) -> tuple[dict, dict, str | None]:
    bundle = Bundle(result)
    checks, observed = common_checks(bundle)
    gate = bundle.gate
    abort = gate.get("abort") or {}
    leg = bundle.leg("ask")
    poll = abort.get("supervising_poll") or {}
    run_id = poll.get("run_id")
    launches = [r for r in bundle.receipts_for("ask", "--detach")
                if "--internal-worker" not in (r.get("argv") or [])]
    banner = (launches[0].get("banner") or {}) if len(launches) == 1 else {}
    record = bundle.run_record(run_id)
    run = record.get("run") or {}
    at_abort = next((r for r in abort.get("run_records_at_abort") or []
                     if (r.get("run") or {}).get("run_id") == run_id), {})
    serve_pid = gate.get("serve_pid")
    abort_receipts = [r for r in bundle.receipts_for("status", "--abort", "--run-id")
                      if _option(r.get("argv") or [], "--run-id")
                      == (redacted(run_id) if run_id else None)]
    abort_processes = [p for p in bundle.processes if run_id
                       and "multi_llm.py status" in p.get("args", "")
                       and "--abort" in p.get("args", "")
                       and f"--run-id {run_id}" in p.get("args", "")
                       and "/opt/mllm/bin/uv" in p.get("args", "")]
    group = run.get("pgid")
    in_group = [p for p in bundle.processes if group and p.get("pgid") == group]
    workers = bundle.workers(run_id)
    descendants = [p for p in bundle.processes if any(
        w["pid"] in bundle.ancestors(p["pid"], first=True) for w in workers)]
    provider = abort.get("provider_at_abort") or {}
    still = abort.get("supervising_at_abort") or {}
    taps = [t for t in bundle.tap() if t.get("tool") == "multi_llm"]
    report = {}
    if len(taps) == 1:
        try:
            report = json.loads(taps[0].get("output") or "")
        except ValueError:
            report = {}
    message = str(report.get("message") or "")
    log_path = run.get("log_path")
    events_errors = leg.get("error_events") or []
    messages = _json(bundle.out / "gate-session-messages.json") or []
    tool_states = [p.get("state") for m in messages if isinstance(m, dict)
                   for p in m.get("parts") or []
                   if p.get("type") == "tool" and p.get("tool") == "multi_llm"]
    method = (f"{abort.get('method')} with id {abort.get('session_id')} — "
              f"{abort.get('via')}") if abort.get("sent") else None
    observed.update({
        "abort": {k: abort.get(k) for k in ("method", "via", "sent", "session_id",
                                             "sessions_listed", "http_status",
                                             "response", "sent_at_s", "tap_seen")},
        "supervising_poll": poll,
        "supervising_at_abort": still,
        "provider_at_abort": provider,
        "banner": banner,
        "run_at_abort": at_abort.get("run"),
        "run_after": run,
        "abort_receipts": [{"exit": r.get("exit"), "result": r.get("result"),
                            "argv": r.get("argv")} for r in abort_receipts],
        "abort_processes": [{k: p.get(k) for k in ("pid", "ppid", "args")}
                            for p in abort_processes],
        "serve_pid": serve_pid,
        "worker_group": group,
        "group_alive_at_end": [p for p in in_group if bundle.alive_at_end(p)],
        "descendants_alive_at_end": [p for p in descendants if bundle.alive_at_end(p)],
        "report": {k: report.get(k) for k in ("state", "ok", "reason", "run_id", "log",
                                               "abort_dispatched", "message")},
        "client_exit": leg.get("exit"),
        "client_error_events": events_errors,
        "session_tool_states": tool_states,
    })
    checks.update({
        "abort_sent_while_supervising_a_claimed_run": (
            abort.get("sent") is True and bool(run_id)
            and run_id == banner.get("run_id")
            and (at_abort.get("run") or {}).get("state") == "running"),
        # Mid-run, concretely: the fan-out's provider turn alive in the worker's
        # group and the tool's poll of the claimed run still in flight.
        "abort_landed_mid_run": (
            bool(group) and provider.get("pgid") == group
            and still.get("run_id") == run_id),
        "session_abort_accepted": (abort.get("http_status") == 200
                                   and str(abort.get("response")).strip() == "true"),
        "session_reported_aborted": any(
            (e or {}).get("name") == "MessageAbortedError" for e in events_errors)
        or any((s or {}).get("error") == "Tool execution aborted"
               for s in tool_states),
        "tool_dispatched_abort_for_the_run": (
            len(abort_receipts) == 1
            and abort_receipts[0].get("exit") in STATE_FOR_EXIT
            and bool(abort_processes) and bool(serve_pid)
            and all(p.get("ppid") == serve_pid for p in abort_processes)),
        "run_recorded_aborted": run.get("state") == "aborted",
        "no_worker_group_process_survives": bool(group) and bool(in_group) and not any(
            bundle.alive_at_end(p) for p in in_group),
        "no_worker_descendant_survives": bool(workers) and not any(
            bundle.alive_at_end(p) for p in descendants),
        "report_names_the_run_and_its_log": (
            len(taps) == 1 and report.get("run_id") == run_id
            and bool(log_path) and report.get("log") == log_path
            and report.get("abort_dispatched") is True
            and bool(run_id) and run_id in message and log_path in message),
    })
    return checks, observed, method


# --------------------------------------------------------------------------- #
# The five rows
# --------------------------------------------------------------------------- #

def _grade(result, scenario, grader) -> None:
    checks, observed, abort_method = grader(result)
    bundle = Bundle(result)
    document = write_grader(bundle, scenario, grader.__name__, checks, observed,
                            abort_method)
    assert_graded(document, observed)


def test_init_and_status(init_status):
    """Scenario 1: `/multi-llm` is the payload's, `init` creates its files,
    `status` answers its documented result."""
    _grade(init_status, "init-status", grade_init_status)


def test_detached_launch(detach):
    """Scenario 2: the tool composes `--detach`, claims the banner's run, and
    the worker runs in its own process group, outside opencode's tree."""
    _grade(detach, "detach", grade_detach)


def test_supervision(supervision):
    """Scenario 3: `status --wait` on the claimed run, non-terminal polls first,
    and the reported poll count equals the polls observed."""
    _grade(supervision, "supervision", grade_supervision)


def test_completion(completion):
    """Scenario 4: one terminal result naming the run; its per-model outputs
    under the plan's dir; nothing written outside it."""
    _grade(completion, "completion", grade_completion)


def test_cancellation(cancellation):
    """Scenario 5: abort the session mid-run; the tool aborts the run it
    claimed, its group is gone, and its report names the run and its log."""
    _grade(cancellation, "cancellation", grade_cancellation)
