"""Row 8's five graders, as pure functions over a bundle on disk.

The host-side half of the `opencode-package-gate-*` rows
(`tests/test_container_opencode_package_gate.py` drives them;
`opencode_package_gate.py` is the in-container driver that writes the evidence
read here). Kept out of the test module so the graders import without Docker:
that module's `requires_docker()` probes the daemon at import, and
`tests/test_opencode_package_gate_graders.py` runs every check below against
synthetic bundles in the plain suite, one targeted mutation per check, so a
check that turned vacuous fails there rather than staying green on a correct
product.

Each grader takes a `harness.RowResult` — the row, its exit facts and its
bundle directory — and returns `(checks, observed, abort_method)`. A live row
passes the RowResult `harness.run_row` returned; :func:`result_from_bundle`
builds the same object from a bundle already on disk. Nothing here writes,
except :func:`write_grader`, which the live rows call.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import harness

#: Where the image installs the package, and the payload's own skill tree — the
#: only place a row-8 dispatcher may run from.
INSTALLED_ROOT = "/opt/mllm/opencode-plugin"
PAYLOAD_SKILL_DIR = f"{INSTALLED_ROOT}/skills/multi-llm"
#: The checkout copy every other row dispatches from. Seeing it here means the
#: row graded the wrong object.
CHECKOUT_SKILL_DIR = "/opt/mllm/skill"
#: The host binary `opencode run` and `opencode serve` are, as `/proc` names it.
#: The fan-out's own opencode goes through `/opt/mllm/bin` and is the worker's
#: descendant, never its ancestor.
HOST_OPENCODE = "/opt/mllm/real/opencode"
#: How long the detach handshake may keep a worker under its first parent. The
#: double spawn (`utils/process_control.py`, `_spawn_detached_posix`) forks an
#: intermediate in a new session, which forks the worker, writes its pid to the
#: launcher and exits: milliseconds, during which the worker's ppid chain still
#: runs through the launcher into opencode's tree. A 0.1 s sample can land
#: there. A "first parent" alive a second past the worker's first sighting is
#: no such intermediate, whatever its argv says.
DETACH_HANDSHAKE_S = 1.0

SCENARIOS = ("init-status", "detach", "supervision", "completion",
             "cancellation")
ROW_TIMEOUT_S = {"init-status": 600, "detach": 600, "supervision": 600,
                 "completion": 600, "cancellation": 900}

#: The dispatcher's documented exit codes and the states they report
#: (`cli_common`'s EXIT_* and the surface's `dispatcher.states`).
EXIT_OK, EXIT_NOTHING_TO_DO, EXIT_NEEDS_HUMAN = 0, 2, 3
STATE_FOR_EXIT = {EXIT_OK: "done", EXIT_NOTHING_TO_DO: "nothing-to-do"}
#: `status_file.TERMINAL_RUN_STATES`.
TERMINAL_RUN_STATES = ("done", "failed", "aborted")
#: What `detach.abort_run` writes into the record of a live run it stopped.
#: The worker's own SIGTERM handler writes `aborted_by_request (SIGTERM)`, and a
#: run found already dead is closed out as `aborted_by_request (process already
#: gone)`, so only this exact value says the dispatched abort did it.
ABORTED_BY_DISPATCH = "aborted_by_request"


def row_id(scenario: str) -> str:
    return f"opencode-package-gate-{scenario}"


def registry_entry(run_id: str) -> str:
    """The one file a detached run writes outside its plan's dir.

    Its entry in the project's live-run registry (`utils/run_registry.py`):
    written as the worker takes its lock, removed on any terminal state.
    """
    return f".multi-llm/runs/{run_id}.json"


def gate_row(which: str, /, **fields) -> harness.Row:
    """The row the live test drives and a bundle on disk is graded as.

    `which` is the gate scenario; `fields` are `harness.Row`'s own, among them
    `scenario`, the stub's scenario file.
    """
    return harness.Row(
        row_id=row_id(which),
        lane="host",
        host="opencode",
        creds="none",
        route="package",
        # A probe row: `promotable_fields` promotes nothing from it, because
        # the host's "turn" is a script, not a model's choice.
        probe=f"gate-{which}",
        row_timeout_s=ROW_TIMEOUT_S[which],
        **fields,
    )


def result_from_bundle(out_dir: Path, scenario: str) -> harness.RowResult:
    """A `RowResult` read back from `runs/<utc>/<row-id>/` alone."""
    facts = _json(Path(out_dir) / "row-result.json") or {}
    return harness.RowResult(
        row=gate_row(scenario),
        exit_code=facts.get("exit_code"),
        wall_clock_s=float(facts.get("wall_clock_s") or 0.0),
        out_dir=Path(out_dir),
        timed_out=bool(facts.get("timed_out")),
        cap_reason=facts.get("cap_reason"),
    )


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

    def ancestors(self, pid: int) -> list[int]:
        """The chain of parents each process was FIRST seen under, nearest first.

        An orphan reparented to PID 1 when its parent died still traces back to
        the tree that started it — the question "did anything the worker
        started survive" needs that, and the latest ppid would hide it.
        """
        parent = {p["pid"]: p.get("first_ppid", p["ppid"]) for p in self.processes}
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

    def host_opencode_pids(self) -> set[int]:
        pids = {p["pid"] for p in self.processes
                if p.get("args", "").startswith(HOST_OPENCODE + " ")}
        for key in ("host_pid", "serve_pid"):
            if self.gate.get(key):
                pids.add(self.gate[key])
        return pids


def worker_ancestry(bundle: "Bundle", workers: list[dict]) -> list[dict]:
    """Per worker process: whether opencode was on its ppid chain in a sample
    that counts, measured from the samples' own record.

    What counts is every sample from the worker tree's first sighting under its
    FINAL parent onward, i.e. after the detach handshake. The driver records,
    per worker, the union of every sample's chain (`ancestors_seen`) and the
    chain at its first sighting (`first_chain`), not each sample's, so the
    pre-handshake samples are taken out the only way that record allows:

    * the tree's root is the worker process whose parent (as first seen) is no
      worker; its descendants' chains run through it;
    * when the root was first seen under a parent it later lost, and that
      parent was the double spawn's intermediate — a fork of its own parent
      (the same argv), alive no longer than `DETACH_HANDSHAKE_S` past the
      root's first sighting and gone by the end — the root's `first_chain` is
      the pre-handshake chain, and its pids are excused from every worker's
      union (they are the launcher's tree: opencode, the plugin's uv, the
      launcher, the intermediate);
    * whatever remains of each union must hold no opencode pid, and so must
      each worker's chain under its final parent, read from the table (a
      final parent inside opencode's tree, a subreaper say, would otherwise
      hide behind the excused pids).

    Otherwise no sample is excused, which is the rule unchanged: a worker that
    was ever a plain descendant of opencode — first seen there, reparented
    later or never — still fails. Only the recorded fields are read, so older
    bundles grade the same way they did, unless the handshake race is in them.
    """
    worker_pids = {w["pid"] for w in workers}
    table = {p["pid"]: p for p in bundle.processes}
    opencode = bundle.host_opencode_pids()

    def first_parent(process: dict):
        return process.get("first_ppid", process.get("ppid"))

    def latest_chain(pid) -> list:
        chain, seen = [], set()
        while pid and pid not in seen:
            seen.add(pid)
            chain.append(pid)
            pid = (table.get(pid) or {}).get("ppid")
        return chain

    results = []
    for worker in workers:
        root, seen = worker, {worker["pid"]}
        while first_parent(root) in worker_pids and first_parent(root) not in seen:
            seen.add(first_parent(root))
            root = table[first_parent(root)]
        excused: set = set()
        handshake = None
        if first_parent(root) != root.get("ppid"):
            intermediate = table.get(first_parent(root)) or {}
            launcher = table.get(first_parent(intermediate)) if intermediate else None
            last_alive = intermediate.get("last_alive")
            if (launcher is not None and intermediate.get("args")
                    and intermediate.get("args") == launcher.get("args")
                    and isinstance(last_alive, (int, float))
                    and isinstance(root.get("first_seen"), (int, float))
                    and last_alive - root["first_seen"] <= DETACH_HANDSHAKE_S
                    and not bundle.alive_at_end(intermediate)
                    and isinstance(root.get("first_chain"), list)
                    and root["first_chain"][:1] == [intermediate.get("pid")]):
                excused = set(root["first_chain"])
                handshake = {"root": root["pid"], "intermediate": intermediate.get("pid"),
                             "launcher": launcher.get("pid"),
                             "alive_after_first_sighting_s": round(
                                 last_alive - root["first_seen"], 3)}
        union = worker.get("ancestors_seen")
        counted = set(union) - excused if isinstance(union, list) else set()
        final = latest_chain(worker.get("ppid"))
        results.append({
            "pid": worker["pid"], "root": root["pid"], "handshake_excused": handshake,
            "opencode_in_counted_samples": sorted(opencode & counted),
            "final_chain": final, "opencode_in_final_chain": sorted(opencode & set(final)),
            "ok": ((worker.get("samples_alive") or 0) >= 1 and isinstance(union, list)
                   and not opencode & counted and not opencode & set(final)),
        })
    return results


def watch_complete(watch: dict | None) -> bool:
    """A `WorkspaceWatch` summary that measured: the workspace, more than once,
    every file each time."""
    watch = watch or {}
    return (watch.get("scope") == "workspace"
            and (watch.get("censuses") or 0) >= 2
            and watch.get("truncated") is False)


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


def _written(leg: dict) -> list[str]:
    """Every workspace path the leg moved: the hashed pair's delta, and every
    path the leg's watch saw created, changed or removed in between."""
    touched = (leg.get("workspace_watch") or {}).get("touched") or {}
    return sorted(set(leg.get("census_delta") or {}) | set(touched))


def grade_init_status(result: harness.RowResult) -> tuple[dict, dict, None]:
    bundle = Bundle(result)
    checks, observed = common_checks(bundle)
    debug = bundle.gate.get("debug_config") or {}
    init, status = bundle.leg("init"), bundle.leg("status")
    init_result, status_result = _tool_result(init), _tool_result(status)
    init_added = sorted(name for name, (before, after)
                        in (init.get("census_delta") or {}).items()
                        if before is None and after)
    init_written, status_written = _written(init), _written(status)
    facts = {fact for e in bundle.stub_entries(init) for fact in e.get("facts") or []}
    status_payload = status_result.get("result") or {}
    # The skill dir the command's legs ran, read off the tool's own uv argv:
    # the template this build ships names no path, so there is nothing to
    # read one off on the config side (`debug_config` in the driver).
    command_dirs = sorted({str(_option(part["result"].get("argv") or [], "--project"))
                           for leg in (init, status)
                           for part in leg.get("tool_parts") or []
                           if isinstance(part.get("result"), dict)})
    command_skill_dir = command_dirs[0] if len(command_dirs) == 1 else None
    observed.update({
        "command_keys": debug.get("command_keys"),
        "command_skill_dir": command_skill_dir,
        "command_skill_dir_source": "the tool's uv argv (--project), init and status legs",
        "template_matches_payload": debug.get("template_matches_payload"),
        "description_matches_payload": debug.get("description_matches_payload"),
        "init_facts": sorted(facts),
        "workspace_set_aside": bundle.gate.get("workspace_set_aside"),
        "init": {k: init_result.get(k) for k in ("ok", "state", "exit", "error",
                                                  "artifacts")},
        "init_mode": (init_result.get("result") or {}).get("mode"),
        "init_added": init_added,
        "init_written": init_written,
        "init_watch": init.get("workspace_watch"),
        "status": {k: status_result.get(k) for k in ("ok", "state", "exit",
                                                      "error")},
        "status_payload": status_payload,
        "status_census_delta": status.get("census_delta"),
        "status_written": status_written,
        "status_watch": status.get("workspace_watch"),
        "legs": [{k: leg.get(k) for k in ("label", "exit", "timed_out",
                                          "wall_clock_s", "error_events")}
                 for leg in bundle.legs()],
    })
    checks.update({
        "command_registered": "multi-llm" in (debug.get("command_keys") or []),
        # `command["multi-llm"]` is the payload's `dist/command.md` byte for
        # byte, every `{SKILL_DIR}` resolved to the payload's tree, recomputed
        # from the INSTALLED bytes — so only this payload's hook made it.
        "command_is_the_payloads_own": (
            debug.get("template_matches_payload") is True
            and debug.get("description_matches_payload") is True),
        "command_ran_the_payload_skill_dir": command_skill_dir == PAYLOAD_SKILL_DIR,
        "command_expanded_into_the_turn": "multi_llm_command" in facts,
        "opencode_runs_exited_zero": all(
            leg.get("exit") == 0 and not leg.get("timed_out")
            for leg in (init, status)),
        "init_returned_success": (
            init_result.get("ok") is True and init_result.get("exit") == EXIT_OK
            and init_result.get("state") == "done"
            and (init_result.get("result") or {}).get("mode") == "init"),
        "init_created_project_files": (
            watch_complete(init.get("workspace_watch"))
            and ".multi-llm/providers.yaml" in init_added
            and all(name.startswith(".multi-llm/") for name in init_written)),
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
        "status_wrote_nothing": (watch_complete(status.get("workspace_watch"))
                                 and not status_written),
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
    ancestry = worker_ancestry(bundle, workers)
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
        "host_opencode_pids": sorted(bundle.host_opencode_pids()),
        "worker_ancestry": ancestry,
        "workers": [{k: w.get(k) for k in ("pid", "ppid", "first_ppid", "pgid",
                                            "sid", "state", "args", "first_seen",
                                            "last_alive", "samples_alive",
                                            "first_chain", "ancestors_seen")}
                    for w in workers],
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
        # Measured in every sample a worker process was alive in after the
        # detach handshake, from that sample's own process table: no opencode
        # host process anywhere on its ppid chain (`worker_ancestry`). The
        # latest chain alone stops at init once the double spawn's
        # intermediate is gone, whatever it was before. The plugin's own
        # children (the launch, the polls) ARE opencode's, which is exactly
        # what the worker must not be.
        "worker_outside_opencode_tree_in_every_sample": (
            bool(workers) and bool(host_pid) and all(entry["ok"] for entry in ancestry)),
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
        plan_dir) if plan_dir and run_dir.startswith(plan_dir + "/") else None
    watch = leg.get("workspace_watch") or {}
    touched = watch.get("touched") or {}
    delta = sorted(leg.get("census_delta") or {})
    written = _written(leg)
    # The run's own registry entry is the one write outside the plan's dir
    # the product documents, and only while the run lives: one still there at
    # the end is as stray as any other file.
    bookkeeping = ({registry_entry(run_id), registry_entry(run_id) + ".tmp"}
                   if run_id else set())
    transient = [name for name in written if name in bookkeeping
                 and name not in delta
                 and (touched.get(name) or {}).get("present_at_end") is False]
    stray = [name for name in written
             if not (plan_dir and name.startswith(plan_dir + "/"))
             and name not in transient]
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
        "workspace_watch": watch,
        "workspace_written": written,
        "transient_bookkeeping": transient,
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
        # Every path the leg moved, ends AND everything the watch saw between,
        # is under the plan's dir, the run's transient registry entry aside.
        "no_stray_writes": watch_complete(watch) and bool(written) and not stray,
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
    receipt = abort_receipts[0] if len(abort_receipts) == 1 else {}
    receipt_result = receipt.get("result") or {}
    abort_processes = [p for p in bundle.processes if run_id
                       and "multi_llm.py status" in p.get("args", "")
                       and "--abort" in p.get("args", "")
                       and f"--run-id {run_id}" in p.get("args", "")
                       and "/opt/mllm/bin/uv" in p.get("args", "")]
    dispatch = abort.get("dispatch") or {}
    dispatch_run = dispatch.get("run") or {}
    group = run.get("pgid")
    in_group = [p for p in bundle.processes if group and p.get("pgid") == group]
    workers = bundle.workers(run_id)
    descendants = [p for p in bundle.processes if any(
        w["pid"] in bundle.ancestors(p["pid"]) for w in workers)]
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
        "run_at_dispatch": dispatch,
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
            and bool(abort_processes) and bool(serve_pid)
            and all(p.get("ppid") == serve_pid for p in abort_processes)),
        # The driver's own reading at the first 20 ms scan that saw the tool's
        # abort start, before that abort could reach its kill: the claimed run
        # still `running` in its record, a worker process of it alive.
        "run_live_when_abort_dispatched": (
            dispatch.get("seen") is True and bool(run_id)
            and (dispatch.get("abort") or {}).get("ppid") == serve_pid
            and dispatch_run.get("run_id") == run_id
            and dispatch_run.get("state") == "running"
            and any(w.get("state") != "Z" for w in dispatch.get("workers") or [])),
        # The dispatcher's own answer. `status --abort` exits 0 (`done`) only
        # when `abort_run` found the run non-terminal and alive and stopped it;
        # exit 2 (`nothing-to-do`) is its "nothing was running" for a run
        # already over (measured, 2026-10-25: exit 0, result `done`).
        "abort_stopped_a_live_run": (
            receipt.get("exit") == EXIT_OK
            and receipt_result.get("exit") == EXIT_OK
            and receipt_result.get("state") == STATE_FOR_EXIT[EXIT_OK]
            and receipt_result.get("mode") == "status"),
        "run_recorded_aborted": run.get("state") == "aborted",
        # ...and the record says the dispatched abort closed it, not the
        # worker's own signal handler (`aborted_by_request (SIGTERM)`).
        "run_closed_by_the_dispatched_abort": (
            run.get("error") == ABORTED_BY_DISPATCH
            and run.get("exit_code") == EXIT_NEEDS_HUMAN),
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


#: Each scenario's grader, by the name `gate-grader.json` records.
GRADERS = {
    "init-status": grade_init_status,
    "detach": grade_detach,
    "supervision": grade_supervision,
    "completion": grade_completion,
    "cancellation": grade_cancellation,
}


def grade_bundle(out_dir: Path, scenario: str) -> tuple[dict, dict, str | None]:
    """Grade a bundle already on disk, exactly as its live row would."""
    return GRADERS[scenario](result_from_bundle(out_dir, scenario))
