# Unattended decisions and prerequisites

Read when adding a decision flag, changing interactivity, or changing prerequisite gates.

Back to the [development guide](../../skills/multi-llm/AGENTS.md). These notes retain detailed contracts and regression history from that guide. Paths beginning with `docs/`, `packaging/`, or `plans/` are repository-relative; other implementation paths are relative to `skills/multi-llm/`. Dated incidents explain the rules; verify version-specific claims against current code.

## Contents

- [Decision sites and recorded resolutions](#decision-sites-and-recorded-resolutions)
- [Explicit host and interactivity](#explicit-host-and-interactivity)
- [Prerequisite enforcement](#prerequisite-enforcement)
- [Adding a decision point](#adding-a-decision-point)

## Decision sites and recorded resolutions

**`utils/unattended.py` owns the unattended contract (D6): fifteen decision *sites*, three
dispositions, no questions.** `ROWS` is the closed table — each site keyed by a
stable `site_id`, the modes that reach it, the disposition it lands on (**auto-select**
/ **fail closed** / **explicit flag** — "ask in free text" is not a fourth option) and where it is
enforced; `DECISION_FLAGS` is the D5a input contract, one flag per row whose values enumerate
exactly what a question would have rendered. **Exit 3 is reserved across every mode for "unattended
run cannot proceed"** (missing uv, unmet prerequisite, contradictory or empty decision flag),
distinct from the exit-2 phase-already-complete guard, so a host can tell "needs a human" from
"already done".

`validate_decision_flags(argv)` runs in the dispatcher *before* any parser so `--models ""`,
`--human-task <id>=` and an unknown `--force-strategy` are exit **3** at parse time rather
than argparse's exit 2; `choices` on each parser is the second layer.
`record_resolution(phase_dir, site_id, decision)` prints `[AUTO_RESOLVED] <site>: <decision>` and, under `--detach`, appends to
`run.resolutions` via `detach.record_resolution` → `status_file.append_resolution` (a foreground
`.status.json` still gains no `run` keys — `detach.record_resolution` keeps its
handle guard, unlike `report_progress`; the `[AUTO_RESOLVED]` line is the foreground
record).

Auto-resolutions are announced only under `--yes`, so with `--yes`
unset a run is byte-for-byte what it was. **The apply phases' own two sites (rows 10 and 11) are
recorded, not just taken**: both fire *before* `run()` crosses the detach boundary,
where no run handle exists yet and `detach.record_resolution` is a silent no-op — so
`ApplyOrchestratorBase._record_resolution` prints the marker at the site and `_flush_resolutions` replays the entries
into `run.resolutions` immediately past the boundary.

One leg prints it: under `--detach` the invoker reaches both sites and then re-executes
itself as the worker, which reaches them again, so the marker is skipped on the leg that is about to
fork (`_is_detach_invoker`) and kept on the one that writes the record. Row 11 is not taken at all
under `--dry-run` — that flag is itself one of the nine that bypass the no-selection
confirmation, and the run it describes as "apply all valid items" applies none. Measured 2026-09-16
on a `--yes --exec-subagents` apply-suggestions run: 29 of 33 groups were applied because nobody had
selected anything, `.status.json` said `"resolutions": []`, and the row-11 decision left
no trace in the log, the payload or the status view.

`--yes` also now *implies* `--agent-decide` in every apply phase
(`apply_orchestrator_base.apply_unattended_defaults`) and resolves the human-task strategy to `skip-dependents`; it
**never** implies `--skip-prerequisites`.

## Explicit host and interactivity

**Python never sniffs its host (D6b).** No `isatty()`, no `CI=` read,
no `--print` detection: interactivity is *requested* (`--interactive`), never
detected, so the same argv produces the same phases in a terminal and in a pipe.
`utils/interactive.is_tty()` is gone and `resolve_models(..., unattended=True)` never reaches a picker — it returns
`defaults.models` or raises `NeedsHuman` (exit 3). `full-workflow.md`'s "no TTY →
skip review-tasks" branch is deleted.

`tests/test_unattended.py` parses every runtime module's AST (so a *comment* about
`isatty` is not a call) and fails any reintroduction; the one allowlisted
`isatty` is `utils/metrics.py`'s ANSI-colour gate, which decides how a line looks
and never which phases run. `--yes` with `--interactive` is exit 3, never
precedence. **`MULTI_LLM_ROUTE` is the one route signal, and it picks only the spelling a
printed line uses, never which phases run** — the same test the `isatty` allowlist
applies.

The tool servers own it: `modes.ROUTE_ENV`/`modes.TOOL_ROUTES` are its name and values, the
surface generator writes each host's pair into `dist/cli-surface.json` as `dispatcher.route_env`, and
both servers set it on every child they spawn, polls included, so neither server spells the name or
the value. `utils/next_phase.resolve_route` is the only reader and validates it: a value outside
`TOOL_ROUTES`, or one the resolved host has no `next_command` cell for
(`package` under claude-code), is treated as unset with one stderr line.

A detached worker started on a tool route records it as `run.route`
(`detach.RunHandle`), and a reader whose environment names none (a shell's
`status`, `util next-command`) spells that run's commands in that route, its
template and the reserved flags it composes (`resolve_route`'s `recorded`,
`next_phase.recorded_route`: the pinned run's record, else that phase's or the plan's newest run of the
reader's host; another host's record and claude-code ignore it; no record keeps
`default`; fix round 24, live run 10: a shell's `status --host codex` printed the file
shim's sentence for a run the plugin's tool started).

On claude-code it is ignored: the route there is read from the dispatcher's own skill root
(`cli_common._SKILL_DIR`, symlinks preserved) — `skill` when it ends in an
`install_paths` tail (`.claude/skills/multi-llm`), else the plugin's `default`. A
valid value exported by hand beside a file shim of the same host is the accepted residual; it is not
a user setting.

## Prerequisite enforcement

**The prerequisite gates are the one place `--yes` must not be generous (D6a).**
`check_workflow_prerequisites.py` owns rows 5/6/13 — no thin orchestrator, no gate logic in markdown.
Standalone + `--yes` = exit 3 with one `[PREREQUISITE_UNMET]` block naming *both*
fixes (apply first, or `--skip-prerequisites`) in absolute paths and D5a-shaped commands;
`--skip-prerequisites` = `[PREREQUISITE_SKIPPED]` + a recorded skip through the same
`--skip --reason` state path (surfacing downstream as one `[PREREQUISITE_SKIPPED_UPSTREAM …]` line, never
re-asked); inside the workflow, `ensure_prerequisites(..., in_workflow=True, apply_runner=…)` runs the missing apply phase **once** and
re-checks — a second unmet re-check is exit 3, not a retry. Nothing is produced before the gate.

**`full_workflow_orchestrator.py` is the caller that injects `apply_runner`** (applying suggestions
needs the agent's subagent channel) — for row 5 via `generate_tasks_orchestrator.run_generate_tasks(..., in_workflow=True, apply_runner=…)` and for row 6 before it
dispatches `implement`. The runner is supplied only under `--yes` (the
auto-run is the *unattended* resolution) and is bounded to **one auto-run per phase per run**: a
second request for the same phase is refused and the gate ends at exit 3. With no runner, workflow
mode behaves like standalone. **Streams:** this script's stdout stays pure JSON (callers
`json.loads` all of it) and its `[PREREQUISITE_*]` blocks go to stderr + the JSON's
`message`; `implement_orchestrator.py`, whose stdout is already prose+markers, prints its
own on stdout.

## Adding a decision point

**Adding a decision point without its flag is a review error.** A new gate needs a row in
`utils/unattended.ROWS` (site id, modes, disposition, where it is enforced), a `DECISION_FLAGS`
entry whose parser really accepts it, the resolution written into `instructions/full-workflow.md` *and* the
mode file, and — if it auto-resolves — a `record_resolution` call. `tests/test_unattended.py` asserts
all four; `tests/test_unattended_e2e.py` then re-runs every standalone mode under `--yes`
with **stdin at /dev/null, no TTY and `gum`/`fzf` replaced by
tripwires**, so "zero prompts" is observed rather than inferred from the absence of an
`AskUserQuestion` string.
