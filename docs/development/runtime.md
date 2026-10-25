# Detached runs and recovery

Read when changing process lifecycle, run identity, status persistence, uv bootstrap, or durable retry commands.

Back to the [development guide](../../skills/multi-llm/AGENTS.md). These notes retain detailed contracts and regression history from that guide. Paths beginning with `docs/`, `packaging/`, or `plans/` are repository-relative; other implementation paths are relative to `skills/multi-llm/`. Dated incidents explain the rules; verify version-specific claims against current code.

## Contents

- [Detach boundary and worker lifecycle](#detach-boundary-and-worker-lifecycle)
- [Status file schema](#status-file-schema)
- [uv bootstrap](#uv-bootstrap)
- [uv runtime flags](#uv-runtime-flags)
- [Process primitives](#process-primitives)
- [Durable retry commands](#durable-retry-commands)

## Detach boundary and worker lifecycle

**`--detach` self-daemonizes every long-running mode**: the fan-out modes
(`review-plan`, `review-tasks`, `review-code`, `ask`)
plus the executor-backed write modes (`apply-suggestions`, `apply-task-suggestions`,
`apply-code-fixes`, `implement`) plus `generate-tasks` and
`full`, whose orchestrators landed with the Phase-3 sequencing — so the plumbing
they registered for is now the plumbing that runs (`generate-tasks` claims
`{plan}/generate-tasks/`, `full` claims `{plan}/full/`)) plus the two
composites `improve-plan` and `improve-code` (`{plan}/improve-plan/`,
`{plan}/improve-code/`). Everything else exits 3 rather than accepting a flag that does nothing.

The write phases cross the **same** boundary through `apply_orchestrator_base.maybe_detach_write_phase` /
`implement_orchestrator.run_implement_write_phase`, placed after all pre-flight and before the first child, so one spawn
implementation and one `.status.json` claim serve both halves; per-task write progress lands
under `progress.write_subagents`. One boundary crossing, `utils/review_orchestrator_base.maybe_detach`, sits *after* all
pre-flight (flags, inputs, model resolution, phase-dir creation, lock pre-check) and *before* any
model or agent CLI runs, so a bad flag or an unconfigured provider fails on the caller's terminal
with **no child and no log**.

The `--reaggregate` branch of every review orchestrator crosses the same boundary, after its
one pre-flight (the phase dir exists): measured 2026-09-14 on review-tasks, that branch ran ahead of
the boundary, the re-dispatch of the failed validation batches finished in the invoker's foreground
with no run banner, and the opencode plugin — which composes `--detach` on every launch
and adopts a run only from the banner — reported the completed run as "the launch reported no run_id
on its banner".

The invoker forks (POSIX: `setsid` + a second fork, so the worker reparents to init
and the invoker has no child to reap; Windows: `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`), waits ≤10 s for the worker
to publish its own identity, then prints the log path plus the
`status`/`--abort` commands. **`{phase}/orchestrator-run.log` is handed in as the
worker's stdout/stderr on both platforms, so it holds the whole worker and not just the part after
the boundary** — `process_control.spawn_detached` opens it append-only and dup2s it onto the child's fds 1/2
in the intermediate, and `enter_detached_worker` (which `maybe_detach` reaches only at the
*end* of pre-flight) compares `(st_dev, st_ino)` and reopens nothing.

Until 2026-09-16 the POSIX leg gave the worker `/dev/null` and the worker reopened the
log itself, so every setup line an orchestrator prints on its way to the boundary went on the floor:
a completed `apply-suggestions` log opened at `[DETACHED_RUN]` with no
`Loaded 33 suggestion groups` and no `No user preferences detected` — the notice under which it applied all 29
items — and `review-plan`'s log had no
`Loaded plan:`/`Backup saved:`/`Models:`. The invoker's own copy still
goes to the caller's terminal, which is the pre-flight rule; the setup lines therefore sit *above*
the `[DETACHED_RUN]` banner in the log, and a worker that dies before the handshake now leaves
its reason in the file the launcher's failure message tails.

The child re-executes `uv run --project <skill-dir> -- python <skill-dir>/multi_llm.py <mode> … --internal-worker` — **never bare `python`**
(`sys.executable` inside `uv run` loses the lockfile and `--project`)
— with `--detach`'s implications (`--yes`) written out and paths
absolute. `--detach` implies `--yes` and is exit 3 with
`--interactive` at parse time. Never combine it with a host's own background primitive:
Claude Code uses `run_in_background` and never passes `--detach`.

**The next-step block in `orchestrator-run.log`**: a finished worker prints `[NEXT_COMMAND] <command>`,
then per `optional_before` entry `[NEXT_COMMAND_OPTIONAL_WHY] <why>` directly above `[NEXT_COMMAND_OPTIONAL] <command>`,
or `[WORKFLOW_COMPLETE]` followed by one `[MANUAL_TASK] <id> <title>` line per deferred implement task,
above its result line (`next_phase.render_next_phase_lines`, the one renderer); a command line carries the command
byte for byte and nothing else, so copying one copies no prose. A launched detached run's
`user_next_step` reaches a host on its **terminal poll**, never on the invoker (which owes the
poll as `next`).

Because `build_worker_argv` rewrites `--plan-file` to an absolute path, the invoker
ships the plan argument as it was typed in `MULTI_LLM_PLAN_ARG` (beside `MULTI_LLM_REPLAY_FLAGS`,
same reason: the worker's own argv can only carry the rewritten path); `begin_worker_run` pops
it and `RunHandle` records it once as the run record's `plan_arg`, next to
`plan_file`. The worker's block and the terminal poll print that value, so a relative
plan stays relative; a record without it (older, or a hand-started worker) falls back to the
absolute `plan_file`.

A native leg's continuation has the same gap in the foreground: its `next`
(`--reaggregate`, `--reaggregate-consolidation`) is a printed follow-up and names the plan
absolutely, so the hand-off writes the plan as typed into the continued phase's
`.status.json` as `plan_arg` and the continuation prints that, only while it
names the same file from its cwd (`multi_llm._remember_plan_spelling` / `_continued_plan_arg`; racing re-run 4,
Claude Code 03, 2026-10-05).

Since fix round 14 the native hand-off itself names the plan as typed when that spelling passes the
same check (`cli_common.typed_plan_for`: the generate-tasks follow-up, the review and code-review
validation hand-offs, consolidation's, the salvage reaggregate), and implement's handoff JSON
carries `plan_arg` for the closing `util finalize-tracking --plan-file`. Both files sit in the run's
output folder, which every jailed batch can write, so a `plan_arg` read back from either
is printed only when `next_phase.recorded_plan_spelling` passes it: `realpath(strict=True)` names the same
existing file (non-strict collapsed `plans/$(cmd)/../foo.md` onto `plans/foo.md`) and it holds
only `shlex.quote`'s bare characters or a plain space, else the plan as received (fix round
11).

A continuation with no passing hand-off spelling falls back to the phase's run record's
`plan_arg` under the same check, so the "advance it with: … --reaggregate <absolute
plan>" line a detached run's status prints ends with the plan as its launcher typed it (fix round
11).

## Status file schema

**`{phase}/.status.json` is one file with one schema**: today's checkpoint plus two reserved
objects, `run` (the process machine `starting → running → {done, failed, aborted}`, heartbeat, pid +
`process_start_time` identity, `exit_code`, `error`,
`resolutions`) and `progress` (display only). `run` is the
detached worker's alone (only `claim_run` writes it); `progress` is written
by **every** run, foreground or detached — `detach.report_progress` falls back to
`status_file.update_progress` when no handle owns the phase dir, so the invariant is "a foreground
`.status.json` gains a `progress` object and still no `run`
object", not "gains no new keys".

Its reserved keys are `{step, batch_index, batches_total, tasks_done, tasks_total, tasks_generated, failed_task_ids, models, models_done, models_succeeded, models_total, interval_s}` (`status_file.py`'s module docstring is the
contract): `models` is the fan-out's per-model map keyed by spec,
`models_done` counts *finished* slots and `models_succeeded` the slots that produced a
usable result, `interval_s` is the resolved ticker cadence. `run.state` is
**not** the top-level `state` (the phase machine resume reads) — a crashed fan-out is
both.

`progress.models_done` is written by the fan-out that owns the models: the review base advances it
to `len(model_specs) - len(failed_models)` beside `models_completed`, and `code_review_orchestrator` does the same at
its own `models_complete` write and, since 2026-09-16, via a `slot_finished()` of its own
inside `run_all_reviews` (mirroring the base's) — before that fix its fanout step wrote
`models_done=0` and nothing advanced it until the last model returned, so a mid-run poll of a
finished-in-part run read `models 0/3` beside three completed reviewers.

**Progress is reported while it happens, not only at the end of a stretch.** `run_all_models`
advances `models_done` per finished *slot* (a failed or skipped slot included — the
recomputation at `models_complete` is what turns slots back into successes),
`dispatch_subagent_tasks` reports the step its batch is on before it dispatches
(`validating`, `revalidating`, `consolidating`, else `dispatching <label>`
— `_DISPATCH_STEPS`), and both executor mirrors write per finished task rather than once at the
end (`subagent_executor.run_tasks` counts against the batch's full size; `write_batch_scheduler._run_batch` takes its
`finished_in_batch` rows under a lock because a disjoint group dispatches on threads —
`in_flight` in the same mirror now names the opposite, a dispatch started and not yet
classified, F10).

Measured 2026-09-16 on a 516 s `review-plan`: 435 s of fan-out reported
`models 0/4` and 62 s of validation reported `grouping`, leaving ~5 s of the
run with an informative progress line. Primitives live in `utils/status_file.py` (re-exported by
`review_orchestrator_base` as `write_status`/`update_status`, so nothing about the write
path changed) because `status` must stay sub-second;
`run`/`progress` merge key-by-key like `output_files`, a plain
`write_status` cannot erase them, and every write after the worker's initial claim is
**compare-and-set on `run_id`** so a closed-out run can never resurrect itself.

Liveness is the heartbeat (10 s; expired at 60 s = six missed beats); `{phase}/.lock`
(`utils/process_control.py`: `flock`/`msvcrt`, held on an open fd) makes
one writer per phase dir, and a second run against a live holder is exit 3 naming it. Every run ends
in exactly one terminal state with `run.exit_code` recorded — the worker records its own
(nothing is its parent after the double spawn), and `status` closes out
`failed(heartbeat_timeout)` with `exit_code: null` when the heartbeat expires and the identity probe
says the process is gone.

A composite whose run ends other than `done` (aborted, closed out dead, failed —
`full` included) also retires the live model maps in the phase dirs it drove, which
are exactly its `sequence` phases' dirs through `PHASE_DIRECTORIES`
(`detach._retire_driven_phase_maps` over `_driven_phase_dirs`, skipping a dir with a live
`run` of its own); until 2026-10-09 `full` swept every sibling
holding a `.status.json`, so an aborted `full` marked a foreground
`ask` beside it `cancelled`, and it now leaves `ask/`,
the `improve-*` run dirs and anything else it never drove alone.

The machine's middle word is written too: both sites that materialize `validation_tasks.json` write
`state: validation_pending` beside the `validating` progress step, which `status`
used to print as `phase state: grouping_complete` / `progress: validating` for the whole validation stretch.
**The phase machine's terminal word is `complete`** (2026-09-11):
`run_review_orchestration`, `reaggregate_from_existing_files` and `code_review_orchestrator`'s three completion sites
write `state: complete` and `progress.step: complete` right after `mark_phase_completed`, so
`status` shows a finished run as finished — before that the last checkpoint the
fan-out wrote (`grouping_complete`) stayed on disk forever and the status view read a done run as
still grouping.

No reader gates resume or reaggregate on the exact word; `state.json`'s
`phases_completed` remains the record that matters.

## uv bootstrap

**The uv bootstrap is split across three layers (D6c), and the wording lives in
`hosts/uv-missing.txt`.** `install-host --check` (install time) and each rendered shim's
`command -v uv` preflight (invocation time — the only host-side logic that cannot be Python)
are D7's; the **Python backstop** is `unattended.require_uv()`, run by the dispatcher before every mode
for the cases the first two cannot cover (uv removed after install, a host calling
`multi_llm.py` directly, a hand-edited shim). Same message, same exit 3, never an install,
and **no `python3 -m` fallback ever** — dependencies are pinned in `uv.lock`,
so a bare interpreter turns "install uv" into a `ModuleNotFoundError` or a silently wrong version
mid-fan-out.

`references/uv-check.md` is maintainer documentation — the written contract each shim's inline uv step
is authored against, not a file anything reads on the shim side at run time — and it points at the
message file; **`utils/detach.resolve_uv()` delegates to `unattended.find_uv()`** rather than keeping its
own probe — it used to carry a hand-written two-entry list — the two home-relative locations only,
with no `.exe` variants, no Homebrew and no system paths — and name that shortened
list back to the user in its own error text, so the detached path every fan-out mode goes through
could refuse on a machine where the foreground preflight had just succeeded.

`tests/test_uv_candidates.py` owns the parity: every row of its `READERS` table, each asserted
element-for-element **and in order** against `UV_CANDIDATES`, plus a discovery half whose
`git grep` hit set must *equal* the reader table — so an unlisted reader goes red on the
PR that adds it, rather than drifting silently.

## uv runtime flags

**`utils/unattended.uv_run_args()` is the single owner of the `uv run` flag list**, and the
flags are `--frozen --no-dev`. `--no-dev` is the load-bearing half:
`--frozen` suppresses only the lockfile *staleness check* and does not stop the sync,
and `uv run` syncs the **dev** group by default — measured against this tree,
`--frozen` alone installs 17 packages (ruff, pytest, jsonschema…) where
`--frozen --no-dev` installs one (pyyaml), so without it a first dispatch pulls the whole dev
group and every offline posture fails on a package it never needed.

Baking it in is non-destructive: `uv run --no-dev` against a dev-synced environment leaves
those packages in place and installs only what is missing (only `uv sync --no-dev` would remove
them, and nothing on these paths calls it).

Four consumers read that one owner — the foreground launcher, `cli_common.dispatcher_prefix()` (every printed
follow-up and every `MULTI_LLM_RESULT.next`), `detach.build_child_command()` (the worker re-exec) and
`cli_common.render_instruction_text()` (the `{UV_RUN_FLAGS}` token every command in `instructions/*.md` and
in each `hosts/*/shim.md.tmpl` carries, so the file route's verbatim runbook text cannot dispatch
with a different flag list than Python does) — plus one deliberately hand-typed spelling: Claude
Code's `SKILL.md` is read **raw** by the host (it expands `${CLAUDE_SKILL_DIR}` itself
at load time), so a token there would reach an agent unsubstituted while a flagless
`uv run --project` would put the first-run dev-group sync on the host that runs most — it
therefore writes `--frozen --no-dev` out, and `test_instruction_portability.TestTheClaudeShimSpellsTheFlagsOut` compares that copy back to
`uv_run_args()` element for element so it cannot drift (those two shim-level commands are
consequently pinned to the *default* list; the core the shim loads at step 4 is rendered and does
honour `MULTI_LLM_UV_RUN_ARGS`).

The maintainer invocations below (`uv run --project skills/multi-llm -- pytest`) are outside the rule on purpose: they want
the dev group. Only the first consumer was ever asserted, which is how the worker came to re-exec
with **neither** flag: a careful `--detach`ed launch handed off to a child that
recreated the 17-package sync in a log where nobody would see it. `child_env()` also
exports the resolved `MULTI_LLM_UV`, so the worker re-resolves to the binary the preflight
approved instead of re-probing a `PATH` a detached session may not share.
`MULTI_LLM_UV_RUN_ARGS` **replaces** the list and never appends — that is what makes
`--no-sync` expressible at all — so it can legitimately drop `--no-dev`, and
blank or whitespace-only behaves as unset.

## Process primitives

**`utils/process_control.py` is the only place
`fcntl`/`msvcrt`/`ctypes` process internals appear**
(locking, liveness, creation-time identity, process-group/Job Object teardown, the detached spawn).
On Windows the executor's runner, the sandbox probe, the verify command and each Lane A provider
CLI (`llm_client._run_provider_cli`) are spawned through `popen_tree`, which puts each in a Job
Object of its own before it runs, and their timeout ends that job. The Lane A CLI gets no process
group of its own, so a terminal Ctrl-C still reaches it; its job travels into `live_cli_registry`,
and the timeout, an interrupt, a stop from outside (`130`/`143`, or `STATUS_CONTROL_C_EXIT`) and
the registry's sweep each end it right after the CLI's own `TerminateProcess`. A CLI that exits on
its own leaves its job alone, as a POSIX CLI's groups are left alone. A detached run's job has a
name, `Local\multi-llm-run-<run_id>`: the launcher spawns the worker into it suspended
(`spawn_detached_run`) and the worker holds a handle for life (`enter_detached_worker(...,
run_id=)`), because a job's name lasts only while a handle to it is open. The kills by recorded
pid (`status --abort`, the dead-run close-out, `util runs --sweep`) open it by name, trust it only
if a live recorded pid is in it, and kill through it (`kill_run_tree`); the failed-handshake
teardown uses the launcher's own handle. `taskkill /T` is left for a child that got no job and for
a run whose job cannot be opened or trusted (another logon session, a worker that never held it,
a squatted name). The one other `ctypes` user on Windows is `utils/prompt_files.py`, which
removes prompt files through handles it holds open; that is filesystem work, not process control.
It is the façade `plans/windows-macos-portability.md` owns; per that plan's sequencing rule this work typed it
first, so its signatures are a cross-plan interface. A test fails any other runtime module that
imports `fcntl` or `msvcrt`.

## Durable retry commands

**Nothing durable stores a skill-dir-absolute command.** The plugin cache is keyed by commit hash,
so the skill dir moves on every update. `{phase}/.status.json` records `reaggregate_mode` +
`plan_file` + `replay_flags` (`cli_common.durable_reaggregate_fields` — the caller's selection
flags as a *list*, never a resolved panel and never a command string) and readers rebuild the
command from the *running* skill dir through `cli_common.retry_command` (`detach.resolved_reaggregate_command`, which
is what `status --view run` prints). A stored flag list is spliced back verbatim, never validated
against today's parser: dropping an unrecognized `--quick` is how a retry widens to
`defaults.models`.

One flag is left out by meaning rather than by name: in an apply phase or `full`, a
replay carrying `--yes` or `--detach` drops `--agent-decide`, which
`--yes` already sets (`cli_common.drop_implied_judge_flag`; racing run 4: a tool result offered the
resume as `--agent-decide --resume` beside a `resume_command` of `--resume`).
`implement`'s `--human-task` answers are never replayed either
(`cli_common._REPLAY_DROP_ANSWERS`, also off a stored list): `state.json` keeps them and a plain
`--resume` applies them, and a replayed answer for a task no longer
`failed` is refused (fix round 14).

An answer answers one stop: the first status written for its task that the answer did not write
(`state_manager._consume_answer`, in `update_task_status_in_state`, so both legs) drops it, and
`--task <id>` does not apply one recorded for that task (fix round 15). Printing a resolved
path is fine — it is session-scoped; persisting one is not. Both places the review base prepares
validation tasks pass `orchestrator=<the phase's own script>`: the fan-out call omitted it (measured 2026-09-14),
`prepare_batched_validation_tasks` defaulted to review-plan's, and a failed review-tasks run told the session
model to re-invoke `review-plan --reaggregate`.
