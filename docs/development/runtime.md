# Detached runs and recovery

Read when changing process lifecycle, run identity, status persistence, uv bootstrap, or durable retry commands.

Back to the [development guide](../../skills/multi-llm/AGENTS.md). These notes retain detailed contracts and regression history from that guide. Paths beginning with `docs/`, `packaging/`, or `plans/` are repository-relative; other implementation paths are relative to `skills/multi-llm/`. Dated incidents explain the rules; verify version-specific claims against current code.

## Contents

- [Detach boundary and worker lifecycle](#detach-boundary-and-worker-lifecycle)
- [Status file schema](#status-file-schema)
- [uv bootstrap](#uv-bootstrap)
- [uv runtime flags](#uv-runtime-flags)
- [Process primitives](#process-primitives)
- [Advisory guards](#advisory-guards)
- [Atomic file writes](#atomic-file-writes)
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

The write cores tell a host the same thing. Step 2 of `instructions/apply-suggestions.md`,
`apply-task-suggestions.md`, `apply-code-fixes.md` and `implement.md` runs the orchestrator in the
foreground only on the native-subagent path, where it emits batch or task JSON and applies nothing;
under `--exec-subagents` that one command is the whole write phase, so the core says *run detached*
(`--detach`). The `python`/`required` *run detached* sentences in `utils/host_capabilities.py` and
both shim templates' step 5 name the apply modes and `implement` beside the fan-out modes, while a
single-model call that applies no batch (`generate-tasks`) may stay in the foreground. An unscoped
"run it in the foreground" sent host-lane row `codex-apply-executor` into the foreground with no
`run` block to grade (2026-11-06); `tests/test_instruction_portability.py`
(`TestTheWritePhasesDetachUnderTheExecutor`) guards the wording.

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

On Windows, a `.cmd` trampoline can lose those inherited handles before Python starts.
The launcher also passes the log path in `MULTI_LLM_WORKER_LOG`; the dispatcher's
`--internal-worker` entry consumes it and restores stdout/stderr before any pre-flight output,
only when `MULTI_LLM_DETACHED_WORKER=1`. This restores logging alone: run ownership, locks,
handshake and Job Object setup still happen at the detach boundary. The late reopen remains
a fallback for workers started outside the dispatcher. The transport is removed before any
provider child starts (`test_worker_preflight_recovers_output_lost_by_a_launcher`).

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
object", not "gains no new keys". `claim_run` drops the predecessor's `run`, `progress` and
run-scoped results (`status_file.RUN_SCOPED_KEYS`: `high_priority_groups`, `validation_counts`) and keeps the
top-level `state`, except a refusal close-out's: a `runner_rejected` or `executor_refused`
`state` goes, with the `reason_code`, `runner`, `runner_key` and `refused_batch` written beside
it (`status_file.REFUSAL_STATES`, `REFUSAL_KEYS`), since `status` reads them as the refusal
of the run it is polling. A review phase's checkpoint also records the review round,
`round_id`, which its fan-out stamps and every file of the round records; `claim_run` keeps it, so a
detached `--reaggregate` continues the round on record
([round identity](review-results.md#round-identity)).

Its reserved keys are `{step, models, models_done, models_succeeded, models_total, interval_s, tasks_generated, notes, subagents, write_subagents, sandbox_bypassed, sandbox_fallback}` (`status_file.py`'s module docstring is the
prose contract, `schemas/status_file.schema.json` the declared one, held to every writer by
`tests/test_result_contract.py`: see [Result schemas](dispatcher.md#result-schemas)): `models` is the fan-out's per-model map keyed by spec,
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

### One writer per phase dir

**A review run (review-plan, review-tasks, review-code) holds `{phase}/.lock` from its entry to
its exit, foreground or detached** (`detach.PhaseRun`, code health U81 round 3). The phase mains
take it before they read `state.json` or any file of the phase (review-code's reaggregate after
its output-dir check), so every step that can stamp a round, regroup, validate, reaggregate,
auto-heal or complete the phase runs under it, and a context manager at each entry
(`run_review_orchestration`, review-code's `main`) releases it when the run returns or exits. A
second run is refused at its entry, before it reads or writes anything: exit 3, `another run of
this phase is in progress (pid …, alive): … owns this phase dir`, with the holder's status and
abort commands. Before it, a foreground run took no lock and only the round claims
(review-results.md, "Round identity") stood between two runs: Codex found five interleavings in
which the second run restored a superseded round, completed the other's phase, retired its
verdicts or sealed a report to them.

- **Re-entrant within one process.** The OS refuses a second descriptor even to the holder's own
  process, so `PhaseLock` lists every lock this process holds (`process_control.held_phase_lock`)
  and a take of one already held uses it as it is; its taker releases it. That is the case under
  a composite's step lock (`improve_workflow.ImproveWorkflow.run_mode`) and a commit front door's
  lock (`commit_front_door._take_lock`).
- **A `--detach` launcher decides at its entry**, before any pre-flight write (`backup.md`,
  `state.json`, a task file, the checkpoint; `detach.take_for_launch`). A free lock is taken and held
  to the boundary. Against a live holder: `--if-not-running` re-attaches to the detached run whose
  label and checkpoint name it, deciding from those alone (it writes nothing); `--force` stops that
  run (below); anything else refuses at exit 3. `detach.launch` runs the same gate for a mode that
  takes no entry hold, so the lock is held from its first check to its worker.
- **A live holder's lock is never broken.** `--detach --force` asks the detached run that holds the
  dir to stop through the abort path (`detach.abort_run`, what `status --abort` runs), then waits on
  a take of the same lock, at most `FORCE_STOP_WAIT_S` (twice the abort's SIGTERM-to-SIGKILL
  window): getting the lock is the proof that the old worker is gone. It waits only after an abort
  that confirms every process of the run stopped (`detach.CONFIRMED_STOPS`: `aborted`,
  `not_running`). The lock dies with the worker, but a provider CLI or its child that outlived it
  can still write in the dir, so every other outcome (`needs_force`, `survived`, a teardown that
  could not confirm every process stopped, any value added later) refuses at exit 3, launches
  nothing, keeps the run's records, and names the cleanup (`status --abort`, `util runs --sweep`).
  A holder that outlives the wait, or one `--force` cannot stop (a foreground run, a composite's
  step, another launcher, this process), refuses at exit 3 naming it. `process_control.break_lock` frees only a lock it can take,
  so a dead holder's lock is taken over as before and a live one never is; the lock file is never
  unlinked.
- **A teardown not confirmed refuses every entry, the lock free or not.** A stopped run's lock
  frees with its worker, so a later invocation would take it at once while what outlived the
  worker still writes. Each entry therefore reads the earlier run's cleanup record once it holds
  the lock and before its first write (`detach.refuse_unconfirmed_cleanup`): a foreground run, a
  launcher (the free lock and `--force`'s alike), a detached worker, a re-entrant take of a held
  lock, a commit front door's lock and `RunHandle.start`. The record is `.status.json`
  `run.cleanup` and the run's registry entry (`.multi-llm/runs/<run_id>.json`), in the shape the
  kill paths write (code health U77; `detach.unconfirmed_cleanup`). **The registry decides**: the
  kill paths keep a run's entry until its teardown is confirmed (with the same `cleanup` when a
  kill could not confirm every process stopped) and remove it once a sweep or an abort confirms
  them stopped, but may leave the checkpoint's value behind. So only entries of this dir or of the
  checkpoint's run are read (another dir's entry, malformed or not, never is), and the
  checkpoint's value counts only while that run's entry is kept; with no entry it is stale, warns
  in one line and goes on. No record, or `state: complete`, goes on; any other string `state`
  (`pending`, `incomplete`, one added later) refuses at exit 3, names the cleanup
  (`status --abort --run-id`, `util runs --sweep`) and keeps the records. Any other shape (not an
  object, a `state` missing or not a string), and a read that raises at all, refuses at exit 3
  as a malformed cleanup record naming the file, after releasing a lock the entry took.
- **The lock crosses the detach boundary held (POSIX).** `flock` belongs to the open file
  description, so `detach.launch` passes a duplicate of its descriptor to the worker
  (`process_control.spawn_detached_run(keep_fds=…)`, numbered in `MULTI_LLM_PHASE_LOCK_FD`) and the
  worker adopts it on its first line (`detach.prepare_worker_output`, then
  `process_control.PhaseLock.adopt` at its entry, which checks the descriptor is `{phase}/.lock` by
  `fstat` against `stat` and makes it close-on-exec). The launcher only ever closes its own copy,
  never unlocks it (`PhaseLock.disown`), once the handshake resolves or the spawn fails: a worker that
  adopted the lock keeps it, and one that died took its copy with it. No other process can take the
  lock between the launcher's pre-flight and the worker's (`tests/test_review_round.py`,
  `TestTheLockCrossesTheDetachBoundaryHeld`). The lock's descriptor is otherwise never inherited: a
  provider CLI must not keep a phase locked.
- **Windows keeps the old hand-off**: an `msvcrt` lock belongs to the process that took it, so the
  launcher releases its hold right before the spawn and the worker takes it again. Another run can
  take it in between; the worker then refuses at exit 3 and leaves `{phase}/.worker-refused-<run_id>`
  (written only by a detached worker, under its own run id), and the launch reports that refusal at
  exit 3 instead of a run (`detach._await_handshake`).
- **The detached worker** adopts that lock (or takes it) at its entry, so its pre-flight and the
  secondary guard run under it; `RunHandle._acquire_lock` takes that hold over instead of taking the
  lock again, and releases it in `finish`, after the terminal `run` record. An aborted worker holds
  it until `os._exit` (`RunHandle._on_signal`), so the kernel drops it after every thread has ended.
- **Lock release is a run's last step.** Every thread a review phase starts goes through one helper,
  `review_orchestrator_base.phase_thread` (`detach.run_phase_work`): the fan-out's provider calls,
  with the logs they write and the earlier round's result they set aside or adopt, the slot guard's
  probes and snapshots, and the validation and salvage dispatches (`dispatch_in_round`). None of
  those modules calls `asyncio.to_thread` itself (`tests/test_review_round.py`,
  `TestACtrlCHoldsTheLockUntilTheFanOutEnds`). A Ctrl-C does not stop such a thread: the
  cancellation ends the coroutine, not the thread. `PhaseRun`'s exit therefore stops the live
  provider CLIs (on an interrupt, and again every `PHASE_WORK_RESWEEP_S` while it waits, for a CLI
  started after the first sweep) and waits for every listed call before it releases the lock; a
  call given up before any thread started it is never run and not waited for. Past
  `PHASE_WORK_JOIN_S`, or when the wait itself is interrupted, the lock is held until the process
  exits (`PhaseLock.hold_until_exit`), never released early.
- **Late results are bound to their round.** The dispatch runs under `review_round.results_bound_to`
  its claim, and `task_result.persist_result` writes a result file only while the checkpoint
  still records that round, deciding under the `.status.lock` mutex held across the write; a result
  of a replaced round (or one whose round cannot be checked) is logged (`… dropped, not written`)
  and dropped. Defence in depth behind the lock; batch files a host's own subagents write arrive from
  outside the run and are judged by content (review-results.md, "Round identity").
- **Fail closed, never stuck.** A refusal is retried `ENTRY_LOCK_ATTEMPTS` times
  `ENTRY_LOCK_RETRY_S` apart (a `status` poll probes the lock by taking it), then refuses. A lock
  file that cannot be opened or locked refuses every lane; nothing proceeds unlocked. A holder that
  died released its lock with its descriptor, and the label it left is overwritten; a payload
  naming a dead process in front of a live holder breaks only when `break_lock` proves the lock
  free, as for the worker.

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

**An owned child's kill.** `kill_tree(pid, child=proc)` is the POSIX kill of a `Popen` child this
process still holds unreaped: the executor's timed-out runner and the timed-out verify command. It
rests on one kernel rule. POSIX returns a pid to the system only when its process is waited for,
and only the parent waits. A fork never takes a pid that is still in use, and a group or session
number is always its creator's pid. So while the child is unreaped, its pid and the number of the
group it leads name that tree and nothing else.

That holds only under a lifecycle contract, and the caller keeps it, not the kill:

- the caller holds the only waiter on `proc`, and waits on it only after `kill_tree` returns;
- nothing in the process reaps children it does not name: no `os.wait()`, `os.waitpid(-1, …)`,
  `waitpid(0, …)` or `waitpid(-pgid, …)`, `wait3`, `wait4`, `waitid(P_ALL | P_PGID, …)`;
- children become zombies: SIGCHLD is neither `SIG_IGN` nor flagged `SA_NOCLDWAIT`, either of
  which has the kernel reap every child at its exit. Every process entry (`multi_llm.main`, which
  the detached worker runs too, `process_control.enter_detached_worker`, and each orchestrator run
  as its own `__main__`) sets SIGCHLD back to `SIG_DFL` when it inherited `SIG_IGN`, which an
  `exec` keeps (`process_control.restore_sigchld_default`); `subprocess` needs that anyway, since
  under `SIG_IGN` every `Popen` reads its child's exit as 0.

`tests/test_no_child_reaper.py` fails any plugin source (`skills/multi-llm/`, `packaging/`) that
reaps an unnamed child or sets SIGCHLD's disposition, bar `signal.signal` back to `SIG_DFL`: by
name, alias, keyword, or by number to a call that sets one. It reads the syntax, so a signal held in a variable is outside it. The detach
double-fork's `waitpid(intermediate, 0)` names its own child and is allowed. The kill itself reads
the contract right before each signal: the child with `waitid(…, WNOWAIT)`, the disposition with
`sigaction(SIGCHLD, NULL, &old)`. `signal.getsignal` can only refuse: it sees a `SIG_IGN` set
through Python's `signal` module, never `SA_NOCLDWAIT` or a native library's setting, so its
`SIG_DFL` proves nothing. Once the child or the disposition is known to fail, the kill sends
nothing more. Once the disposition cannot be read, the owned path sends nothing more and the rest
of the kill is the one by number below. A reading cannot hold the anchor: another thread can reap
between it and the signal. So the check narrows a breach of the contract, and it does not close it.

Under the contract the kill counts the child's zombie as exited and never reaps it, so its `Popen`
reads the real exit status. After the SIGTERM it waits out the grace, or less once a look at the
process table sees nothing of the group run. A look is not atomic. A member can fork after the
table's names were read and exit before its own row was, so the new child is not listed. So no look
alone ends the kill. The group always gets one SIGKILL by number, which with only the child's
zombie left changes nothing. On Linux a process with a SIGKILL pending can fork no other:
*copy_process* abandons the fork under the same locks `killpg` takes. So a look after it that sees
nothing run proves the group gone, and only then is the verdict `killed`; otherwise it is
`survived`. A member this process may not signal (another user's) is outside that proof, and so is
a process that joins the group from elsewhere in its session after the SIGKILL. Only the group whose
number is the child's own pid is anchored. Every other target (a detached worker, a pid or group
read off disk) is known only by number, and `kill_tree` treats it by its identity rules instead.

**Where the owned kill runs.** Its proof has two halves, each checked for one platform only, so
`_sigaction_abi` gates it on both:

- the kernel's: the group SIGKILL leaves no member forking. Linux's *copy_process* publishes a new
  process under *tasklist_lock* held for writing, after checking for a pending fatal signal, and a
  group signal holds that lock for reading, so a fork either finishes before the SIGKILL and gets
  it, or is abandoned. XNU's `killpg` collects the group's pids, drops the lock, then signals, so a
  member can finish a fork the SIGKILL never reaches. Current FreeBSD looks sound (*pg_killsx*
  coordinates with fork), but it has not been run here: a later extension, not a default.
- the C library's: SIGCHLD's disposition is read through `struct sigaction`, whose layout is the
  C library's and the ABI's. glibc's generic layout (a handler, a 1024-bit mask, an `int` of
  flags, a restorer; `SA_NOCLDWAIT` 2) holds for x86_64, i386 and aarch64 programs, which override
  none of it, and musl uses the same layout and value there. MIPS puts the flags first; Bionic
  (Android's) does on 64-bit Arm. The ABI is the program's, not the kernel's (`uname` says
  `aarch64` under a 32-bit Arm userland), so it is read off the dynamic loader the executable
  names: `ld-linux-x86-64.so.2`, `ld-linux.so.2` or `ld-linux-aarch64.so.1` for glibc, which must
  also name itself through `confstr(_CS_GNU_LIBC_VERSION)`; `ld-musl-<arch>.so.1` for musl, which
  is its libc too and refuses that name. The loader is paired with the kernel's machine, since
  Alpha, SPARC and SuperH call their glibc loader `ld-linux.so.2` as i386 does. A static program
  names no loader and is not named. Android is refused by `sys.getandroidapilevel` on Python
  before 3.13, which reports `linux` there, and by `sys.platform == "android"` from 3.13.

| Platform | Kill of an owned child |
|----------|------------------------|
| Linux, glibc or musl, x86_64, i386 or aarch64 programs | the owned kill |
| Linux, any other C library (Bionic, uClibc), a static program, or another ABI (32-bit Arm even on an aarch64 kernel, x32, MIPS, RISC-V, PowerPC, s390x, …) | by number; with Python's own pidfd calls the identity kill, where the executor's timeout kill dates its child (below) and an undated one is `unresolved` and gets nothing |
| macOS, FreeBSD and the other BSDs, glibc on another kernel | by number |
| Windows | its Job Object (`kill_tree`'s Windows path) |

By number, the `child` argument is ignored and the kill is the one below for a target known only
by number. Without pidfds the child's zombie still counts as running, so a timeout kill sits out
the grace and the `_KILL_CONFIRM_S` (5 s) SIGKILL window, then reports `survived`. Slow and
pessimistic, never aimed at a number the proof does not cover. The executor's timeout kill then
confirms the exit itself (`process_control.own_child_stopped`): the child has exited (its zombie,
by `waitid(…, WNOWAIT)`, else its `Popen`'s own `poll`) and its group runs nothing, and only then
is that `survived` its `killed`. With pidfds (the identity kill below) the zombie counts as
exited, as everywhere else a process is known by its pidfd; there a child whose anchor cannot be
read (an ABI `_sigaction_abi` does not name) is not held through it. The executor's timeout kill
dates its runner instead (`process_control.dated_own_child`): the start time from the child's
`/proc` row, read while that row names this process as its parent and the `Popen` has no exit
status, so the pid is still reserved and the row is the child's; the kill holds it by that, no
ABI needed. A child whose ownership cannot be established (another parent, already reaped,
SIGCHLD known ignored, no row) goes undated: nothing is sent, and the verdict is `unresolved`.

**A target known only by number.** Every other POSIX kill: `kill_tree` without an owned child
(`status --abort`, the dead-run close-out, `util runs --sweep`, the failed-handshake teardown, the
owned kill's hand-over, every kill off the platforms above) and the Lane A sweep of a CLI's
snapshotted groups. A number names whoever holds it when the signal lands. A pid is handed out
again once its process is reaped, and a group's number once no process uses it as a pid, a group
or a session. A process can also leave its group and join another of the same number, made again
inside its session, without ever being reaped, so no process proves what a group is across a read
of it. Which kill runs depends on two things the process can read about itself:

| Platform | Kill of a target known only by number |
|----------|----------------------------------------|
| Linux with working pidfds (5.3+) and a readable `/proc`: Python's own calls in any ABI, the `syscall()` route in an ABI `_sigaction_abi` names | the identity kill: only processes identified one by one are signalled, never a number |
| macOS, the BSDs, a Linux without pidfds or `/proc`, a Python without its own pidfd calls in an ABI `_sigaction_abi` does not name | by number, as before (the residual risk below) |
| Windows | its Job Object, unchanged |

*The identity kill* (`process_control._identity_kill`, `_identity_sweep`) signals only processes
identified before it, each by its pid and start time, and each through a pidfd verified to hold it:

- **What is signalled.** The target pid with its recorded start time, and the *members* the caller
  recorded the same way: for a detached run, every process its worker started (below); for the
  Lane A sweep, the CLI's snapshotted descendants. Nothing else, and never a group: a process that
  is in the target's group but was not recorded gets no signal, however it got there.
- **A Lane A snapshot adopts only under a verified root.** The CLI's children are walked from its
  pid, so only while its row, in the same scan, has the start time the CLI was registered with
  (`LiveCliRegistry.snapshot_for`, `refresh`). `returncode is None` alone proves nothing once the
  kernel reaps children itself: the pid may name another process by then, and its children are
  not the CLI's. A root with another start time, no row, or none registered adopts nothing new
  (what an earlier verified look snapshotted still counts), and its sweep reports
  `-<pid>: unresolved` (the key of a member with no group): nothing found under it now was
  signalled. Off the identity kill the same rule holds for the kill by number's sweep.
- **No identity, no signal.** A pid alone is never held: whatever has it now may be another
  process. A target with no recorded start time is held only through the owned child that anchors
  its pid (`_hold_owned`: the `Popen` this process holds for it, under the anchor of "An owned
  child's kill", read before the pidfd opens and again after), which needs an ABI whose SIGCHLD
  disposition can be read (`_sigaction_abi`); a dated target needs none. With neither, the target gets nothing, its
  recorded members are still held by their own identities, and the verdict is `unresolved` (`gone`
  when no process has the pid at all). That covers the callers that had none: `status --abort
  --force` keeps the recorded start time (it overrides a probe that could not read it; an identity
  still unreadable at the kill is `skipped`, nothing sent), and the launcher's teardown of a failed
  handshake kills the worker by the start time it read the moment the spawn reported the pid
  (`process_control.detached_worker_start_time`): the intermediate spawn that made the worker's
  session is still the launcher's unreaped child then, so no other session can have its number,
  and a row in that session is the worker's (or a process it started). A worker whose start time
  could not be read so is not signalled, and the launch error says it was not confirmed stopped.
- **Holding a process** (`_hold_process`): open a pidfd on the pid, read its `/proc/<pid>/stat`
  row, then ask the pidfd whether its process is still there (`pidfd_send_signal(fd, 0)`; a
  zombie is). A pid is not handed out again until its process is reaped, so a "yes" after the read
  means the row, start time and group included, was that process's. A "no" retires the target: its
  pid may already name its replacement, and nothing is read from that row. A recorded start time
  must match, or the target is gone. Every later signal goes through the pidfd, which reaches that
  process or fails with `ESRCH`, and whether it has exited is the pidfd's answer too: it polls
  readable once its process is a zombie or reaped (`_Pidfds.exited`, 5.3+ like the calls), so no
  `/proc` row is needed for it. Python's own `os.pidfd_open` and `signal.pidfd_send_signal` are used
  where they exist; the standalone builds `uv` installs are compiled without them, so on the ABIs
  `_sigaction_abi` names the calls go through the C library's `syscall()` (434 and 424, the generic
  table's numbers, which x86_64, i386 and aarch64 programs share). `_pidfd_api` opens one on this
  process once, and a kernel or sandbox that refuses it leaves the kill by number.
- **The kill**: SIGTERM to every held process, the grace, SIGKILL to every held process not known
  to have exited, the confirmation window (`_KILL_CONFIRM_S`).
- **Reading the groups, after** (`_IdentityUnit.report`). The recorded group, each member's
  recorded group, and every group a held process was seen in during the kill (one it moved to
  included) are read for processes that are none of the held or recorded ones (zombies aside),
  until none is left or the confirmation window ends: what a held process's own end takes with it
  (a CLI stopping its tool children on SIGTERM) gets that long to go. Anything still there makes the
  verdict `unresolved`, never `killed` or `gone`. A group learned only from a record, with no held
  process seen in it, is not read when its number is the pid of a live process that is none of the
  recorded ones: a group's number is the pid of the process that made it and is not handed out
  while the group has a member, so that group was made by that process. This keeps a long-gone
  CLI's group, or a group from before a reboot, from reading as a false `unresolved`.
- **Unknown is not gone.** A held process whose exit its pidfd cannot answer and whose row cannot
  be read, a member whose identity cannot be read, a process table that cannot be listed for the
  read, or a record that may miss a process (*members_complete* false): each makes the verdict
  `unresolved`. A target whose own identity cannot be read is `skipped`, with nothing sent.

*What a detached worker records* (`process_control.record_cli_identities`, from `begin_worker_run`):
every process it starts through `popen_tree`, the one spawn every provider run goes through (Lane
A's CLI in the worker's group; the executor's task in a session of its own), as JSON lines in
`orchestrator-run.<run_id>.clis` in its phase dir. Each launch writes an intent before the spawn
and the process's pid, start time and group after it (or `failed`). Only where the identity kill
runs; elsewhere a spawn is what it was. Read back with `read_cli_record`.

The identity comes from one `/proc` row, kept only while that row shows the process is still this
one's child (`_record_spawned`). `popen_tree` returns before anything waits on the child, so its
pid can name another process only once the child was reaped behind its `Popen` (SIGCHLD ignored,
`SA_NOCLDWAIT`), and whatever took the pid then is another parent's. A row that is not this
process's child, or that cannot be read, is written as unverified (`"start": null`): the intent
stays unanswered, the record incomplete, and nothing is signalled by it. The launcher reads the
detached worker's start time under the same rule (its session is the unreaped intermediate's).

A record is its run's alone (`_open_record`), on write and on read: the phase dir is opened with
`O_DIRECTORY|O_NOFOLLOW` and must be a directory this user owns; the record is opened relative to
that descriptor with `O_NOFOLLOW` (`O_CREAT|O_APPEND` to write; an identity line never creates it)
and must be a regular file of this user's with one link. Both opens are `O_NONBLOCK`, so a FIFO or
device in the record's place is refused by that check instead of waited on; the descriptor is made
blocking again once it passed. A symlink or hard link to another run's record, a FIFO, or a file
another user planted, is refused: a launch then does not start, and a read is incomplete. Every line carries the
run id (`"run"`); a reader ignores a line of another run (a warning on stderr names how many) and
reads a line that names no run as a torn one.

- an intent that cannot be written refuses the launch (`LaunchRefused`, an `OSError`; nothing
  started);
- an intent with no identity (the worker died between the spawn and the second line, or the line
  could not be written), a short write (ended by a newline so later lines stay whole), a line that
  does not parse, a file that cannot be read: the record is incomplete, and the kill's verdict
  `unresolved`;
- no file: nothing was started.

The window this leaves: a process exists before its identity is written, and what it starts before
the close-out reads the record is in its group, which the read covers but never signals. Execution
is gated on the intent, which is cheap; gating it on the identity would need a handshake with the
child before exec.

*The launch gate* (`process_control.close_launch_gate`). Every teardown of a run's processes closes
it before its kill, whether or not the worker still runs (`detach.stop_recorded_run`: an abort,
the retry of a cleanup not confirmed, the dead-run close-out, `util runs --sweep`'s close-out and
prune): a `gate: closed` line, written under an exclusive lock on the record (`wait_for_fd_lock`
on the record's own descriptor); a phase dir that no longer exists reads as closed, with nothing
recorded. `popen_tree` takes the same lock, reads the record and writes its
intent in one hold, so a launch either got its intent in first, and the abort's read under that
lock sees it, or finds the gate closed and is refused (`LaunchRefused`: "run … is being aborted",
nothing started, no spend). Once the worker is stopped the teardown reads the record again
(`detach._kill_after_the_gate`): an identity written since its first read (a launch past the gate
whose process came up during the kill), or a first read that could not vouch for every launch,
makes it kill again with every identity recorded by then. The second kill does not stand for what
only the first saw: the teardown's verdict is the worst of the second kill's and of what the first
still owes (`process_control.KillObligations`: each group it found running something
unidentified, each identity it could not confirm stopped), each re-checked on its own, read-only,
once the second kill is done. A group is cleared only once it runs nothing but zombies, an
identity only once no process runs as it; a first verdict worse than anything it named (a kill
that raised) stands. So a CLI that moved to a group of its own, left a tool there and exited keeps
the teardown `unresolved`, though the second kill no longer sees that group. Only a reread after
the gate closed can confirm the cleanup `complete`. `util runs --sweep`'s prune makes one kill:
the close-out of the run's still-open record takes its verdict rather than killing again. An intent still
without an identity, or a gate that could not be closed (the record refused as above, its lock not
had in `DEFAULT_MUTEX_TIMEOUT_S`), leaves the verdict `unresolved` at best.

The worker keeps refusing launches through its own teardown: from the start of `RunHandle.finish`
and of its signal handler until it exits (`process_control.refuse_launches`), `popen_tree` raises
`LaunchRefused` without touching the record ("run … is ending"), so an executor thread scheduled
between the handler's `finish()` and `os._exit` starts nothing. The record stays named, so a
launch already past its intent still writes its identity for the abort's reread.

What it does not prove, and what the verdict says then:

- Every process nobody recorded is outside it and gets no signal: a tool child of a provider CLI,
  a process forked during the kill, everything left in a crashed worker's group when its record
  names nothing alive. Anything of those still running in a group the kill reads makes the verdict
  `unresolved`; one that moved to a group the kill never learned of is not seen at all. Before
  this, the kill by number signalled the whole group and reached them.
- A group whose number was made again by an unrelated program, and that a held process of the
  target's joined, reads the same way: `unresolved`, a false alarm, never a signal.
- Two processes given the same pid within one `/proc` clock tick have the same start time; a
  process this one may not signal (another user's) is outside all of it.
- What a teardown's first kill could not confirm is kept for that teardown's own second kill, not
  past it: a later retry (`status --abort`, `util runs --sweep`) knows the run's recorded
  identities and groups, not a group an earlier teardown saw a member move to. The earlier
  teardown's `incomplete` cleanup is what says to check by hand.

*By number* (`_posix_kill_tree`, `_posix_kill_group`, `_GroupTarget`) is the kill as it was before
the identity kill, unchanged: the pid's start time read with `probe`, then `killpg` on the
recorded group, or on the pid's own group, and `kill` on the pid alone; a recorded group whose
leader is gone is signalled by its number alone. Every check precedes its signal by a read, so a
process can exit and be reaped, and its pid or its group's number be handed to another program,
between the two: that program gets the signal. It is a narrow window (a process must die and the
kernel must hand its number out again in it), but it is not closed off Linux's pidfds. The
`TestAProcessReplacedAfterTheLastCheckGetsNothing` and `TestAGroupMadeAgainDuringTheReadSignalsNothingInIt`
scenarios in `tests/test_signal_identity_posix.py` pin both kills.

**When not everything is confirmed stopped.** `skipped` (an identity that could not be read: no
signal), `unresolved` and `survived` all mean some process of the run may still run. The callers
say so rather than report the run cleanly stopped, and keep what they need to try again:

- `run.cleanup` on the run's record: `{state, verdict, pid, pgid, checked_at}`, `state` one of
  `pending` (an abort's kill under way), `incomplete`, `complete`.
- The run's registry entry carries the same `cleanup` (`run_registry.mark_cleanup`), and it and the
  CLI record are kept while it is not `complete`: a new run may claim the phase dir and replace its
  record, and the entry still names the old run's pid, start time and group, and its record by run
  id.
- `status --abort` marks the record `pending` before its kill. A worker that finishes inside it (its
  SIGTERM handler) sees that, and leaves its registry entry and CLI record to the abort, which
  retires them once its kill is confirmed. `INCOMPLETE` (or `REFUSED TO DIE` for `survived`) and
  exit 3 otherwise; the run is closed (`aborted`) either way.
- The dead-run close-out (`detach.close_out_dead_run`, reached from `status`, an abort of a run
  already gone, `status --sweep` and `util runs --sweep`) records the same `run.cleanup` beside
  `failed`. `status` shows a `cleanup: INCOMPLETE` (or `PENDING`) line with the abort command that
  tries again; `status --sweep` prints an `INCOMPLETE` line for each run whose cleanup is not
  confirmed, now or from before, with that command, and exits 3 (`needs-human`).
- Every close-out and retry is held to one run id: the record is read under the phase dir's
  checkpoint lock (the one `claim_run` takes), its run id checked, and that record's pid, start
  time and group are what the kill targets; a record that is another run's is neither signalled
  nor written.
- `util runs --sweep` closes out an entry whose worker is gone, or whose `cleanup` is not
  `complete`, from the entry's own identities (`detach.close_out_entry`), never from the phase
  dir's current record. A kill that was `survived`, `unresolved`, `skipped` or raised leaves the
  entry `needs_attention` (the kill's verdict in the row), and the sweep exits 3. Another abort or
  sweep tries again; one that confirms every process stopped writes `complete`, and the entry and
  the CLI record go. A record that may miss a process stays incomplete for good: check by hand,
  then delete the entry's `.multi-llm/runs/<run_id>.json`.

- The executor's timeout kill of its runner (`subagent_executor._kill_timed_out`) returns its
  verdict. Only `killed` and `gone` confirm the runner stopped (the kill by number's `survived`
  over the child's own zombie becomes `killed` once its exit is confirmed, above). Every other
  outcome is never classified as killed: `survived` (still running after the SIGKILL window), `unresolved` or
  `skipped` (not everything identified, and what was not got nothing), a kill that raised
  (`None`), any other word. The `timeout` error says it may still be running
  and names its pid, a `WARNING` line names the pid and group, and in a detached worker
  `detach.note_left_running` writes `run.cleanup` `incomplete` with that pid. `finish` then keeps the
  CLI record, which holds the runner's verified identity, and the registry entry; `status` and
  `status --sweep` report it, and an abort (or `util runs --sweep` once the worker is gone) stops the
  runner by that identity. A foreground run records no launches, so the warning and the error are
  all there is.
- The write scheduler's timeout kill of `driver.verify_command`
  (`WriteBatchScheduler._kill_verify_tree`) keeps the same rule, through the same
  `_unconfirmed_kill`, and the same `detach.note_left_running` record; its output's closing note
  names the verdict ([write scheduler contract](write-scheduler.md#write-scheduler-contract),
  "When the verify command times out").

These verdicts are the same everywhere, the kill by number's included: a kill that could not
confirm the run's processes stopped is never reported as a clean stop, since a false `aborted`
can leave a paid provider CLI running unseen. Off the identity kill only the *signalling* is the
one it always was (the six by-number definitions are unchanged); its callers' verdicts changed with
the rest. So an abort whose first probe reads the worker and whose kill then cannot (`skipped`) is
`INCOMPLETE`, exit 3, with its record and registry entry kept, on macOS as on Linux, where before it
was `aborted` with both gone.

## Advisory guards

Code that decorates a command rather than doing its work (a `status` block, a hint line, a
recorded route, the launch roster) catches broadly and falls back, so one unreadable record never
fails a poll or a launch. Such a catch hands what it caught to
`utils/advisory.note_swallowed(label, exc)`, or wraps its block in `advisory(label)`:

- by default nothing is printed, exactly as with a bare `except Exception`;
- `MULTI_LLM_DEBUG=1` writes `multi-llm debug: swallowed in <label>:` and the traceback to stderr.
  Stdout, the `MULTI_LLM_RESULT` line and the exit code do not change. The Codex tool server does
  not forward the variable: run the dispatcher command in a shell to use it;
- `MULTI_LLM_STRICT_ADVISORY=1` re-raises, **only inside a pytest process** (`"pytest"` in the
  process's own `sys.modules`). Anywhere else the variable is ignored and the default holds: a
  detached worker a test spawns inherits it but `execve`s a fresh interpreter that imports no
  pytest, and a host or a shell that exports it gets the silent guard too, so strict mode never
  reaches a real run. `tests/conftest.py` sets it for the files in `STRICT_ADVISORY_FILES` (the
  tests of `run_status`, `detach`, `next_phase` and the next-step emitters). A test there that
  feeds a malformed input on purpose takes the `lenient_advisory` fixture, which restores the
  silent guard and yields the labels that swallowed something, and asserts that list. A swallow
  in a child process is neither raised nor recorded.

Rules for a site:

- What a healthy run meets (no `write_tasks.json` before any emit, a host id the capability data
  does not list) gets a narrower `except` before the broad one and stays silent in every mode.
  The broad clause is for what nobody expected.
- Signal handlers, `atexit` and the close-outs (`RunHandle.finish` and what it calls,
  `close_out_dead_run`, `abort_run`, a failed handshake's kill) keep their plain silent excepts.
  They must never raise, strict mode included, and a signal handler prints only through
  `stream_bootstrap._FdWriter`.
- The label names the site the way a reader searches for it: `"status: native implement view"`.
- The debug report goes through `sys.stderr` and falls back to fd 2 through `_FdWriter`; it never
  raises.

`utils/run_status.py` and the `utils/status_*.py` modules split out of it, `utils/detach.py` and
`utils/next_phase.py` use the guard; the broad excepts elsewhere in the tree are still silent.

## Atomic file writes

Replace a file whole through `utils/atomic_io.py` (`atomic_write_bytes`, `atomic_write_text`),
not a private temp-then-`os.replace` copy. The guarantees, each pinned by an injected fault in
`tests/test_atomic_io.py`:

- The temp file sits beside the target (the same volume, so the rename is atomic on Windows too) and
  is created with `O_EXCL`: a file or symlink already at its name is never opened, nor removed.
- While it holds bytes it is owner-only, and never readable more widely than the file it replaces.
  Replacing an existing file, or with an explicit `mode`, it is created owner-only (0600). A new
  file with no `mode` is created empty with the umask's mode, as `open()` would, read back with
  `fstat` (so the process umask is never set, which would race every other thread creating a
  file), and narrowed to owner-only before its first byte. Either way it gets its final mode only
  once written.
- The mode is applied before the `fsync`, then `os.replace`, then on POSIX the directory is
  flushed. Any exception, `KeyboardInterrupt` included, closes the descriptor (a failed `os.fdopen`
  included), gives a temp file made read-only by its final mode its owner write bit back (Windows
  deletes no read-only file, and refuses to replace a read-only target), removes it, and leaves the
  old bytes.

A call site states only how it differs, through a parameter:

| Parameter | Default | Who changes it |
|-----------|---------|----------------|
| `keep_mode` | on: an existing file's bits carry over | no caller turns it off |
| `mode` | none | 0600: executor results, ask records, the version cache, `state.json`, shims; codex's and cursor's config keep the mode they had |
| `fsync` | on | off for `state.json` (rewritten dozens of times a run, under its lock) and doctor's version cache (inside the startup budget) |
| `follow_symlinks` | off: a symlink is replaced by a file | `update_plan_tasks.replace_file`: a plan or task list kept elsewhere and linked in stays linked |
| `newline` (text) | `\n` | `os.linesep` where a writer has always written what `write_text` does (`update_plan_tasks`, `review_round`, the HTML report, ask's records) |
| `temp_prefix` | `.<name>.` | where something matches leftovers by name (`.state_`, `.metrics_`, `.tracking_`, `.multi-llm-shim-`, `.multi-llm-trust-`) |

`state.json` itself goes through `utils/state_file.py`: `write_document` (the one writer),
`read_document` / `read_document_strict` (the raw readers; a raw read migrates and writes
nothing), `locked` (the state lock, still importable as `metrics.state_file_lock`) and `update`
(read, change, write under the lock; nothing is written when the body raises or changes nothing).
The strict reader raises `StateDocumentError` (a `ValueError`) for a file that is not UTF-8 JSON
holding an object; each command turns it into its own refusal (`util batch-done` exits 3 with
`... is not a JSON object`, `util metrics` exits 1 with `Error:`). `StateDoc` lists the top-level
keys the skill writes and `schemas/state_file.schema.json` lists the same properties;
`tests/test_state_file.py` compares them, and checks the keys `StateManager`'s writers put in a
saved file against both.

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
validation tasks build their round with `review_validation_round`, which records
`orchestrator=<the phase's own script>`: the fan-out call once omitted it (measured 2026-09-14),
`RoundProvenance` defaulted to review-plan's, and a failed review-tasks run told the session
model to re-invoke `review-plan --reaggregate`.
