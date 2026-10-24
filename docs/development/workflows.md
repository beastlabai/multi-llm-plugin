# Workflow orchestration

Read when changing task generation, full or composite workflows, ask mode, or implementation bookkeeping.

Back to the [development guide](../../skills/multi-llm/AGENTS.md). These notes retain detailed contracts and regression history from that guide. Paths beginning with `docs/`, `packaging/`, or `plans/` are repository-relative; other implementation paths are relative to `skills/multi-llm/`. Dated incidents explain the rules; verify version-specific claims against current code.

## Contents

- [Task generation](#task-generation)
- [Full workflow state machine](#full-workflow-state-machine)
- [Improve-plan and improve-code](#improve-plan-and-improve-code)
- [Ask mode](#ask-mode)
- [Implementation artifacts and state](#implementation-artifacts-and-state)

## Task generation

**`generate-tasks` is Python (`generate_tasks_orchestrator.py`), and it is the one phase whose subagent
sees the repository.** The sequence is: `ensure_prerequisites` (row 5, called — never reimplemented)
→ materialize **one** `task_generation` task into `{plan}/generate-tasks/task_generation_tasks.json` (the standard
envelope, so the executor needs no special case) → dispatch → validate → `update_plan_tasks.main(...)`
**in-process** → verify the outcome.

The task is `context_mode: workspace-read` with the pinned repo root as the only path its prompt names, and
it requires **both** `supports_workspace_reads` and the `agent-readonly` profile: a runner with
neither is refused *before dispatch* with exit 3, because a non-agentic runner does not fail on this
prompt — it invents a task list from the plan's prose, which every later phase then treats as ground
truth. Either invoker persists `task_generation_result.json` through the same `persist_result` leg
(`--exec-subagents` continues in-process; a native host writes the file and re-invokes
`generate-tasks <plan> --reaggregate`, the one follow-up printed).

That `--reaggregate` only ever *consumes* an existing result file — it re-dispatches nothing
and refuses when the file is absent — so `status`'s `resume with:` for a
failed `generate-tasks` run is the plain re-invoke (`run_status._RESUME_VERB_BY_MODE`, verb
`--yes`, the same command the phase's own retry hint prints); the review phases keep
`--reaggregate`, which does re-dispatch there (a detached worker's own hint replays its
`--internal-worker` as `--detach`, so it agrees, 2026-10-05); the four write phases
and `full` resume with `--resume`, the verb their own exit-3 block names
(`full` had no entry and fell through to `--force` until 2026-10-09) —
`--force` is the apply parsers' alias for `--yes`, under which the apply
base re-ingests every human decision instead of replaying the recorded ones (measured 2026-09-15:
the status line said `--force` while the log said `--resume`, and the
session model ran `--force` twice) — and their cost line counts batches from
`progress.write_subagents` instead of mentioning the reviewer panel — a run whose mirror records no
dispatched batch, under a phase not `in_progress`, prints a commandless
`no resume:` line instead (2026-10-05).

Measured 2026-09-14: the two lines disagreed, the session model followed the status line, and the
run ended on `--reaggregate found no task_generation_result.json`. `consume_document` writes the phase machine's terminal word
(`update_status(phase_dir, {"state": "complete"})` plus `report_progress(step="complete")`, exactly as the review phases do) on success and
touches it on no failure path: `claim_run` deliberately keeps the top-level
`state` across runs (a native host's `task_generation_pending` has to reach the
`--reaggregate` run), so without that write a successful retry read `phase state: subagents_failed`
under `run.state: done` (measured 2026-09-14, in the status view the opencode plugin hands the
session model).

It writes the **host-visible** pair at the same site: `detach.record_output_files(phase_dir, tasks=…, plan=…)` beside the two
`record_artifact` calls, and `report_progress(step="complete", tasks_generated=…)` so `status` prints the one
number this phase exists to produce, as `generated N tasks` — what the run produced, never a
`tasks N/N` pair, which read as a live tasks.md count and went stale once
apply-task-suggestions split a task (2026-10-04; a checkpoint holding the old pair under
`generate-tasks` renders the same) — with neither, the poll result named only the log, the
deliverable does not even live in the phase dir (it lands in the sibling `tasks/`),
and an opencode host globbed `generate-tasks/`, reported its four intermediate JSON blobs and
never mentioned `tasks.md`, the plan edit or the count (measured 2026-09-16).

`update_plan_tasks` then calls `state.refresh_plan_hash()` + `save()` after its own plan
write, because the StateManager it builds *after* that write reads the marker insertion as drift and
would record `plan_changed: True` about the plugin's own edit — which the next
`implement --resume` reports as "Plan has changed since last session".

`StateManager._load_or_create` is **set-only**: nothing but a consumer (`implement --resume`) or a
`refresh_plan_hash()` clears `plan_changed`, because setting it also re-stamps
`plan_hash`, so a clear-on-agreement branch erased a genuine human edit the moment any
intermediate load+save — `full`'s `_record_progress`, `review-tasks`,
`check_workflow_prerequisites` — ran between the edit and the resume (F1, 2026-09-16).

Stale residue is kept out at the other end instead, by every plugin-made plan edit re-stamping:
`update_plan_tasks`, both `_finalize` sites, `--mark-completed` when the phase
declares `edits_the_plan` (apply-suggestions only — for the two phases that never open the
plan, drift there is a HUMAN's edit) and `util finalize-tracking --plan-file` (the native leg's summary
back-reference). `run_status._progress_text` also renders the executor mirrors (`progress.subagents`,
`progress.write_subagents` — `<kind> <ok>/<total> ok`, then the non-ok statuses and the failed ids)
because a dispatch-only phase records no `step` before the end and the line printed
`(none recorded)` beside a mirror that said otherwise.

The write mirror carries one row per task and no `counts`/`ok`
summary, so `_mirror_text` counts the rows (`_write_mirror_as_counts`) and lists the write
outcomes (`applied 7, no-op 2`) — read only the summary keys, a finished nine-task run printed
`implement 0/9 ok` (measured 2026-09-16); it drops that breakdown when it collapses to a single
`applied N` equal to the `ok` count already stated (a clean run where
every batch applied), since repeating the same number under a different label read as a third,
smaller quantity (F14, 2026-09-16).

The mirror also carries `items_total`/`items_applied` — the suggestions or fixes
inside each batch, never the batch count itself, which is what let a 15-batch, 29-suggestion
apply-suggestions run mirror, and the model report, `15/15 ok, applied 15` (14 short) — rendered as
`<kind> <ok>/<total> batches ok, <applied>/<total> items applied` whenever the scheduler recorded both; the read path's mirror carries neither
key and keeps the bare `<kind> <ok>/<total> ok`.

Both `total` and `items_total` used to include the human task(s)
`human_task_strategy` skips, though they never dispatch — a clean 14-task implement run with one
deferred manual task mirrored `13/14 batches ok`, which reads as one short rather than as the
phase's own deferral; excluded from both denominators now (F9, 2026-09-16), and
`run_status._mirror_text` renders `skipped_task_ids`/`human_task_strategy` as
`deferred manual: <ids> (<strategy>)`, minus the skips the user chose, which the mirror names apart
(`declined_task_ids`/`declined_dependent_task_ids`, from `HumanTaskPlan.declined()`) and which render as
`skipped at your request: …` (fix round 14), and minus the tasks skipped with a deferred manual task
(`deferred_waits_on`), which render as `skipped (depends on manual T005): …` (fix round 17).

A native-leg implement, which writes no report and no mirror, is read back from
`state.json` by `status --view run` (and `--json`'s
`implement_tasks`) once it is started, completed included (`completed`, fix round
16): every task's bucket, the reasons of the failed and unrun ones, check findings and the answers
on record (`run_status.state_implement_view`, fix round 14).

An executor-leg implement run's block adds an `all tasks:` line beside its own run's
counts (`run_status.implement_task_tally`: every task of the decomposition, every run, one bucket each, from
`state.json`, through the one classifier it shares with `state_implement_view`,
`_implement_buckets`, fix round 17; `implement tasks:` in a pinned poll's
`progress_lines`, `implement_tally` under `--json`), and its mirror names a
check accepted with its findings apart (`accepted_with_findings_task_ids`) — fix round 16, run 6's resumes read
"1/1 batches ok" against seven tasks.

The mirror's dispatch counter — renamed `completed`, it used to count FINISHED tasks as
`dispatched`, so a single-task batch running minutes read `dispatched: 0` the whole
time, indistinguishable from a stall — is joined by `in_flight` (`{task_id, dispatched_at}`,
patched by `_begin_sweep` at dispatch time so it is visible before the task finishes),
rendered as `in flight: <task_id> (since <duration>)` (F10, 2026-09-16). And `write_batch_scheduler._mirror_status` writes the
phase-level `state` (`complete` / `subagents_failed` /
`running`) beside the run record, as the read phases do, instead of leaving
`phase state: (none)`.

It writes `progress.step` there too (`applying` on a mirror taken before the
first dispatch, `stopped`, `complete`), because the step is set before a
*dispatch* and never replaced: a write phase preceded by judging reported `dispatching human-decision judging` for
all 288 s of batch application and in the terminal payload, next to `run.state: done` (measured
2026-09-16).

Before validation, `subagent_executor.normalize_result` supplies `[]` for any of
`depends_on`/`files_to_create`/`files_to_modify` (`TASK_GENERATION_LIST_KEYS`) a task
left *out* — on both executor parse lanes and, for the native host's file, in
`consume_document`, which writes the completed document back before validating because
`update_plan_tasks` reads the file. Absent and empty mean the same thing to every consumer, and a
small model spells "none" by omission (measured 2026-09-14: `opencode/nemotron-3-ultra-free` returned ten
well-formed single-file tasks with no `files_to_create` on any and failed rung 6 twice); a key
present with the wrong type is left for the schema to name, and no other kind is completed.

The prompt's reply rules say the three keys are always present. `subagent_type` is
**required-but-nullable** in `generated_tasks.schema.json` since 2026-10-02 (OpenAI's strict
structured-output subset needs every property in `required`; codex used to decline the
schema on that one key and fall back to a `recovered` parse), `normalize_result`
supplies `null` when a reply omits it (to codex's native answer too, before
`structured_output.classify_attempt` validates it — its `normalize` hook, fix round 8), the prompt
still never names it, and the `executor_role` agreement the schema's `allOf`
used to state is `task_decomposer.validate_generated_tasks`'s fifth rule.

Every node also carries a `type` since 2026-10-05: `"schema_version": {"const": "1.0"}` had none,
codex's gate passed it, and OpenAI answered every codex generate-tasks dispatch with
`invalid_json_schema … schema must have a 'type' key` (400, rung 7, measured 2026-10-04); `structured_output._strict_openai_errors` now models the
documented strict subset (types — a non-empty `enum` stands in for one, as the API
accepted live, `const` never does —, `anyOf` only,
`items`, unsupported keywords and formats, `$defs`) so a schema
outside it is declined (`recovered`) rather than refused, and `tests/test_strict_schema_subset.py`
walks every schema an adapter is handed — `consolidation_batch.schema.json` is its one named exception, carried
by claude-code and qwen-code and declined for codex.

The document is then validated against `schemas/generated_tasks.schema.json` **plus** `task_decomposer.validate_generated_tasks`
(unique `T###` ids, `depends_on` naming only declared ids, acyclic, at
least one dependency-free task) — a failure is `schema_invalid` (rung 6; on the executor lane
the reply first gets its one schema-repair re-ask, see the executor bullet) and nothing is written.
Both checks are the executor's own `validate_result("task_generation")` verdict since 2026-10-05, so a
rule-invalid reply (a cycle) is re-asked and never persisted, `task_is_complete` never counts
one as done, and a result file on disk that fails either is never resumed: a plain re-invoke
archives it and re-dispatches (`stale_result_problems`).

Success requires the *artifacts*: a fresh `<!-- TASKS_FILE: … -->` marker, a rendered
`tasks/tasks.md` that parses, and a task count matching the document; otherwise the plan and
`state.json` are restored to their pre-consume bytes and the run exits 3. The same care
binds the *other* end: when a plan carries a `TASKS_FILE` marker whose target is missing,
`implement` is exit **3** naming the path and `--generate-tasks`, never the old
WARNING-and-parse-the-plan fallback, which fabricated 18 tasks titled ". Overview" with
`camera.update` declared as a file to create (measured 2026-09-16); only a plan with **no**
marker at all (the legacy embedded-task format) still parses its own content.

`{PROJECT_ROOT}/.multi-llm/tmp/generated_tasks_*.json` is **not** part of any contract any more — the temp file, its
`.gitignore` bootstrap and the second git-root resolution are deleted, not ported.

## Full workflow state machine

**`full` is Python too (`full_workflow_orchestrator.py`): a resumable state machine over the
other modes on the same `state.json`.** It calls each phase's `main(argv)`
in-process through the `modes.py` table (never `multi_llm.main`, which owns one
`MULTI_LLM_RESULT` line and one result context per invocation), forwarding the cross-cutting
flags but **never** `--detach`/`--internal-*`. A phase counts as progress only
when *it* recorded itself completed or skipped — exit 0 at a `[…_PENDING]` marker is not a
finished phase — so a phase that did not finish is D6 row 14: exit **3** naming the phase, its own
printed follow-up and the `full <plan> --resume` re-invocation, never a question.

Every phase boundary is a resume point (`--resume` continues from the first phase not
recorded; `--force` re-runs recorded ones and is *not* forwarded); row 12 runs
`review-tasks` under `--yes`/`--review-tasks` and otherwise records the
skip; the two conditional apply phases ask the existing prerequisite counters and record their
skips.

`full` and `status` resolve an upstream phase the same way,
**completed wins over skipped**, because `mark_phase_completed` does not clear an earlier skip
record: `step_apply_code_fixes`/`step_apply_task_suggestions` test `_recorded(<review>) == "skipped"` and
`check_apply_task_suggestions_prerequisite` uses the effective status too (2026-10-01; before it, a review skipped and
later completed made `full` skip its apply phase while `status` showed
it pending). `upstream_skip_notices` follows the same rule: a phase that was skipped and later
completed prints no `[PREREQUISITE_SKIPPED_UPSTREAM …]` notice. `--full` never proceeds past a
failed task generation.

`full` owns no workflow phase but does own a run dir (`{plan}/full/`) for
`--detach`'s `.status.json`, log and lock.

Its identity is class attributes read through the instance (`MODE_NAME`,
`RUN_DIR_PHASE`, `STATE_KEY`, `REPORT_TITLE`; the module globals are gone,
so a missed read is a `NameError`, never `full`'s value), and the
composites subclass it through six hooks whose `full` defaults are its old code,
moved: `step_argv(step)` (the five steps a composite reuses — both reviews, both applies,
consolidation — build no argv of their own), `skip_reason(step)` (asked inside
`run()`'s unchanged `step.skippable and not args.force` guard; `full`'s is
`_recorded`), `nothing_to_do()` (`False`), `resume_flags()`
(row 14's flags after the plan), `report_artifacts()`, and the module function
`consolidation_skip_reason(plan_dir)` (`already ran on these review groups` / `not recommended by review-plan` / `None`),
which a planner that builds no workflow can ask too.

The refactor guard's golden and `tests/test_full_workflow_orchestrator.py` hold `full`'s per-step argv
byte for byte.

## Improve-plan and improve-code

**`--improve-plan` and `--improve-code` are composites: a `sequence` row
each, a thin orchestrator each, and `utils/improve_workflow.py`**
(plans/improve-plan-and-improve-code.md). One review, then the apply of its findings, unattended;
the row's `phase` is a run dir only, as `full`'s, and the two
orchestrators exist only because `SCRIPT_TO_MODE` keys one mode per script.

Every rule lives in `improve_workflow` once, pure over the mode word, the flags and
`Records` (`read_records`; an unreadable `state.json` is
`StateReadError`, exit 3, never a plan from nothing): `skip_reason` says whether a
step runs (a review on record only as skipped runs, unlike under `full`) and
`step_flags` what it gets — the review `--yes` plus the typed
`--models`/`--quick` (no model list of its own) and `--force --rerun-all`
on a `--force` round; the apply `--yes --agent-decide --no-confirm`, plus `--resume`
only when the review is passed over and `apply_phase_in_progress` holds, never
`--force`/`--fresh` — so both legs plan one state one way.

Executor leg: `ImproveWorkflow(FullWorkflow)` through the hooks above, each argv `step_flags` +
`cross_cutting_flags` (never `common_flags`). Native leg:
`instructions/improve-plan.md`/`improve-code.md` load each phase core as `util improve-steps` plans
it (`utils/improve_steps.py`: the same two functions, read-only; `[IMPROVE_STEP_FLAGS]`,
`[IMPROVE_STEP_FRESH]` to skip a core's Resume Detection, `[IMPROVE_RERUN]`).

Nothing to do is `[ALREADY_COMPLETE] <mode>: …`, exit 2, asked before the detach boundary so a tool route's
`--detach` re-run launches no worker; an apply step that keeps every batch of a
`--force` round and dispatches nothing is still stamped after the review it applies
(`WriteBatchScheduler._finalize` keeps a resume's old completion date only while it is not older than that
review, 2026-10-09), so the next re-run is this exit 2; the next step answers as the row's
`last_phase`, on exit 0 and 2 (`full`: 0 only), `status`'s
`resume with:` is `recorded_continuation` (the typed selectors, never `--force`,
then `--resume`), and `--blocked <composite>` answers for the first step
`skip_reason` would run.

A live run owning a dir the composite writes (its `sequence` dirs,
`full/`, its own run dir; `live_owner`, read-only) is refused at exit 3
with its status and abort commands, never joined, `--force` included, and
`ImproveWorkflow.run_mode` holds each step's phase-dir `.lock` while it runs
(`full` gets neither). `--interactive` is exit 3 before
`--yes` is implied. Legacy: no frozen copy; with `MULTI_LLM_LEGACY_INSTRUCTIONS=1` both serve
the current core with no notice, and `test_every_served_mode_has_a_frozen_copy` exempts exactly these two.

## Ask mode

`ask` mode (`ask_orchestrator.py`) is a read-only free-text Q&A side-channel:
markdown output, no JSON schema / grouping / validation, output under `{plan}/ask/<question-slug>-<hash8>/`. It
shares concurrency mechanics with the review modes via `run_models_concurrent` in
`review_orchestrator_base.py`.

## Implementation artifacts and state

**`implement`'s write phase closes the artifact/resolution/denominator gaps the apply
phases already closed for themselves.** `run_implement_write_phase` now seeds every materialized task
`pending` in `state.json` before the scheduler loop starts, so
`check_workflow_prerequisites`'s task count reads the plan's total rather than however many happened to
finish or be skipped so far (F5); records its own artifacts the way `ApplyOrchestratorBase._record_write_artifacts` does,
since `implement` never goes through that base (`write_tasks.json`,
`write_batch_report.json`, `subagent_batch_report.json`, `summary.md`, `prior_changes.jsonl`, and
every `files_touched` path an `ok` outcome names — F7); buffers the row-8
human-task-strategy auto-resolution past the detach boundary the way
`_record_resolution`/`_flush_resolutions` do, since it used to fire before a run handle existed
and left `.status.json.resolutions` empty under a log that printed `[AUTO_RESOLVED]` (F8); and
stamps `write_tasks.json` with `inputs.tasks_sha256` for the plan's external
`tasks/tasks.md` (F15).

Its `subagent_batch_report.json`'s own `task_file` field used to read `null` on
the write path — `run_task_file` fills it from the argument the read path is handed, and
`WriteBatchScheduler` takes no such argument — so `_write_report` now names it directly:
`write_task_file_for_phase` always persists the same materialized list at `{phase_dir}/write_tasks.json`, which
every write orchestrator passes as the scheduler's own `phase_dir` too (N9). 2026-09-16.
