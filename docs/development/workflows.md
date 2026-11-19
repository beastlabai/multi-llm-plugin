# Workflow orchestration

Read when changing task generation, full or composite workflows (a composite's `--cascade` included), commit reviews, ask mode, or implementation bookkeeping.

Back to the [development guide](../../skills/multi-llm/AGENTS.md). These notes retain detailed contracts and regression history from that guide. Paths beginning with `docs/`, `packaging/`, or `plans/` are repository-relative; other implementation paths are relative to `skills/multi-llm/`. Dated incidents explain the rules; verify version-specific claims against current code.

## Contents

- [Task generation](#task-generation)
- [Full workflow state machine](#full-workflow-state-machine)
- [Improve-plan and improve-code](#improve-plan-and-improve-code)
  - [A dry run previews one step](#a-dry-run-previews-one-step)
- [A cascade (`--cascade`)](#a-cascade---cascade)
- [Commit reviews](#commit-reviews)
- [Ask mode](#ask-mode)
- [Implementation artifacts and state](#implementation-artifacts-and-state)

## Task generation

**`generate-tasks` is Python (`generate_tasks_orchestrator.py`), and it is the one phase whose subagent
sees the repository.** The sequence is: `ensure_prerequisites` (row 5, called — never reimplemented)
→ materialize **one** `task_generation` task into `{plan}/generate-tasks/task_generation_tasks.json` (the standard
envelope, so the executor needs no special case) → dispatch → validate → `update_plan_tasks.apply_tasks_document(...)`
**in-process** → verify the outcome. `util update-tasks` reaches the same function through
`update_plan_tasks.main`, so a hand-written list gets the same schema and graph rules
(`validate_result("task_generation", …)`), and both plan writes go through a temp file and
`os.replace` (`update_plan_tasks.replace_file`), so a crash mid-write leaves the old plan.

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

The mirror's finished tasks are its `tasks` rows; its dispatch counter used to count them as
`dispatched`, so a single-task batch running minutes read `dispatched: 0` the whole
time, indistinguishable from a stall. The tasks still running are `in_flight` (`{task_id, dispatched_at}`,
patched by `_begin_sweep` at dispatch time so it is visible before the task finishes),
rendered as `in flight: <task_id> (since <duration>)` (F10, 2026-09-16). And `write_reports.WriteReports.mirror_status` writes the
phase-level `state` (`complete` / `subagents_failed` /
`running`) beside the run record, as the read phases do, instead of leaving
`phase state: (none)`.

It writes `progress.step` there too (`applying` on a mirror taken before the
first dispatch, `stopped`, `complete`), because the step is set before a
*dispatch* and never replaced: a write phase preceded by judging reported `dispatching human-decision judging` for
all 288 s of batch application and in the terminal payload, next to `run.state: done` (measured
2026-09-16).

Before validation, `task_result.normalize_result` supplies `[]` for any of
`depends_on`/`files_to_create`/`files_to_modify` (`TASK_GENERATION_LIST_KEYS`) a task
left *out* — on both executor parse lanes and, for the native host's file, in
`consume_document`, which writes the completed document back before validating because
`task_is_complete` re-reads the file. Absent and empty mean the same thing to every consumer, and a
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
marker at all and no default task list beside it (`tasks_file.default_tasks_path`,
`{plan}/tasks/tasks.md`; the legacy embedded-task format) still parses its own content.

The marker has one reader, `utils/tasks_marker.py` (text only; `utils/tasks_file.py` re-exports it
and does the rest): `TASKS_FILE_MARKER_PATTERN` (whitespace
inside the comment tolerated and the value may be empty, since a reader that misses a marker
parses the plan's prose), `tasks_marker_spans` (the markers that count: none inside a fenced
code block or inline code, which are examples of the syntax), `tasks_marker` (the first one's
path as written; `""` for a marker naming nothing, `<!--TASKS_FILE:-->` included, which is
still a marker), `resolve_tasks_marker` (each relative, no `..`, real path inside the plan's
directory, else `TasksFileRefused`; markers leading to different files are
`TasksFileAmbiguous`, since following the first would implement a stale or pasted list) and
`read_task_list` (the named file's text; without a marker the default task list's when that is
a file, as `find_tasks_file` reads it, so no two phases read two task lists, and the plan's only
when it is not; `TasksFileMissing` when a named file is not there). Code detection errs on the marker's side: a
fence never closed is not code, and a stray backtick never reaches a marker on its own line.
The writer, `plan_updater.insert_tasks_file_reference`, rewrites every marker that counts
(its `## Implementation Tasks` section whole, any other in place) and leaves quoted examples
alone, and `generate_tasks_orchestrator.verify_outcome` checks the result with
`resolve_tasks_marker`, so generation never reports success on a marker implement refuses.
Every phase that parses
the task list goes through `read_task_list`: implement (a refusal is the same exit-3
`TASKS_FILE_MISSING` block), `decompose_plan_file`, the scheduler's check-task ids and the
known-checks task list. `find_tasks_file` (review-tasks, apply-task-suggestions,
code-review) keeps its exit 1 for a refusal. Readers that only *name* the file — the
workflow-file exclusions in `output_handler`, the code-fix batcher's do-not-edit list —
use `tasks_marker` (directly, or through `plan_updater.get_tasks_file_path`) without the containment, since naming one file too many is the
conservative side there. Until 2026-11-08 implement read the marker with a second pattern
that wanted exactly one space after `<!--`, so `<!--TASKS_FILE: …-->` turned the plan's
`## 1. Overview` headings into tasks, and it opened `../` paths review-tasks refused; every
reader took the first match, so a marker quoted in a code block above the real one won.

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

**`--scope-policy` is resolved once per run and forwarded as one value**
(plans/scope-policy-and-report-tags.md §4): `common_flags` ends with `--scope-policy <value>`, so
every phase and the prerequisite gates' one auto-run apply get it, and `step_generate_tasks` sets the
child namespace's `scope_policy` to the same value. Only a user's choice is forwarded — the typed
flag, else the live choice an earlier run of the same mode recorded in `state.json`'s
`scope_policy_runs[<mode>]` (`utils/scope_policy_runs.py`: written when typed, live until the
run's last phase records a newer completion or skip or the run closes it at its end, never read
under `--force`); with no choice no phase gets the flag and each resolves from its own review
round's record, so omission and an explicit `expand` differ only there. A phase therefore reads a
received `--scope-policy` as explicit: it always is the user's. generate-tasks and implement keep a
typed choice under their own mode word the same way, for the re-run or `--resume` that never
repeats it, and add `scope_policy_runs.generation_guidance`/`implementation_guidance` to their
prompts under `preserve` only.

Starting a new run without a choice closes the previous run's policy record
(`scope_policy_runs.record_run_start`), so resuming an unfinished default run cannot restore
an older `preserve` choice. This applies to `full` and the improve composites under `--force`,
generation under `--force`, and implementation under `--fresh`. Leaf-mode dry runs leave the
record unchanged.

**Scope deferrals finish an apply; a scope blocker does not** (plans/scope-policy-and-report-tags.md
§5). A bounded `--scope-policy preserve` apply that leaves optional extensions deferred records the
phase completed with the round's `scope_policy.<review>.completion.outcome: "eligible work
completed"` and its `deferrals` kept, so `check_workflow_prerequisites` reads it as met and the
bounded one-auto-apply/recheck finds nothing missing (no loop; `status` adds `scope_deferred` and a
notice). A `SCOPE BLOCKER:` decline records `outcome: "blocked"` and `blockers`;
`scope_blocker_reason` (apply phases only: the record is the review round's, and the review phase
must stay a met prerequisite of the apply that clears it) then makes the apply prerequisite checks
unmet (`scope_blocked`; `check_apply_task_suggestions_prerequisite` too, which `collect_missing`
and implement's own gate turn into a hard prerequisite of implement rather than the row-7
advisory), so a dependent phase refuses with the two ways forward (amend the plan and review
again, or the apply mode with `--scope-policy expand`). `--skip-prerequisites` cannot skip a scope
blocker (`ensure_prerequisites` and implement's gate refuse, exit 3). An explicit `expand` over a completed apply
with open deferrals, or one that stopped on a scope blocker, is not "already complete": the
composites ask `improve_workflow.releases_deferrals` (through `Records.scope_deferrals` and
`Records.scope_blockers`) and `full`'s `skip_reason` asks `scope_apply.releases_deferrals`, both
with the run's choice — the typed flag or the run's live `scope_policy_runs` record — so the
override survives a later run that omits it. Deferrals and blockers only come from a `preserve`
apply, so `expand` is the one policy that releases them; the check sits where a step is chosen,
never in `full`'s `_recorded`, which also judges whether a step that ran recorded itself.

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
`--force`/`--fresh` — so both legs plan one state one way. Both steps also get the one
`--scope-policy` `scope_policy_choice` resolves (typed, else `Records.scope_policy_run`, never on a
new round; the consolidation pass only when this run's review writes its round), the
continuation carries a typed one like the other selectors, and `expand` (that same choice) over a
completed apply with open scope deferrals or a scope blocker runs it again
(`releases_deferrals`) instead of `[ALREADY_COMPLETE]`.

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
`--yes` is implied.

### A dry run previews one step

**Under `--dry-run-subagents` a composite, and `full`, runs the first step that would run, with the
flag, and no step after it.** Each later step needs the real output of the one before (an apply needs
its review's verdicts, which need its findings), so a preview of it would describe nothing. `plan_steps`
passes over every step after its first `run` one (`skip (dry run: previews step <i> only)`,
`_dry_run_plan`), and both legs follow it: `ImproveWorkflow.run_preview` runs that one step in this
process (a review prints its reviewer plan and calls no model,
[review results](review-results.md#a-dry-runs-reviewer-plan); an apply renders its batches), and
`util improve-steps` plans the same single step for the native leg. The entry point runs the preview
before its detach boundary, so it never detaches (`cli_common.FOREGROUND_PREVIEW_FLAGS`), and it
records nothing: no progress record, no scope-policy choice, no row-14 stop. Exit 0 once the step
previewed, or the step's own exit when it refused. A cascade is never planned (planning a stage
records it): both legs print `dry_run_cascade_note` and preview one plain round instead.

**`full` plans from its own step decisions, read and never written** (`FullWorkflow.run_preview`). A
phase on record is passed over (`skip_reason`); so is a step that would record a skip or not run at all,
which each step answers through its `Step.pass_over` (row 4's consolidation pass, row 12's
review-tasks, and the two conditional applies; the real steps record the same reasons through the same
helpers). A skip the preview passes over counts as recorded for the steps after it (`_recorded` reads
`_preview_skipped`), so an attended preview that passes review-tasks over also passes its apply over,
as the real run would. The first step left runs its phase through `Step.command`, the mode word and
flags the real step hands it; the steps after it read `skip (dry run: previews <label> only)`. A
phase that answers exit 2 (nothing to do) is read as the real run reads it (`_preview_stop_on_exit_2`,
`_verify`'s reading without its words or recorded follow-up): with the phase recorded completed or
skipped and no batch of it left unexecuted, or for the advisory consolidation pass, the walk goes on
to the next step (`nothing to do (exit 2); a real run reads it as done and goes on`), the phase's
notice is dropped from the result line, and the closing line names it beside the phase that
previewed; with nothing recorded, or a batch left unexecuted, where a real run stops on row 14,
the preview says why and stops with exit 3.
The common case is `full --force` over a completed review: `--force` is not forwarded to the
phases, so review-plan answers "already completed" and a real run goes on to the next phase
(Fable review 2 of 2026-11-19, N3). Only one phase that dispatches is ever previewed. The
prerequisite gates' auto-run never fires (implement's gate is the phase's own, standalone), nor does
any row's recorded resolution. A workflow with every phase on record, passed over or with nothing to
do previews nothing and exits 0, as a real run of it does. `full`'s entry point, like the
composites', runs the preview before its detach boundary.

**A consolidation pass previews its batches without writing them.** When improve-plan's step 1b (or
`full`'s Phase 1b) is the step previewed, it runs review-plan's `--consolidate --dry-run-subagents`,
whose handler builds the batches as a real pass does (`consolidation.preview_consolidation_tasks`),
renders each command through `task_file_runner.dry_run_report`, and writes nothing but
`subagent_dry_run.json`: no `consolidation_tasks.json`, no prompt files, no host index, and no
`[CONSOLIDATION_*]` marker naming a file that is not there
(`review_plan_orchestrator.consolidation_preview`). `--consolidate-dry-run`, the stats-only preview,
builds the same document unwritten.

Until 2026-11-19 the flag was forwarded to every step and the review ran for real; its unrecorded
phase then failed `_verify`, so the composite ended at exit 3 ("did not complete") after a paid
review and a preview of its validation only. `full` kept forwarding it phase by phase a little longer,
so its review previewed and `full` then stopped at exit 3 on the unrecorded phase; and a preview whose
step was the consolidation pass wrote the pass's task file and prompt files while its closing line said
nothing was recorded. `tests/test_full_workflow_orchestrator.py` (`TestDryRunPreview`),
`tests/test_improve_orchestrator.py` and `tests/test_native_prompt_files.py` hold both.

## A cascade (`--cascade`)

**`--cascade` turns `--improve-plan` and `--improve-code` into one review-then-apply round per
model, one model after another** (`utils/improve_cascade.py`, `utils/cascade_provenance.py`,
`utils/cascade_ledger.py`, `utils/cascade_archive.py`). A stage is today's composite run on a
derived namespace (`derive_stage_args`), so `skip_reason`, `step_flags` and `plan_steps` decide what
a stage runs, unchanged, and each stage reviews the work as the stage before it left it. Three parts
are the cascade's own: the forward ledger (later stages are told what earlier ones rejected and
changed), a per-stage archive with a per-round table, and per-stage resume. `improve-commits` does
not take the flag (`takes_cascade`: a fix commit moves the pinned range's head), so its parsers
refuse it as unknown. Without `--cascade`, and with no unfinished cascade on record, nothing
changes: the same flags, prompts byte for byte, files and report.

**`advance` is the only writer.** `peek` and `applies` are read-only: they answer before a detach
boundary and in the read-only native planner, so a refused or finished launch leaves nothing behind.
`advance` creates, continues or abandons a cascade, binds the review round on record to the current
stage, closes out a finished stage, freezes the next stage's inputs, and hands back a `StagePlan`
(the stage to run, its namespace and the notices to print first), `Complete` or `NotACascade`. Its
first call of an invocation (`initial=True`, the user's flags) may start, restart or abandon a
cascade; a later call is bound to a `cascade_id`, ignores the restart and selector flags, never
starts one, and refuses with `[CASCADE_CONFLICT]` once the pointer names another (or an abandoned)
cascade. A later call also names the stage it follows (`after_stage`): when, after binding,
close-out and skipping, the current stage is still that one, neither closed nor skipped, it raises
`StageNotFinished` (exit 3: the stage's steps stopped before it was done) instead of handing the
same stage out again, so neither leg can run a stage's paid review over and over. Refusals are `CascadeRefusal` (`code` 3 or 1, `message` printed whole; `refusal_exit`
prints an exit 3 through `fail_closed`, so the result line is needs-human).

**`advance` reads, decides and writes under the composite run dir's lock** (`_planning`): the
`process_control.PhaseLock` every writer of that dir takes
([One writer per phase dir](runtime.md#one-writer-per-phase-dir)), re-reading the pointer and the
record after taking it. Without it two planners started together (two native `util improve-cascade
next` calls, or an executor run beside one) both read "no cascade", each created its own and both
handed out a paid stage-1 review. A lock this process already holds on the run dir (the detached
worker holds it for its whole run; `improve_workflow.holding`'s registry or
`process_control.held_phase_lock`) is used as it is and left held. One it takes is tried
`STEP_LOCK_ATTEMPTS` × `STEP_LOCK_RETRY_S` apart, as a composite step tries its own, and registered
with `holding` while held, so the close-out it may run takes the review and apply locks after it and
re-enters the run dir's (the run → review → apply order holds). A lock another live run holds is
exit 3 with `live_owner_refusal`'s text (`<composite> cascade planner` names a planner holding it),
nothing read or written. An initial call decides once before taking it, read-only: an answer that
writes nothing (not a cascade run, a finished cascade's command again) takes no lock and creates no
run dir. Two planners started together therefore hand out one cascade: the other is refused, or,
when the first released the lock within its retries, continues that same cascade at stage 1.

### The pointer and the record

**One directory per cascade, and a pointer naming the current one:**

```text
{plan}/<composite>/cascade-current.json                  the pointer
{plan}/<composite>/cascades/<YYYYmmddTHHMMSS>-<id8>/
    record.json          the record
    start.md             improve-plan: the plan when the cascade started
    files.txt            improve-code: the frozen --files list, when one was typed
    summary.md           the per-round table (a view)
    ledger.md            every ledger entry up to the last closed stage (a view)
    stage-<k>/inputs/    frozen at the stage's start, manifest.json last, never rewritten
    stage-<k>/artifacts/ published once at close-out, .complete.json last inside it
```

`<composite>` is the composite's run dir (`improve-plan/`, `improve-code/`) and `<id8>` the first
8 characters of `cascade_id`. The stamp has no colons and stage dirs are named by index, not model,
to keep archived paths short (Windows). The pointer is
`{"schema_version": 1, "cascade_id": …, "dir": "cascades/<stamp>-<id8>"}`. A new cascade never
renames or reuses a path: its directory, `start.md` / `files.txt` and record are written first, an
unfinished predecessor is marked `abandoned`, and only then is the pointer replaced, so a crash
leaves at worst an orphan directory nobody reads. `--force` without `--cascade` marks the unfinished cascade `abandoned` and removes the
pointer. Finished and abandoned cascades stay where they are.

**`record.json`** holds `cascade_id`, `mode`, `state` (`running`, `finished`, `abandoned`),
`started_at`, `finished_at`, `scope_policy` (for the stages not started yet), `selectors`
(`files_file`: `"files.txt"` or null; `base_ref` as typed or null; both null on improve-plan) and one
entry per stage: `index`, `model`, `state` (`pending` until its inputs are frozen and it is handed
out, then `running`, `done` once closed out, or `skipped`), `bound_round`, `scope_policy` (the one
it started under), `started_at`, `finished_at`, `skip_reason` and `counts`. Both files are written
atomically inside the plan's output dir (`write_in_plan_dir`) and checked on **every read** against
`schemas/cascade_record.schema.json` through `schema_validator` (the plugin's draft-07 subset, so
`jsonschema` stays a dev dependency), plus what a schema cannot say: the record's id is the
pointer's, its dir name ends in that id's first 8 characters, the mode is this composite's, and the
indexes run 1..n. A file that fails is `CascadeRecordError`, exit 3, naming the file and the way
out (repair it, or move the pointer aside): planning from a damaged record would start a new paid
cascade or continue at the wrong stage (the `StateReadError` precedent). An abandoned cascade on
the pointer reads as none.

### Stages and the stage namespace

**The stages are the review's resolved model list, in its order** (`resolve_stage_models`:
`interactive.resolve_models`, unattended, anchored at the plan, with the review's phase word):
`--models` as typed, else the `--quick` list, else `defaults.modes.<review>` / `defaults.models`;
canonical, duplicates collapsed, never reordered (weakest first is the user's order, and the user
docs say so). Creation prints `[IMPROVE_CASCADE] new cascade <id8>: <n> stages (<models>)`; with no
model flag typed it also prints `[AUTO_RESOLVED] models: <list> (<defaults key>)` once, because each
stage's review receives `--models` and so never records the auto-selection. One model is a
one-stage cascade that prints `NOTE: --cascade with one model is a single round`.

**improve-code's typed selectors are frozen at the start**, so a bare continuation reproduces them.
A typed `--files` or `--files-from <file>` list is read through review-code's own reader and written
once to `files.txt`; every stage's review gets `--files-from <absolute files.txt>`, never one
`--files` token per path (Windows command-line limits, and verbatim copying on the native leg).
`--base-ref` is recorded as typed and every review and apply gets it. `--files-from -` is refused
with exit 1 (`files_from_stdin_refusal`'s words): a stage's review runs long after the shell's stdin
is gone. With no selector typed, each stage's review resolves review-code's default scope as today;
that scope follows `tracked_files`, which the apply updates, so a later stage's default scope
includes earlier stages' fixes. A fix outside a typed list is not reviewed again: widening the scope
across stages is out of scope, and the user docs say so.

**`derive_stage_args` sets every planning input explicitly** on a copy of the user's namespace, so
nothing leaks from the invocation's own `--force`, `--resume` or selectors: `models` (the stage's
one model), `quick` false, `force` (a fresh round, from the binding below), `resume` false (the
apply's `--resume` stays `Records.apply_in_progress`'s inference, which already tells a new review
cycle from an apply in progress), `scope_policy` (the typed one, else the stage's), `files` none,
`files_from` (`files.txt`, when typed), `base_ref` (the recorded one), `cascade_stage_file` (the
stage's absolute manifest path) and `cascade` true. `step_flags` adds
`--cascade-stage-file <manifest>` last to every review of a stage, stage 1 included, and
`plan_steps` plans step 1b as `skip (cascade)`: the per-round table is the cascade's summary, a
one-model round rarely reaches the consolidation threshold, and the pass can still be run by hand on
the live review. Every stage starts
with a fresh round (`--force --rerun-all`), stage 1 included, so a completed non-cascade review on
record is never read as stage 1's result.

**The stage namespace plans; it never prints a continuation.** Its `models`, `files_from` and
`scope_policy` are the stage's, so `continuation_flags` on it would name one model and the frozen
file. The continuation is built from the user's own namespace with `cascade` true, which puts
`--cascade` right before `--resume` and never `--force`, a stage's derived `--models` or the frozen
selectors (`recorded_continuation(..., cascade=True)` for the readers below).

**The scope policy** is the typed `--scope-policy` at the start, else none: each round then resolves
its own, and a new review round is `expand`. A stage records the policy it started under. A
different value typed on a continuation becomes the record's policy for the stages not started yet
(a `NOTE:` names them), while the round under way follows today's rules (`[SCOPE_POLICY_CONFLICT]`
for a review round already started); it is never a `[CASCADE_CONFLICT]`. The default does not
change; the user docs recommend `preserve`, because extensions compound over rounds.

### Provenance and binding

**A retry cannot be told from a foreign review by its model or its round id** (every fan-out stamps
a new `round_id`), so each stage's review carries explicit provenance. `utils/cascade_provenance.py`
owns the format and imports nothing from the orchestrators.

- **The manifest.** `freeze_inputs` writes the stage's inputs, then `write_manifest` writes
  `stage-<k>/inputs/manifest.json` **last**: `cascade_id`, `mode`, `stage`, `of`, `model` and
  `files` (`{name: {sha256, bytes}}`). An `inputs/` without a manifest is a crash mid-write and is
  removed and rebuilt; once the manifest exists nothing in `inputs/` is rewritten, and a later
  `freeze_inputs` only verifies it (`load_manifest`, every hash), so every call of the stage's
  rounds and the archive read the same bytes.
- **The hidden flag.** `--cascade-stage-file <manifest>` (help suppressed) is on review-plan's and
  review-code's parsers; review-commits forwards to review-code's parser, so only a hand-typed flag
  reaches it there. The composite always passes it absolute, because the dispatcher moves the cwd
  to the repository root; one typed relative is read from the cwd like the plan (the dispatcher's
  mode input-path options, detach's path flags). `replay_flags` drops it in every spelling, as it
  drops `--commit-stage` (`--cascade` is an exact option there, being a prefix of the hidden flag),
  so it never reaches durable replay flags, "Retry with:" lines or `status`.
- **Checked before anything is written.** `review_lane.check_cascade_stage` verifies the manifest
  (`block_from_manifest`), that it is the composite's current handoff (`check_current`: the run
  dir's `cascade-current.json` names the manifest's cascade, its `record.json` is `running`, and the
  manifest's stage is not `done` or `skipped` there), and that the round could record its block,
  before the phase dir, its lock, `state.json` or the checkpoint is touched: a refused stage leaves
  nothing, and a `--detach` launcher refuses in the foreground. A command printed for a cascade
  superseded since (`--force --cascade`, a second planner), abandoned or finished, or for a stage
  already closed out, is `StaleHandoff`, exit 3 (`ERROR: cascade stage handoff refused: …; run the
  composite's command again with --cascade`), never a paid review no cascade binds. The check reads
  both files without a lock: the planner writes them atomically, the pointer last. A `--reaggregate`
  given the flag gets the same checks and records nothing; it continues the round on record, block
  and all.
- **Recorded by the opening checkpoint.** `open_round` verifies the manifest again and writes the
  round's `cascade` block under the checkpoint mutex, by `next_block`'s three carry-over rules
  (review-results.md, [Round identity](review-results.md#round-identity)).
- **Read where a ledger is rendered.** The reviewer prompt (`round_prior_rounds`), validation
  materialization (`validator_prior_rounds`) and the judge (`cascade_judge`) read the block
  from the review round's `.status.json` and their frozen file through `ledger_text`, which
  verifies its hash before a byte reaches a prompt. A changed or missing file, or a malformed block,
  stops that call at exit 3 (`error_line`: restore the file, or start a new cascade with
  `--force --cascade`). A call that renders no ledger checks nothing.

**Binding, on each `advance`, for the current stage k.** `bound_round` is the review round id most
recently seen carrying stage k's block; binding only asks whether it is set, so `advance` need not
see every retry round.

| The review dir's `cascade` block | Meaning | Action |
| --- | --- | --- |
| this cascade's stage k, the round live or with results | stage k's round (a first fan-out or a retry) | bind: `bound_round` is its `round_id`, the stage `running`; today's rules decide what runs, and a done apply is closed out |
| this cascade's stage k, finished with no model result | stage k's review failed | a middle stage is skipped; the last stage starts over on an initial call and stops at exit 3 on a later one ([Zero-result stages](#zero-result-stages)) |
| none, another cascade's or an earlier stage's, and `bound_round` unset | stage k's review has not started | a fresh round (`force`, so `[IMPROVE_STEP_FRESH]`) |
| anything but stage k, and `bound_round` set | a foreign round replaced stage k's (a hand-run `--review-plan --force --rerun-all`, same model or not, or a `--force` round with any model list but the stage's one model) | `NOTE: the review on record is not this cascade's stage <k>; starting stage <k> over`; `bound_round` cleared, fresh round |

Before any of it, a stage whose `artifacts/` the archive already published for this cascade is
closed from those files, whatever the live dirs hold now. An `artifacts/` whose `.complete.json`
does not check out stops there, at exit 3 naming the directory (`_published_counts`, through
`read_complete`), before any binding or hand-out: left to the close-out, the refusal would come only
after a foreign round on record had sent the stage's paid review and apply round again. An
unreadable review `.status.json` (present, not a JSON object) is exit 3 naming it too, never "no
block": a bound stage would take the foreign-round row and start over.

### The cascade's forward ledger

**Later reviewers, validators and the judge are told what earlier stages rejected and what they
changed** (`utils/cascade_ledger.py`). The ledger is built from the **archived** stages only
(`stage-<j>/artifacts/` with a valid `.complete.json`): the live phase dirs are replaced by the next
round's `--force --rerun-all`, and `state.json` keeps one record per group id, so a later write
overwrites an earlier round's. A skipped stage has no archive and contributes nothing, and the
reviewer block's "N automated review rounds" counts archived stages.

**Two lists, one stable id per entry**, numbered per stage in `grouped.json` order, then any ids no
group carries, sorted:

- *Rejected or held back*, `R<stage>-<n>`, with the stage's model, the group's title and importance,
  a description clipped to about 240 characters and the recorded reason clipped to about 200. The
  first source that names a group wins: a group validated `invalid` (its validator's reason), a
  `needs-human-decision` item the judge skipped (the decision's reason), an item its batch declined
  (the record's `details.reason`), a scope deferral of a `preserve` round (labelled
  `held back by the scope policy (valid, not applied; raise it again only if the scope now allows
  it)`, with the deferral's reason: never worded as a rejection). `validation_failed` groups (never judged),
  `no-op` items (already addressed) and items the user skipped are left out.
- *Changed by earlier rounds*, `A<stage>-<n>`, with the model, title, importance and where it
  landed: every item recorded `applied`, at its suggestion's `reference` (else `section` or
  `location`) on improve-plan, at the files its batch changed on improve-code (the batch's row of
  `write_batch_report.json`, else its declared files, both from the stage's `records.json`).

An entry is one line, every field `text_clip.clip`-ped:
`- R2-3 [HIGH] (<model>) <title> — <description> — reason: <reason>`, or
`- A1-4 [MEDIUM] (<model>) <title> — at: <where>`.

**Each stage freezes four rendered files**, so no prompt re-renders entries and the prompt code
does not depend on the ledger module; each reader reads its own file through `ledger_text`:

| File | Read by | Content |
| --- | --- | --- |
| `ledger.md` | the reviewer prompt's `{prior_rounds}` | the whole `## Earlier review rounds` block with the rule wording; it may name `ledger-full.md` (on overflow only) and, on improve-plan, always `changes-so-far.diff`, by absolute path: reviewers read files |
| `ledger-validator.md` | validators, inline only | the two lists under `Rejected or held back in earlier rounds:` and `Changed by earlier rounds:`; its overflow line gives a count and **no path** (the materialized-task contract has no bare file pointer); the rule text is the prompt's |
| `ledger-judge.md` | the judge | the rejected-or-held-back list alone |
| `ledger-full.md` | reviewers, on overflow | every entry, uncapped, stage by stage (the cascade dir's `ledger.md` view is the same rendering) |

On improve-plan `inputs/` also holds `plan.before.md` (the plan's bytes at the stage's start) and
`changes-so-far.diff` (`start.md` against it, unified; empty at stage 1). A later improve-plan stage
whose `start.md` cannot be read is refused at exit 3 rather than frozen from part of the history.
Stage 1's four ledger files exist and are empty.

**The inline cap** is `LEDGER_INLINE_MAX_CHARS` (8,000 characters, the wording and the overflow
line included). Over it, entries are taken in one order, every rejected entry HIGH before MEDIUM
before LOW (the latest stage first within a level), then every applied entry (the latest stage
first), and the first that does not fit ends the take, so a less important entry never shows in
place of a more important one; a line counts the rest. Rendering is pure: the same archives give the
same bytes, which the prompt tests pin.

**The reviewer.** `prompts/plan_review.txt` and `prompts/code_review.txt` carry `{prior_rounds}` on
the same line as `{scope_policy_guidance}`. `prior_rounds_block` renders a blank line and the
ledger, or `""`, so a prompt outside a cascade or at stage 1 keeps today's bytes (pinned by
`tests/golden/prompts/plan_review_rendered.txt` and `code_review_rendered.txt`). review-plan wraps
its context factory with `with_prior_rounds` after the opening checkpoint; review-code reads the
ledger once per run and passes it through its slots. A reviewer that repeats a rejected entry on
purpose is asked to start the description with `Re-raises R<n>-<m>:`.

**Validators** get the validator file inline after the scope block
(`build_validation_prompt(..., prior_rounds=)`), under `## Earlier review rounds` with one rule: a
group repeating a rejected entry is judged on its merits but classified `invalid` unless it answers
the recorded reason, with the entry's id in `reraise_of` either way; a group challenging an earlier
change is an ordinary finding.
Only with a ledger does the reply contract gain `"reraise_of": "<R-id, or empty>"` and its rule 6;
without one the contract keeps its bytes. How the field is bound and merged: review-results.md,
[Round identity](review-results.md#round-identity).

**The judge** of a cascade stage (any stage whose review round holds a block, stage 1 included) gets
a `### Earlier review rounds` section after the rules: the sentence "In this run each item comes from
one model's round, so rule 6 (model consensus) does not apply." and, when the judge ledger is not
empty, the rejected list with its re-raise rule (skip a repeat unless it answers the recorded reason,
and name the entry's id in the reason). `HUMAN_DECISION_JUDGE_RULES` is never edited, so the
reference section rendered from it stays the same, and outside a cascade the judge prompt keeps its
bytes. The apply reads the block read-only.

**A judge verdict decides its group only in the stage it was judged for.** Verdict files
(`human_decision_batch_<group id>.json`) outlive a stage in the apply dir, and group ids are content
hashes: a later stage that repeats an earlier stage's finding word for word finds the earlier verdict
at its own file's name, and read as it was it decided the item without the later stage's judge
ledger (and the executor took it for a judge task already done). So in a cascade stage
`_handle_human_decisions`, before any verdict file is read (whenever there is a group in scope or a
verdict file on disk):

1. reads the stage's judge ledger through `cascade_judge` (hash verified): a ledger that fails stops
   at exit 3, whatever verdicts lie on disk;
2. moves aside every verdict file unless `human_decision_tasks.json` beside it records this stage
   and ledger (`retire_foreign_verdicts`: renamed `retired_<name>.other-stage.<stamp>.json`, out of
   every verdict and result-name pattern, through `rename_in_plan_dir`; a file it cannot move is
   exit 3);
3. materializes the judge tasks with `inputs.cascade_stage` (`<cascade_id>:<stage>`) and
   `inputs.judge_ledger_sha256`, so the verdicts written for them are this stage's, and a retry or
   resume of the same stage reuses them as today.

Outside a cascade none of this runs: no new key, no file moved, verdicts reused by group id as
before. Step 23, the recorded decisions and the records are today's.

### Close-out and the archive

**A stage is done** when `skip_reason` of its apply, on the stage namespace and a fresh read of the
records, is `completed` or `skipped` for the stage's round (the apply record not older than the
review). Close-out (`cascade_archive.close_out`) archives it and returns its counts; `advance` then
records the stage `done` with `finished_at` and `counts`, rewrites `summary.md`, and freezes the next
stage's inputs. After the last stage the record is `finished` and `advance` returns `Complete`.

**Copies, not moves.** The live review and apply dirs are copied into the archive and stay as the
phases left them, so `status`, `--reaggregate` and the next-step lookup work unchanged, and the next
stage's fresh round retires them as today. Every regular file is copied except `*.lock`; links are
neither followed nor copied; permission bits and modification times are kept (a reader dates
`write_batch_report.json` by its mtime). Every write goes through the plan-output containment
helpers (`write_in_plan_dir`, `rename_in_plan_dir`, `utime_in_plan_dir` for the copied times, and
the stage manifest's own write in `cascade_provenance.write_manifest`), never a raw join.

**A live dir is never reached through a link** (`_hold_live`). `os.walk` follows a link given as its
root, and a lock file opened through one is created at its target, so before any lock is taken every
folder from the plan's output dir down to each live dir (run, review, apply) is read with `lstat`: a
link there (or a junction, or anything but a folder) is `ArchiveRefusal`, exit 3, with nothing
locked, created or copied. On Linux and macOS each live dir is then reached from the plan's
directory one component at a time without following a link (`held_folder`'s walk) and held by its
descriptor: its lock file is opened relative to that descriptor (`PhaseLock.acquire(dir_fd=)`,
never through a link at the name), each held folder must still be the one at its path once every
lock is taken, and its tree is copied with `os.fwalk` from the descriptor, so a folder swapped for a
link during the close-out is never entered. Windows has no descriptor-relative opens: the `lstat`
check stands, and the dir is walked by path.

**Locks: the composite run dir's, then the review dir's, then the apply dir's**, always in that
order: the same `process_control.PhaseLock` files the phase runs hold
([One writer per phase dir](runtime.md#one-writer-per-phase-dir)). A dir that does not exist is
passed over. A lock this process already holds (`improve_workflow.holding`'s registry or
`process_control.held_phase_lock`: the detached worker holds its run dir's, and `advance` holds it
while it plans) is used as it is; one it
takes is retried as a composite step retries its own (`STEP_LOCK_ATTEMPTS` × `STEP_LOCK_RETRY_S`,
because `status` polls take and drop every lock) and checked with `detach.refuse_unconfirmed_cleanup`.
A lock a live run holds is never broken: exit 3 with `live_owner_refusal`'s text. After locking, the
caller's `recheck` re-reads the round's block, its round id and the apply's completion; a stage no
longer done raises `CloseOutRaced`, the locks are released, and `advance` plans again (three
attempts, then exit 3). The locks stay held until publication is done.

**Publication.** The archive is built in `artifacts.partial/`, `.complete.json` is written last
inside it (`cascade_id`, `mode`, `stage`, `model`, `round_id`, `grouped_sha256`, `review_dir`,
`apply_dir`, `published_at`, and `files`, every file's sha256 by relative path), every file is
closed, the directory is renamed to `artifacts/`, a name that does not exist yet (Windows refuses a
rename with open handles), and only then does `advance` update `record.json`. A `.complete.json` is
valid only while every file it lists still hashes to its value (`read_complete`).

| Found on disk | `advance` does |
| --- | --- |
| `inputs/` without `manifest.json` | removes and rebuilds `inputs/` |
| `artifacts.partial/` | removes it and redoes the close-out, under the locks |
| `artifacts/` whose `.complete.json` names this cascade and stage, the stage not `done` in the record | takes the counts from the published files (no copy, no lock) and advances the record |
| `artifacts/` with a missing, invalid, stale or mismatched `.complete.json` | stops at exit 3 naming the directory: restore it as published (`.complete.json` lists every file's sha256), or `--force --cascade`; never move it aside, which would drop a closed stage from the ledger without a word |

| In `stage-<k>/artifacts/` | |
| --- | --- |
| `review-plan/`, `apply-suggestions/` (improve-plan) or `code-review/`, `apply-fixes/` (improve-code) | copies of the live phase dirs |
| `records.json` | the stage's own `state.json` entries (below) |
| `plan.after.md`, `stage.diff` | improve-plan: the plan after the stage, and `plan.before.md` → `plan.after.md` |
| `files-touched.txt` | improve-code: the files the stage's apply changed, from `write_batch_report.json` where present, else the declared files of each batch that applied an item (the native leg writes no report) |

**`records.json` is extracted at close-out, before a later stage can overwrite anything**
(`extract_records`):

- `processed`: the `processed_<apply dir>` entries recorded after the stage's review completed
  whose `details.grouped_sha256` is the stage's `grouped.json` hash or, unstamped, whose item is one
  of the run's items (`write_tasks.json`'s `native_run.items` and its batches' `expected_ids`, when
  that file is this review's: its stamp is the stage's hash and it was not emitted before the review
  completed); an entry older than `native_run.started_at` is an earlier run's. The time window
  comes first because group ids are content hashes: a stage whose findings repeat an earlier
  stage's word for word has the same `grouped.json` hash, and without it the earlier stage's
  entries would read as this stage's Applied count and `A<k>-<n>` ledger entries;
- `human_decisions`: the `human_decisions_<apply dir>` entries recorded after the stage's review
  completed (judge records carry no `grouped_sha256`, and each stage has a fresh review, so the
  window is unambiguous; an undated one is kept);
- the review's and the apply's `phases_completed` / `phases_skipped` stamps, and the review round's
  `scope_policy` record, its deferrals included;
- `run_items`, `batches` (task id, expected ids, declared files), `files_by_task` (from a
  `write_batch_report.json` of this review cycle) and `nothing_to_apply`;
- `reused_items`: from that report, the processed items of each batch an earlier stage ran and
  this stage's apply reused without dispatching it: a row the scheduler's resume skip marked
  `skipped` whose result file predates this stage's review completion (`_reused_row`). A resume
  inside the stage keeps its own batches: the earlier run's row stands in for the skip
  (`from_earlier_run`, not `skipped`), and with no such row a result file written after the
  review is the stage's. The time window cannot catch these: when a stage's findings repeat an
  earlier stage's word for word, the batches are the same, the resume skip finds the earlier
  stage's result files (`already complete — skipping`) and stamps the items `processed` again
  after this stage's review. This stage applied nothing for them, so they count as No-op, never
  Applied, and get no `A<k>-<n>` ledger entry (`cascade_ledger.reused_items`).

A per-stage code diff is out of scope: it needs a working-tree snapshot primitive (the worktree
hashed again at every stage, loose objects written for untracked files, submodule edits seen only as
gitlinks) that nothing in the cascade consumes. `files-touched.txt` names each stage's files, and git
shows the cumulative change.

### Counting and the per-round table

**`stage_counts` has its own definitions**, read from the stage's archive only and not borrowed from
either existing counter, because those disagree: the native report counts `applied` and `no-op` as
items applied, `status` counts `applied` alone. Its keys are `record.json`'s `stages[k].counts`:

| Column (keys) | Definition |
| --- | --- |
| Findings (`findings`) | the groups in the archived `grouped.json` |
| Valid, Invalid, Needs decision (`valid`, `invalid`, `needs_decision`) | the review's `.status.json` `validation_counts`; with none, the archived `validation.json` records by status |
| Re-raises (`reraises`) | archived `validation.json` records with a non-empty `reraise_of`; `—` for a stage with no archived stage before it |
| Applied (`applied`) | `records.json` processed entries with status `applied` whose batch this stage's apply dispatched (not one in `reused_items`); `0 (nothing to apply)` when the apply handed out no item of the review (`nothing_to_apply`) |
| No-op (`no_op`) | status `no-op`, plus status `applied` in `reused_items` (the native report's "items applied" is Applied + No-op, and the table's notes say both) |
| Declined (`declined`, `skipped_by_user`) | status `declined`, plus status `skipped` (at the user's request), shown as `Declined (n skipped)` |
| Held back (`held_back`) | the scope deferrals of the stage's review round in `records.json` (`scope_policy.deferrals`: valid findings a `preserve` policy kept out of the apply) |
| Judge (A/V/S) (`judge_approved`, `judge_salvaged`, `judge_skipped`) | the stage's judge decisions: `decision_source` `judge_auto_decide` by `decision`, or `judge_auto_decide_salvage` |
| Time (`time_s`) | from the stage's start in `record.json` (`stages[k].started_at`, set once when the stage is handed out, so a retry fan-out, which resets the review's `.status.json` `started_at`, does not shorten it; that `started_at` stands in only with no record) to the stage's last recorded finish: the apply's completion or skip, else the review's completion. Once the stage is closed the table prints the record's `started_at` → `finished_at`: a wall-clock span from hand-out to close-out, so a stage stopped and resumed later counts its idle time too, and the table's notes say so |
| Cost (`cost`) | the reviewer's `usage` `cost_usd` summed, as `$x.xx est.`; validation, the judge and the apply run on the host and are not metered; `not recorded` when no model reported one, never `0` |

**Only the last stage's deferrals can be released.** Each stage's fresh round
(`--force --rerun-all`) replaces the live review, and `state.json` keeps the scope record of the
review on record only, so a `--scope-policy expand` re-run (`releases_deferrals`) offers the
deferrals of the last stage alone: after the cascade is finished, the composite run again without
`--cascade`. Earlier stages' deferrals stay in their archives, are counted in Held back and reach
later stages through the ledger as held back, never as rejected; the table's notes and the user docs
say so.

`render_summary` rewrites `summary.md` and the top-level `ledger.md` view at every close-out, skip
and finish. A done stage's row is read from its `artifacts/` (the record's `counts` stand in for an
archive that cannot be read); a skipped stage is one row reading `skipped: <reason>`; a stage not
closed reads `in progress` or `not started`. Both files are views: nothing points into them and no
prompt reads them.

### Zero-result stages

**A stage whose review produced nothing** (its one model in `models_failed` with
`models_completed: []`, which the review records only on that ending; a live or interrupted fan-out
has no `models_completed` key) is today's D6 row 3, far likelier in a one-model round than in a
parallel one:

- **A middle stage** (k < n) is recorded `skipped` with the model's failure reason. `advance` adds
  `[CASCADE_STAGE_SKIPPED] <k> <model>: <reason>` and
  `[AUTO_RESOLVED] partial-results-recovery: cascade stage <k> of <n> (<model>) produced no result; skipped, continuing at stage <k+1>`
  to its notices and goes on to stage k+1, whose fresh review records again, within the same run, the
  review completion that stage k's `--force` removed.
- **The last stage is never skipped**: skipping it would leave `state.json` with no review
  completion (its `--force` removed it), so a later plain `--improve-plan` would run a full new
  round, and it is the stage the cascade exists for. On an initial call `advance` hands it out fresh;
  on a later call it raises `CascadeRefusal` with exit 3 (`last_stage_stopped_message`: the stages
  before it are closed; run the command again to start stage n over). That difference is how the
  native leg's continue call tells "just failed, report the stop" from "the user ran it again".
- `zero_result_stage` answers, read-only, whether the review on record is the current stage's and
  produced nothing, so the executor leg can ask before any failure reporting.

Any other stop (an apply batch that failed, a human-task answer owed, a scope-policy conflict, exit 1)
stops the cascade at row 14 exactly as today, and the re-run continues that stage. A continue call
made after such a stop (the native leg runs one after every review stop, item 4 of the cores'
section 2C) finds the stage it follows still open and raises `StageNotFinished` (exit 3,
`stage_not_finished_message`: what is missing, and that the command run again continues the
stage), never that stage's block again.

### Resume, re-runs and `--force`

**Cascade intent comes before completion.** `preflight` is `safety_checks` (a live owner, then a
readable `state.json` and scope; unchanged), then `completion`, then the `--files-from -` check. When
a cascade applies (`applies`: `--cascade` typed, or an unfinished cascade on record), `completion`
asks `peek` instead of `nothing_to_do`: `complete` is `CascadeFinished` (its `[ALREADY_COMPLETE]`
line, exit 2, before any detach boundary), `work` goes on, a refusal returns its exit code, and
`none` (`--force` alone over an unfinished cascade) falls back to the plain rules. Without the split
two states would stop at `[ALREADY_COMPLETE]`: a new cascade over completed non-cascade phases, and
a cascade interrupted after a stage's apply completed but before its close-out. `peek` raises every
refusal `advance` would make at the start, so a refused or finished launch writes nothing.

| On record (the pointer's cascade) | Typed | Result |
| --- | --- | --- |
| none, or an abandoned one | `--cascade` (without `--resume`, or with `--models` or `--quick`) | a new cascade from stage 1 |
| none, or an abandoned one | `--cascade --resume` with no `--models` or `--quick` (a printed continuation kept from a cascade `--force` alone abandoned, or whose pointer was moved aside) | exit 3 `[CASCADE_CONFLICT] --resume: there is no unfinished <composite> cascade to resume (…); nothing was run`, naming the two ways to start one: `--cascade` without `--resume`, or with `--models`/`--quick`; nothing written, before any detach boundary |
| finished, no typed model flag or selector other than the record's | `--cascade` (the printed continuation `--cascade --resume` included) | `[ALREADY_COMPLETE] cascade finished at <time> (<n> stages); --force --cascade starts a new one`, exit 2 |
| finished | `--cascade` with a typed `--models`, `--quick`, `--files`, `--files-from` or `--base-ref` resolving to other than the record | a new cascade from stage 1 |
| finished | no `--cascade` | today's rules on the files on record (normally `[ALREADY_COMPLETE]`: the last stage's review stays completed) |
| unfinished | nothing, `--cascade` or `--resume` | continue at the current stage: `[IMPROVE_CASCADE] continuing stage <k> of <n> (<model>)` |
| any | `--force --cascade` | an unfinished one is abandoned (a `NOTE:` line), then a new cascade |
| unfinished | `--force` alone | marked `abandoned` and the pointer removed, with a `NOTE:` line; today's new round with today's models |
| unfinished | `--models`, `--quick`, `--files`, `--files-from` or `--base-ref` resolving to other than the record | exit 3 `[CASCADE_CONFLICT]` naming the flag, both values and the two ways out: re-run without it, or `--force --cascade` |
| unfinished | another `--scope-policy` | no conflict: the round under way follows today's rules, and the stages not started take the typed value |

Only typed values are compared (`_typed_difference`, for an unfinished and a finished cascade
alike), so a bare continuation reproduces the record; models compare in order (the order is the
cascade), files as a set, as the review compares a scope. On a finished cascade an untyped model
list is the record's, never today's defaults resolved again: every surface prints
`--cascade --resume` with no model flag while a cascade runs (a continue call's `[IMPROVE_RERUN]`,
the executor's row-14 line after a bare re-run, `util next-command`), and that command typed after
the finish must answer `[ALREADY_COMPLETE]`, not start a new paid cascade of another list.
Typed when no cascade is on record (the pointer gone after `--force` alone abandoned it, or moved
aside) it refuses for the same reason (`_nothing_to_resume`): `--resume` asks to go on, and
there is nothing to go on with.

**Other readers ask for an unfinished cascade first** (`next_phase.unfinished_cascade`, over
`cascade_status`, read-only), because between two stages every step is on record and the plain rules
would read "everything complete". `util next-command --after` and `--blocked` on a composite, and
`emit_resume`, answer the composite's own continuation with `--cascade`, with the why
`continues the unfinished cascade at stage <k> of <n> (<model>)`; `status`'s `resume with:` for a
composite run carries `--cascade` (`_composite_resume_tokens`). On a composite run's view (a run
record in its run dir) `status` shows `cascade:        stage <k> of <n> (<model>)` in the default
view, `--view run` and pinned polls, and `status --json` carries the run view's `cascade` key
(`{cascade_id, state, stage, of, model, dir}`, `run_view` in `schemas/status_json.schema.json`),
only while the cascade is `running` (`status_views.attach_cascades`). A cascade with no composite
run record (the native leg's, or one a foreground executor run stopped in) is read from the pointer
and the record instead (`status_views.unfinished_cascades`, which looks for the pointer first, so a
plan with none never loads the cascade core): the run view gets a block of its own,
`  <composite>  no run record`, the same `cascade:` line and `resume with: <composite continuation
with --cascade --resume>`, and `status --json` a top-level `cascades` list (each with its `mode`);
neither under `--run-id` or `--question`, and under `--phase` only for the composite it names. That
block stands for the composite: with no run view listed, no `no run recorded for …` line precedes it.
While any cascade is unfinished, the workflow section's suggestion (and a terminal `--view run`
poll's trailer) reads `suggested next: <composite>  (continues the unfinished cascade at stage <k>
of <n> (<model>))` instead of `suggested_next`, which reads every step done between two stages;
under `--phase` only when the phase is that composite, and under a `--phase` naming another phase
no suggestion line prints at all (`run_status._suggests`). The workflow payload itself,
`--view workflow` and the `--json` document's `workflow` key keep their bytes, and so does every
output without an unfinished cascade. All of it is advisory: a record that cannot be read costs the lines, not the poll, and the
composite's own run refuses it naming the file. Once the cascade is finished,
`--after improve-plan|improve-code` and `status` answer as today.

### The two legs

**Executor leg** (`ImproveWorkflow`, under `--exec-subagents`). The entry point and the worker's
`run()` backstop use the split `preflight`, so `peek` answers before the detach boundary; only the
foreground run and the detached worker call `advance`. The workflow loops over the stages: the
stage's namespace stands in for the user's in every planning read (`run`'s skip gate on `force`,
`step_argv`, `skip_reason`, `review_runs`, the scope-policy resolution), and `FullWorkflow.run`'s
step loop is split out so it runs once per stage and returns the failed step **before**
`_phase_failed` or `print_report` runs, which lets the cascade decide skip or stop; only then does
the existing failure path run, once, with the cascade's continuation (`run_steps`, with
`run` as today's behaviour on top of it; the stage namespace is `planning_args()`). Per stage it resets the
records cache, the `_scope_resolved` cache and the per-stage result fields (consolidation
recommendation, validation counts, notice, artifacts); `self.reports` and `_record_progress`'s steps
are keyed by stage and phase, so stages never overwrite each other and the final report lists the
last stage's steps. A failed review step that `zero_result_stage` reports as a middle stage's is a
skip: the needs-human override the review set (`mark_needs_human`) is cleared through
`unattended.clear_needs_human`, which only the cascade calls, and the loop goes on. Progress keys
read `stage-<k>/<phase>`, and the report prints the table, then `### Stage <k> of <n>: <model>`
with the last stage's steps; the executor prints no `[CASCADE_COMPLETE]` (the table stands in). The one `MULTI_LLM_RESULT` line is whatever the
cascade ends with: exit 0, `done`, with the next-command lookup's `next`, or exit 3, needs-human,
for a stopped last stage. Each `advance` after a stage passes `after_stage`, so a stage whose steps
all ran but that the cascade does not find done stops at `StageNotFinished` instead of looping. A
refusal of that call (`_cascade_refused`: an archive it cannot trust, a live lock at close-out,
close-outs that kept racing, a pointer naming another cascade, `StageNotFinished`) stops as row 14
does: the progress record's `current_phase` and `failed_phase` read `cascade`, the stop names the
cascade's continuation (`resume_command`, with `--cascade`) and the result line's `next` is it,
never a follow-up a phase printed before; a refusal of the run's first `advance` (nothing ran, e.g.
`[CASCADE_CONFLICT]`) names none. The report adds the per-round table, `report_artifacts` adds `summary.md`,
`ledger.md` and the stage dirs, and the result fields come from the last stage plus those artifacts.
The detached worker runs the whole cascade, and its run dir's `.status.json` names the stage under
way (`cascade_stage: {index, of, model}`, written only when the run dir already has a checkpoint,
`null` once the cascade ends); `status` reads the pointer and the record either way.

**Native leg** (Claude Code). `util improve-steps` stays read-only. When a cascade applies it runs
the safety checks and `peek`, prints a refusal or `[ALREADY_COMPLETE]` with nothing written, and
otherwise plans no steps and prints
`[IMPROVE_CASCADE_NEXT] <util improve-cascade next …, with the typed flags>` and
`[IMPROVE_RERUN] <the continuation, with --cascade>`, exit 0, so a run that stops before its first
`next` call still has its block for the report check.
`util improve-cascade next <plan> --mode … --host … <flags>` is the cascade's planner and recorder
in one call: it calls
`advance`, then prints the stage's block with the very functions `util improve-steps` prints with
(`[IMPROVE_STEP]`, `[IMPROVE_STEP_FLAGS]`, `[IMPROVE_STEP_FRESH]`, step 1b `skip (cascade)`), plus
`[IMPROVE_CASCADE] stage <k> of <n>: <model>`, `[CASCADE_STAGE_SKIPPED] …` when a middle stage was
just skipped,
`[IMPROVE_CASCADE_CONTINUE] <util improve-cascade next <plan> --mode … --host … --cascade-id <id> --after-stage <k>>`
and `[IMPROVE_RERUN]`. The first call (from `[IMPROVE_CASCADE_NEXT]`) carries
the typed flags and is `advance`'s initial call; `[IMPROVE_CASCADE_CONTINUE]` is bound to the
cascade's id and to the stage the call that printed it handed out (`--after-stage <k>`, which only
goes with `--cascade-id`), and carries no restart or selector flag, so repeating it never restarts a
`--force --cascade` run, it stops at exit 3 `[CASCADE_CONFLICT]` once the pointer names another
id, and at exit 3 `StageNotFinished` while stage k is still open: it never prints the block it was
called after. It exits 0 with steps to run, 2 with `[CASCADE_COMPLETE] <n> stages; table: <summary.md path>`
once every stage is closed, and 3 for a continue call's stop at its stage (a stopped last stage,
or a stage not finished), with `[IMPROVE_RERUN]`. `--json` mirrors
the improve-steps object plus a `cascade` key. `--force` without `--cascade` over an unfinished
cascade: `util improve-steps` hands off, and `next` abandons the cascade (`NOTE:`) and prints today's
plain step block with no `[IMPROVE_CASCADE_CONTINUE]`, so the core runs one round. The dispatcher
lists the util in its registry and help, the plan as the path positional after the `next` word,
`--models` in its variadic options, `--files`, `--cascade-id` and `--after-stage` among its value options and
`--files-from` in its path options. A continue call's `[IMPROVE_RERUN]` is `--cascade --resume`
alone (the record holds the selectors); the hand-off's and the first call's keep the typed
selectors. A stopped last stage (`improve_cascade.LastStageStopped`) and a stage not finished
(`improve_cascade.StageNotFinished`), both raised only on a continue call, print the stop on stderr
and `[IMPROVE_RERUN]` on stdout (`--json`: `cascade.state` `stopped`); every other refusal prints
nothing on stdout, and `[CASCADE_COMPLETE]` prints no `[IMPROVE_RERUN]`, so section 4's lookup is the block.

**The instruction cores** (`instructions/improve-plan.md`, `instructions/improve-code.md`) get a
section "2C. A cascade", referenced from section 1: run the `[IMPROVE_CASCADE_NEXT]` command once;
on exit 0 run its steps as section 2 says (sections 2 and 3 apply unchanged: flags verbatim, inner
`[NEXT_COMMAND]` blocks ignored, the stop rules); when the stage's last step is done, run the latest
`[IMPROVE_CASCADE_CONTINUE]` command, never the first command again, until it exits 2 with
`[CASCADE_COMPLETE]` (each call prints the next stage's block, never the same one again); when a
stage's review step exits 3, run the continue command once before reporting a stop, and go on with
its block if it prints `[CASCADE_STAGE_SKIPPED]`, otherwise it exits 3 and the stop is reported;
never end the turn between stages. Sections 1, 2C and 3 name "the latest `[IMPROVE_RERUN]` command
printed", the report check's rule: a refused first `next` call printed none, so the report ends with
the hand-off's. Section 4's report opens with a `### Cascade` section (`summary.md`'s table copied as
written, never recounted), then Steps 1 and 2 for the last stage, then Output Files with
`summary.md`, `ledger.md` and the stage dirs. The Stop-hook report check knows both planners by name
(host-reporting.md, [Claude Code report checks](host-reporting.md#claude-code-report-checks)).

**Out of scope:** several models in one stage (a stage's `model` could become a list without
changing the record's other fields), widening improve-code's review scope across stages, checks or
tests between stages, stopping rules (repeating the last stage until it finds nothing valid),
`--improve-commits --cascade`, `--full --cascade` and cascades of the review modes alone, a config
key for the order (`--models` sets it), and a per-stage code diff.

## Commit reviews

**`--review-commits`, `--apply-commit-fixes` and `--improve-commits` are front doors over the code
phases, not a phase of their own** (`plans/review-commits.md`). Each turns a range into a review-code or
apply-code-fixes run on a generated **commit brief**, `<commit_reviews_dir>/<slug>.md`, which stands in
for the plan, so the output dir, `state.json`, locks, `status`, run records, `--reaggregate`, resume lines
and the recorded-mode check see only review-code or apply-code-fixes. A `commit-review` phase was
rejected: apply's `review_subdir`, all-of `PHASE_PREREQUISITES`, `REVIEW_PHASE_BY_WRITE_KIND`,
`OUTPUT_TYPE_TO_PHASE`, the literal `code-review/grouped.json` reads and `next_phase`'s ordering would
all have had to learn it, where sharing the phase costs one rendering hook (`utils/commit_spelling.py`,
keyed on `Mode.commit_view_of`). None of the three rows is `is_workflow_phase`, so `full` never runs
them. `utils/commit_range.py` owns the grammar and every rendered line (block, fix block, drift line,
result payload, read rule); `utils/commit_paths.py` the path layer (where a brief lives and what is
one); `utils/commit_front_door.py` the front-door core the three doors and `util commit-brief` share
(`utils/commit_review.py` is its old name, kept as an alias); `utils/commit_brief.py` the header bytes, identity, snapshot and
restart archive; `utils/commit_apply.py` the apply guard.

The core (`commit_front_door.prepare` for steps 1 to 6, `front_door` adds 7):

1. resolve the range to two full SHAs (`resolve_range`; a brief path reads `commit_range` back through
   `resolve_recorded`; a no-argument apply picks the review on record, `commit_apply.select_review`);
2. print the commit block, with the `[SCOPE_DRIFT]` line computed here on review-commits and
   improve-commits so it lands before any detach boundary;
3. build the brief header and compare its identity hash with the recorded one;
4. take ownership of the state the run changes;
5. write or reuse the brief (archiving a replaced round first) and print `[COMMIT_BRIEF] <path>`;
6. record `commit_range` through `StateManager.record_commit_range` under `state_file_lock`, creating
   `state.json` when there is none;
7. re-record the invocation and call the fronted `main` in-process.

**Steps 1 to 3 only read**, so a second invocation with another `--focus` cannot change a run in flight
before it is refused or re-attached; the one earlier write is the brief's empty output and phase dirs,
with a new folder's `.gitignore`. Ownership is the dir's `.lock` with no live detached run recorded
there (`detach.classify_run`): `<slug>/code-review/` for a header rewrite, whichever door asks;
`apply-fixes/` for the guard's writes; the run dir and both step dirs for improve-commits. In the
foreground the lock is taken non-blocking with `ImproveWorkflow.run_mode`'s retry; review-commits and
apply-commit-fixes hold it through the fronted `main`, which runs without `--internal-worker` and never
takes it. improve-commits holds the review dir's lock through `entry_point` too, handed to the composite
with `improve_workflow.holding`: its preflight does not count it as another run's, and `run_mode` runs
the review step under it instead of taking it again, so the round stays owned through the apply step.
`util commit-brief` holds it for steps 4 to 6 only. A refusal is the live-run exit 3 plus
`brief and state left unchanged`. **Under `--detach` the launcher publishes nothing:** it writes the
header, `commit_range` and any new context-plan snapshot to `<phase dir>/.commit-stage-<hex>.json`,
names it with the hidden `--commit-stage` (dropped from replay like `--revs`), and the worker applies
it right after `_acquire_lock` (`commit_front_door.apply_stage`), before the phase reads the brief.
`detach` sits below the front door, so the worker's entry hands it that applier
(`commit_front_door.hand_worker_the_stage_applier`): the dispatcher and, for a worker started directly, the
three orchestrators the doors run (`code_review_orchestrator`, `apply_code_fixes_orchestrator`,
`improve_commits_orchestrator`). A refused or re-attached
launch deletes its stage and a worker that loses the lock applies nothing; a brief that does not exist yet
is the exception, created by the launcher with an exclusive create. The restart's synthesised
`--force --rerun-all` never takes over a live run: in `detach.launch` only a typed `--force` does, which
the stage records as `typed_force` (`commit_paths.stage_takes_over`).

**What step 7 records.** Three readers take the invocation record as it stands: `maybe_detach`'s mode
check, the detached child's argv (`detach.build_worker_argv`) and `_emit_user_next_step`, which answers
only for a workflow phase or a composite. So review-commits records as `review-code` and
apply-commit-fixes as `apply-code-fixes`: the normalized argv minus the front-door-only tokens (the range,
`--focus`, `--context-plan`), plus `--plan-file <brief>` (and `--base-ref <base> --head-ref <head>` on the
review), the restart's `--force --rerun-all` and, under `--detach`, the stage flag, recorded only once the
brief and `commit_range` (or the stage) are in place. improve-commits records as itself, because
`entry_point` crosses its boundary as the row's own `MODE_NAME` and its run dir is
`<slug>/improve-commits/`; the argv carries the pinned range and no `--base-ref`/`--head-ref`, and the
`--internal-worker` child re-enters the front door, which is idempotent there (no block, ownership,
header or write; base and head from the stage) and synthesises them again. Because the doors synthesise
`--plan-file`, `--base-ref` and `--head-ref`, a typed one is `[COMMIT_FLAG_CONFLICT]` (exit 3, checked on
the typed tokens before `_normalize_positionals`), and the rendering hook keeps all three out of every
printed command, which names the commit mode and the pinned `<base12>..<head12>` (a root range's brief
path) instead.

**`commit_range` is written before the phase, never during it.** One writer outside the phase (step 6, or
the worker's stage) serves every door, and the phase carries the record unchanged through
`_load_or_create`/`save()` (its `--force` pops only `phases_completed["code-review"]`). The phase writes
its own fields through the `StateManager` it already holds, since a second instance saved later would put
an old record back: review-code sets `reviewed_header_sha256` and deletes `guard` when a round starts
(never when it only resumes missing models), and apply writes `guard`.

**The brief.** The slug is `<base12>-<head12>`, or `root-<head12>` for a range starting at the root commit
(no `..` in a filename, and never the folder's own name, which `find_output_dir`'s double-nesting guard
would catch). The folder comes only from `provider_registry.get_commit_reviews_dir`, joined to the anchor's
git root; its validation, self-ignoring `.gitignore` and `[COMMIT_REVIEWS_NOT_IGNORED]` are in
[docs/configuration.md](../configuration.md#good-to-know). The header sits between
`<!-- multi-llm:commit-brief v1 -->`, always line 1 (`is_commit_brief` reads it), and
`<!-- /multi-llm:commit-brief -->`; a rewrite keeps everything below the closing marker byte for byte, and
there is no `TASKS_FILE` marker. Commit text is data: each message is fenced one backtick longer than its
longest backtick run, every other `<!--` is written `<!- -`, and the caps (`BRIEF_MAX_COMMITS = 100`, 2048
bytes per message, 32 KiB of messages, 100 stat lines) each leave a marker naming the git command that
prints what was cut. The fencing and the caps are presentation only: neither enters the identity below.

- **Identity, not text.** `header_sha256` is the SHA-256 of the canonical JSON of
  `{base, head, focus, plan, plan_sha256}` (`commit_brief.header_identity`), with `plan` repo-relative in
  git's form and `plan_sha256` the snapshot's. The phrase, window, table, messages and stat are not in it,
  so `last 2 commits`, `-2`, the pinned spelling and the brief path reach one review, and `phrase`/`window`
  stay as first resolved. An omitted `--focus` or `--context-plan` inherits the record and only a typed,
  different value (`""` clears) changes the identity; replays never carry either flag. apply-commit-fixes
  never rewrites the header: a differing typed value prints `[COMMIT_REVIEW_KEPT]`.
- **The context plan is frozen** as `<slug>/context-plan.md`, written from the one read that sets
  `plan_snapshot` and `plan_sha256`. Every reader names the snapshot (the prompt's `{plan_block}`, both
  validation legs, revalidation), validation's `inputs` gain `context_plan_sha256`, and an inherited plan
  whose live file moved on prints `[CONTEXT_PLAN_CHANGED]` and continues on the snapshot. Fresh review,
  resume and reaggregate refuse a missing or changed snapshot. Validation and revalidation task files
  record `context_plan_file` relative to the phase directory; executor dispatch rechecks its raw-byte
  hash even when given a document already in memory. Explicitly repeating `--context-plan <path>`
  restores a damaged snapshot from the user's file, including when the identity is unchanged.
- **A restart archives the replaced round.** A changed identity rewrites the header only after step 4 and
  passes `--force --rerun-all` with `[COMMIT_REVIEW_RESTARTED]`. First, `commit_brief.archive_round`
  renames every entry of `<slug>/code-review/` except `RESTART_KEEP` (`.lock`, `.status.lock`,
  `.status.json`, `orchestrator-run.log`, `.views.lock`), the stage file and earlier archives into
  `restarted-<YYYYmmdd-HHMMSS>[.<µs>]/`, which the phase's non-recursive globs never read. `--rerun-all`
  alone would leave an old-set-only model's `<model>.json`, which `retire_result_from_another_scope`
  misses because a changed focus keeps the scope id, and apply would read the old `user_selections.json`
  against new groups. No decision is carried over, not even by group hash (reworded findings rehash,
  positional `G<g>S<s>` keys name other findings): `[COMMIT_REVIEW_ARCHIVED]` and
  `[COMMIT_DECISIONS_DROPPED]` (counted with apply's own readers) say what moved. Move first, header
  second, so a crash between them leaves the old identity and the next typed change restarts again.
  A `state.json` that exists but gives no record (it does not parse, is `[]`, or holds no range) is
  treated the same way when the header would change: nothing says which header the old round
  answers, so it is archived and restarted, and step 6 still prints `[STATE_REPLACED]`. Under
  `--detach`, `[COMMIT_REVIEW_RESTARTED]` and `[COMMIT_DECISIONS_DROPPED]` print only after the launch,
  once a worker has consumed the stage; a refused launch restarted nothing and says nothing.
- **Briefs are never under review.** `commit_paths.commit_review_artifact(path, folder)` matches
  `<folder>/<slug>.md` and anything under `<folder>/<slug>/` by string, and `_drop_workflow_paths` and
  `finalize_tracking._drop_workflow_files` drop those on every mode, whatever the run's plan.

**Two commits, drift and the pinned-head read rule.** review-code's `--head-ref` diffs `base..head`
everywhere (`SCOPE_SOURCE_COMMIT_RANGE`, `head_ref=` threaded through `git_utils`; `None` stays
byte-identical). A root range's base is the computed, never-abbreviated empty-tree id, and
`commit_range.diff_base` is the one converter. `pinned_endpoints(state)` reads `commit_range`, else the
`head_ref` record of a direct `review-code <plan> --base-ref A --head-ref B` run, and every endpoint reader
keys on it; recorded SHAs re-resolve strictly (`resolve_recorded`, `[RANGE_UNRESOLVED]`), never through
`validate_git_ref`'s `HEAD~1` fallback.

- **Drift is `git_utils.paths_changed_since(head, paths)`, the one change test**, shared with the apply
  guard so the drift list and the guard's skips never disagree about a file. Three reads, each one call
  over all the paths: `git diff --name-only --no-renames -z <head> --` (later commits, staged and unstaged
  edits, mode-only changes, deletions; no rename pairing, so every path is listed under its own name),
  `paths_in_commit(<head>, …)` plus an `lstat` (absent at `head` but present on disk, a non-regular
  entry), and `ls-files -v -z` (skip-worktree `S`, assume-unchanged lowercase). A failing read marks every
  path changed. The front door runs it at step 2 over the range's scope narrowed by `--files` /
  `--files-from <file>` (never `--files-from -`), prints `[SCOPE_DRIFT]` in the block and records
  `commit_range.drifted`; the fronted review-code reads that and never recomputes, and a direct run checks
  in its own pre-flight and records `head_ref.drifted`. Drift warns; the run continues.
- **The pinned-head read rule** (`commit_range.pinned_head_read_rule`, followed by `DATA_UNDER_REVIEW`):
  every code read a model makes goes to `head` (`git show <head>:<path>`, `git show <base>:<path>` for a
  path the range deleted or renamed, `git ls-tree`, `git grep`), never `cat`, `rg` or a file-read tool on
  the working tree, whatever drift found, because drift sees only the scope files and only at launch. It
  renders in the review prompt's `{diff_access}` (both templates), both validation legs and the
  revalidation tasks. It is an instruction, not enforced at read time; Python's own reads use `head`
  (excerpts through `read_at_commit`, report context through `capture_file_snapshots_at`, blame), and
  attribution blames a drifted file's group only when its `anchor_text` is there at `head`
  (`model-drifted` otherwise).
- **Pinned attributes.** Diffs, hunks, range statistics and rename detection use
  `git --attr-source=<head>` (Git 2.41+) so later `.gitattributes` cannot hide a text diff or
  change its counts. Older Git uses a temporary private index loaded from `head` and an empty
  worktree; the real index and checkout stay untouched. `_linguist_generated` uses
  `git check-attr --source=<head>` (Git 2.40+). On an unknown-option failure it uses the same
  pinned attribute fallback, with one warning naming the Git version. Other classification
  failures classify nothing by attribute, with one warning.

**The apply guard** (`commit_apply.take_guard`) is keyed on `pinned_endpoints`, so it runs for
`--apply-commit-fixes`, improve-commits' apply step on both legs, a plain `apply-code-fixes <brief>` and
`apply-code-fixes <plan>` after a direct `--head-ref` review. Its hook in `utils/apply_orchestrator_base.py`
sits right after `filter_items`, before the no-items exit, human decisions and batching, and prints
`[COMMIT_FIXES]` from the decision it saves under the pinned record's `guard`, which `status` and the result
line's `commit_fixes` read too. A new review round removes `guard`, so the next apply decides afresh.

- **Per group, never partly applied.** A group naming the context plan is skipped first
  (`context plan <path> is read-only in this review`), then one with a fix file `paths_changed_since`
  marks, then one naming a protected path. A symlinked context plan keeps the link's own name, so the
  in-repository targets of its chain are saved with their disk ids (`guard.context_plan_targets`,
  `commit_apply.context_plan_paths`): a fix naming one is skipped as the context plan, and both write legs
  watch and deny them like the plan. Granularity is the whole file: a fix is prose a model carries
  out, not a patch to `git apply --check`.
- **Expectations, never approvals.** `guard.expect[path]` is the content a passed file must still have:
  `{"kind": "head", "blob": <id> | null}` from `git_utils.blobs_in_commit` (`null` is absent at `head`, so
  a creation or restoration fix can land), re-pinned after each completed batch to
  `{"kind": "batch", "sha256": …, "task_id": …}` (the executor's audit `sha256_after`; natively
  `util batch-done --status completed`). A blob id and a SHA-256 are compared only with their own kind.
- **Every batch is re-checked right before dispatch**, since it can start long after the first decision
  (after judging, earlier batches, a stopped batch or a `--resume`): `commit_apply.recheck_files` runs in
  `write_batch_scheduler._dispatch_one`, in `util write-prompt` natively, and over every group left on a
  `--resume`. A failing batch is not dispatched (`[COMMIT_FIXES_STALE]`; natively no prompt, exit 0), its
  groups naming a failing file are skipped, those files join the protected set, the batch's other groups
  are held back `failed` through `_refuse_before_dispatch` for a `--resume` to re-batch, and the run goes
  on.
- **The protected set** (`guard.protected`, `{path: blob id | null}`) is every path that differs from
  `head` anywhere in the repository (`git diff --name-only <head>` plus untracked, non-ignored files
  and skip-worktree/assume-unchanged entries, whose disk copy `git diff` never reads, bar a sparse
  checkout's absent ones; workflow paths left out), taken once and reused on `--resume`, because a
  batch writes repo-wide and a fix in unchanged `a.py` can still edit drifted `b.py`. `code_only_rule`
  names it (20 paths, the rest in `apply-fixes/protected_paths.txt`), a fix naming one is skipped, and
  after each batch the blob ids are compared: the executor's exact-path deny layer in `WritePolicy`
  fails the batch and restores the
  pre-batch snapshot, and natively `util batch-done` refuses with `[COMMIT_PROTECTED_WRITE]`, naming the
  copies the guard kept in `apply-fixes/protected/`. Like the executor's rollback store, the guard never
  copies a path `driver.write_paths.deny` names (`commit_apply.copy_protected`), so a credential in the
  protected set is not duplicated into the run's artifacts: it stays watched, and its way back is by hand.
  On the executor that comparison does not depend on the
  post-run undeclared-write sweep (which is skipped over `SWEEP_MAX_FILES`): after every batch the protected
  paths and the context plan are re-hashed against their ids from just before that batch, and an offender is
  restored from the snapshot, else from the guard's copy, else deleted when it was absent. The ids are taken
  per batch, so a user's own edit between batches is never rolled back. Natively the same window runs from
  `util write-prompt`, which records the watched ids in `guard.bound[<task id>]` as it binds the prompt,
  to `util batch-done`, which compares with them (the saved ids when the prompt was not bound through it)
  and names the guard's copy as the way back only when it holds what the path held before the batch. A
  `--resume` drops the bind records and refreshes the context plan's saved id and copy. A watched path
  that cannot be read is named once in `[COMMIT_FIXES]` (`guard.unreadable`) and its groups skipped; only
  a move into or out of readability counts as a change. The runner's `repo-wide` grant is
  unchanged: this is a detective deny layer, not a narrower sandbox.

## Ask mode

`ask` mode (`ask_orchestrator.py`) is a read-only free-text Q&A side-channel:
markdown output, no JSON schema / grouping / validation, output under `{plan}/ask/<question-slug>-<hash8>/`. It
shares concurrency mechanics with the review modes via `run_models_concurrent` in
`review_orchestrator_base.py`, and its model-selection preamble with them through
`interactive.select_models`.

`_main` is a sequence of stages, each with its inputs and outputs explicit: `_resolve_inputs`
(question, plan, hash) → `_accept_partial_entry` or `_prepare_fanout` (models, question dir,
prompt parts, all in the foreground) → `_cross_detach_boundary` → `_hold_question_dir` (the
foreground lock, held until `main` returns) → `_record_launch` → `AskRun`. `AskRun` holds what
the slots share: `skip_existing_answer` and `ask_one` are its methods, `close` sets the deadline's
answers aside and writes `answers.md`, and `report` prints the closing report.
`tests/test_ask_run.py` drives `ask_one` directly.

Each ask slot is single-shot: one `invoke_with_provider` call, no retry ladder. A call that
raises is a failed slot carrying `Exception: <type>: <message>`, redacted
(`slot_exception_error`), with `source: exception`, never a deadline loss. It first gets the
checks a returned call gets: the porcelain check on the uncontained rung, and the mode and home
checks on the plugin's jail, with the raising call counted as unaudited and its home compared by
the lane (see [providers](providers.md)). A violation found there is recorded instead, still with
`source: exception`. Any answer file the call left is set aside as a partial, never counted.
Cancellation (`CancelledError`, `KeyboardInterrupt`) is not caught: the fan-out's teardown retires
the slot. `run_models_concurrent` names a slot that raised `CancelledError` itself and keeps its
siblings' results; the slot is missing from the results like one that raised.

A slot any of these checks fails has its answer file moved to `answer_<model>.md.violated`, a
name the closing scan never reads. When that move fails the file stays where it is, as
evidence, and the scan withholds it (`answers_left_in_place`): the slot stays failed, and its
failure line says the file was left in place.

**Earlier and unfinished answers.** Only text a slot wrote itself can count as its answer.
Past the sandbox gate, before its dispatch, the slot removes what an earlier run left: the
`partial_<model>.md` (`remove_earlier_partial`), the `origin_<model>.txt` (`remove_earlier_origin`),
then, under `--rerun-all`, the `answer_<model>.md` (`remove_earlier_answer`). Each removal is
attempted directly: only its own `FileNotFoundError` shows the file absent, and any other error
ends the slot `not-dispatched`, naming the file to move. A refusal keeps the earlier answer: the
answer goes last, so a refusal over the partial or the origin record never reaches it, and a
refusal over the answer writes the origin record back as it was read before its removal
(`put_back_earlier_origin`; a write that fails too leaves the answer reading "origin not
recorded"). A partial is never the last run's text when an earlier answer or its origin record is
there, since the slot that answered removed it before it was asked. A slot that went ahead would
leave the earlier partial named as its own, for `--accept-partial` to count; having written
nothing, count the earlier answer as its own; or, when its own origin record failed to write,
show its answer with the earlier one's origin. For a model nothing withheld, the record the
slot's arm wrote is taken out again (`restore_unarmed`): the earlier answer passed its checks in
its own run, so a resume keeps it, with its own origin record. `--rerun-all` asks every model for
a new answer, so `close` never counts an answer it did not produce: a model it never dispatched
(refused at the sandbox gate or over a file it could not remove, unrecorded, or still queued at
the deadline) fails with its own reason, its earlier answer named as kept and not counted, and
the run's state and exit code follow (`failed`, exit 3, when no model answered).

The resume skip reads the answer file directly (`probe_answer`) and tells three outcomes apart:
missing or empty asks the model again, readable skips it, and a read error, or bytes that are not
UTF-8 text, ends the slot `not-dispatched`, the file and its origin record left for a later
resume. A slot asked over an answer it could not read would, having written nothing, read it at
step 1 once readable and count it, with a new origin record, as its own. No answer read raises:
an unreadable file is never counted, and the closing scan goes on to the other models' answers.

A stdout answer is written to `answer_<model>.md` atomically, and so is `answers.md`
(`publish_atomically`, which keeps the file's mode, unlike the owner-only records): a write that
fails, a full disk say, leaves the path as it was, never a cut prefix that the closing scan or a
resume would count. The slot then fails with that error, its whole text kept as the partial for
`--accept-partial`. A file the model wrote at the answer path as well, one step 1 could not read,
never moves over that partial: it stays in place, withheld with `UNFINISHED_ANSWER_REASON`
(below), and takes the partial's name only when the partial could not be written
(`unpublished_answer_ending`). Because `answers.md` is replaced rather than written through, a
symlink at `answers.md` is replaced by a regular file: a report linked to another place stops
being updated there, and the link's target is left as it was. `tests/test_ask_answer_persistence.py`
holds the cases.

**The close uses the answers its checked slots held.** A slot that answered with its checks
passed keeps the text it counted: the answer file as step 1 read it, or step 2's recovered answer
as a text-mode read of the bytes `publish_atomically` wrote (`AskRun.checked_answers`). The
closing scan counts that text and `answers.md` shows it, never the file read again, and every
other answer file is read once for both (`collect_completed(held=...)`,
`write_answers_md(sections=...)`), so a provider a crashed run left behind cannot replace an
answer between its slot's checks and the report ([Consuming checked
bytes](review-results.md#consuming-checked-bytes)).

**Withheld answers.** An answer is reused only from a slot whose safety checks passed, as on
the review lanes ([Withheld slot answers](review-results.md#withheld-slot-answers)): the
question folder's `.status.json` keeps the same two maps, `models_rejected` and
`models_unchecked`, through the same `result_files` helpers. `_record_launch` carries both maps
across its opening write, every model's, read whole under the lock that write holds, then reads
the checkpoint back; over a checkpoint it cannot read (after `CHECKPOINT_READ_ATTEMPTS` reads;
a damaged one was recovered just before, below), or one it cannot read back, the run stops at exit
3 before anything is dispatched, with `status_file.unreadable_checkpoint_advice`. Each slot that passes the sandbox gate is recorded
under `models_unchecked` before anything is dispatched or removed (`review_lane.arm_slot`), in
the question folder's checkpoint whatever its resolved answer path points into; one whose
record cannot be written and read back, or whose launched run has no checkpoint left to record
in, ends `not-dispatched` with `[WITHHOLD_NOT_RECORDED]` (only a direct call of `ask_one`, with
no checkpoint at all, runs unrecorded). A model already withheld has its earlier answer file set
aside first (`answer_<model>.md.violated` for a rejection, `.unchecked` otherwise) and its
earlier `partial_<model>.md` removed. The rename is attempted directly: only its own
`FileNotFoundError` shows the file absent, so a metadata error never passes for one. When
either step fails, the slot ends `not-dispatched` and asks nothing over it; a partial left
behind would otherwise outlive the records the slot supersedes, and `--accept-partial` would
count it without a warning. A slot the checks fail records `models_rejected` before it sets
its file aside (`reject_answer`). One whose checks pass leaves `models_unchecked` as it ends
and is marked `checks_passed`: since the earlier answer was set aside before its dispatch,
whatever it leaves is its own, so it supersedes the model's records, succeeded or not. The
exception is a failed run whose unfinished answer can be neither moved nor copied aside: it
stays under `models_unchecked` with `result_files.UNFINISHED_ANSWER_REASON` ("left an
unfinished answer"), so a resume asks the model again and only `--accept-partial` counts that
text. A dispatched slot that never returns (the deadline cut it, or it raised outside the
provider call) keeps its record, and `close` writes one again when it is gone; a provider
call that raises is not such an ending, because its checks still run. So does a slot
cancelled at its launch turn, after its record and before any provider call: nothing
supersedes it. The close writes both maps (`review_lane.withheld_record`).

Every reader honours them. The resume skip asks a withheld model again rather than count its
file; the closing scan leaves the file of every model the maps name unread, a model this run
refused at the sandbox gate or never reached included, and its failure line says so; over a
checkpoint that cannot be read, only this run's checked slots and resume skips are counted.
When a refused, unchecked or unfinished answer is still at its answer path and no record of
it can be written, the close stops at exit 3, `needs-human`, before `answers.md`. Two review-lane rules
have no ask counterpart: there is no salvage pair to retire, and no unjoined provider call,
since `stop_unreturned_provider_calls` waits for every call it stops (its own timeout bounds
the wait). `--accept-partial` stays the operator's opt-in: it refuses over a checkpoint it
cannot read, and while a withheld model's answer file is in place; a withheld model's partial is
counted only with a `WARNING` naming why it was withheld, once both its records are cleared
(`result_files.clear_withheld_model`). A readable checkpoint written before these maps
withholds nothing, so an older question resumes as it did. `tests/test_ask_withheld_answers.py`
holds the cases. Ask recovers a damaged question checkpoint as the review lanes do, with their
function (`_recover_question_checkpoint`, `review_lane.recover_damaged_checkpoint` with this
question's `question`, `question_slug` and `plan_hash`; [Recovering a damaged
one](review-results.md#withheld-slot-answers)): by the process that holds the question lock, the
foreground run after `_hold_question_dir` and before `_record_launch`, the detached worker under
the lock its launch handed it (`detach.PhaseRun.take`) before its boundary, where `RunHandle`
claims the run in the checkpoint. A checkpoint that cannot be read and is not proven damaged stops
the run at exit 3 before anything is dispatched. The two reads before the lock, the question-dir
resolver's `question` and the plan-freshness check's `plan_hash`, go through
`status_file.read_status`: a checkpoint they cannot read (not JSON, not an object, not UTF-8, not
a regular file) gives them no metadata, never an exception, so every run reaches the recovery. The recovery's result glob sets no answer file
aside (they are `answer_<model>.md`); the fresh checkpoint's `checkpoint_recovered` record
withholds them all instead, and `_record_launch` carries it over like the maps, so the resume skip
and `--accept-partial` keep withholding every model by it: each selected model's earlier answer is
set aside as `.unchecked` before the model is asked again (`set_withheld_answer_aside`), and the
closing scan counts no answer file but one a slot of this run produced with its checks passed
(`withheld_after_run`); `--accept-partial` then records the model under the record's `accepted`
(the operator's choice, never a passed check) and reads it back before it reports `[ACCEPTED]`
([Withheld slot answers](review-results.md#withheld-slot-answers)).

## Implementation artifacts and state

**`implement`'s write phase closes the artifact/resolution/denominator gaps the apply
phases already closed for themselves.** `run_implement_write_phase` now seeds every materialized task
`pending` in `state.json` before the scheduler loop starts, so
`check_workflow_prerequisites`'s task count reads the plan's total rather than however many happened to
finish or be skipped so far (F5); records its own artifacts the way `ApplyPhaseReport.record_write_artifacts` does,
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
`WriteBatchScheduler` takes no such argument — so `WriteReports.write_report` now names it directly:
`write_task_file_for_phase` always persists the same materialized list at `{phase_dir}/write_tasks.json`, which
every write orchestrator passes as the scheduler's own `phase_dir` too (N9). 2026-09-16.
