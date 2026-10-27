# Workflow orchestration

Read when changing task generation, full or composite workflows, commit reviews, ask mode, or implementation bookkeeping.

Back to the [development guide](../../skills/multi-llm/AGENTS.md). These notes retain detailed contracts and regression history from that guide. Paths beginning with `docs/`, `packaging/`, or `plans/` are repository-relative; other implementation paths are relative to `skills/multi-llm/`. Dated incidents explain the rules; verify version-specific claims against current code.

## Contents

- [Task generation](#task-generation)
- [Full workflow state machine](#full-workflow-state-machine)
- [Improve-plan and improve-code](#improve-plan-and-improve-code)
- [Commit reviews](#commit-reviews)
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
result payload, read rule); `utils/commit_review.py` the path layer and the front-door core the three
doors and `util commit-brief` share; `utils/commit_brief.py` the header bytes, identity, snapshot and
restart archive; `utils/commit_apply.py` the apply guard.

The core (`commit_review.prepare` for steps 1 to 6, `front_door` adds 7):

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
it right after `_acquire_lock` (`commit_review.apply_stage`), before the phase reads the brief. A refused or re-attached
launch deletes its stage and a worker that loses the lock applies nothing; a brief that does not exist yet
is the exception, created by the launcher with an exclusive create. The restart's synthesised
`--force --rerun-all` never takes over a live run: in `detach.launch` only a typed `--force` does, which
the stage records as `typed_force`.

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
  `.status.json`, `orchestrator-run.log`), the stage file and earlier archives into
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
- **Briefs are never under review.** `commit_review.commit_review_artifact(path, folder)` matches
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
