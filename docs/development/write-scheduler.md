# Write scheduling and batch outcomes

Read when changing write scheduling, retries, partial application, declined items, counts, prior-batch context, or result parsing.

Back to the [development guide](../../skills/multi-llm/AGENTS.md). These notes retain detailed contracts and regression history from that guide. Paths beginning with `docs/`, `packaging/`, or `plans/` are repository-relative; other implementation paths are relative to `skills/multi-llm/`. Dated incidents explain the rules; verify version-specific claims against current code.

## Contents

- [Write scheduler contract](#write-scheduler-contract)
- [Retry after edits](#retry-after-edits)
- [Declined items and partial applications](#declined-items-and-partial-applications)
- [Dispatch counts](#dispatch-counts)
- [Prior-batch context](#prior-batch-context)
- [Write reply extraction](#write-reply-extraction)

## Write scheduler contract

**The write half is a *scheduler*, not a dispatcher (D3f/D3g/D3h/D3l/D3n).** `utils/write_batch_scheduler.py`
drives the four write kinds (`apply_suggestions`, `apply_task_suggestions`, `apply_code_fixes`,
`implement`, added to `subagent_tasks.KINDS` and the schema alongside the six read
kinds), under `--exec-subagents` *and* — via `respond=` — under the simulated native
invoker, so both paths schedule identically.

One loop: batches in ascending `batch_index` with each batch's verification **completed and
recorded before the next dispatch**; `{prior_changes_context}` bound **late, in Python, per dispatch**
from the log on disk (the literal token reaching a runner is a refused dispatch, and the hash-bound
task file is never rewritten — `implement` prompts carry the token too since F6
(2026-09-16): the dependency section named only a preceding task's *title*, so a verified one-line
summary per finished task now reaches the next task's dispatch the same way an apply batch's does,
rather than the bare titles alone); the render is also scoped to the dispatching phase now (N8,
2026-09-16): `prior_changes.render(output_dir, phase=self.phase)` prefers that phase's own entries when trimming to the entry
cap/`MAX_CHARS`, because the log is one file shared by every write phase a plan has ever
run and an unscoped tail let two of implement's eleven visible slots go to `apply-task-suggestions`'
older batches instead of implement's own; unlike that token, `apply_task_suggestions_orchestrator.py`'s batch prompts
used to also embed a **frozen** `tasks_content[:3000]` excerpt of tasks.md, identical across all
batches since every prompt is materialized up front — by the last few batches of an 11-batch run it
showed a dependency-graph edge an earlier batch had already changed, so it is gone (F7, 2026-09-16):
each prompt just tells the runner to read the live file first, which step 1 of its own instructions
already said; the write-batch **result contract** appended at the same seam on both invokers
(`_with_result_contract`: the `write_batch_result.schema.json` shape plus every `expected_ids` member
spelled out in prompt order, because the prompt never names them and an omitted id reads as a
partial application) — the batchers' prompts end in the prose "Return Format" the native markdown
flow compresses into a summary line, and measured 2026-09-15 a real runner applied a batch, reported
it exactly as asked, and was recorded `unparseable_result` four attempts running; a resumed batch
**reuses** its pre-batch snapshot (`_reusable_snapshot`, same declared set and, when the manifest
records them, the same `expected_ids` — after a selection change an apply id names other
items, and the snapshot it holds is set aside as `pre_batch/<id>.items-<sha8>`, never deleted, while the
new batch takes its own — and moved back into the slot when those items return to the id
(`TaskSnapshot.reinstate`; a re-selected item used to re-snapshot the post-edit bytes) — never across
review cycles: a slot or set-aside copy whose manifest `taken_at` predates the review
this phase applies (`_review_completed_at`, the `earlier_rows` cutoff; the stamp is kept
across re-saves) is a previous cycle's rollback point, set aside and never reused, since restoring
from it would roll that cycle's applied edit back too (2026-10-03); a renumbered batch whose items
already landed is skipped on the earlier result's verdict stamp, which names its
`expected_ids`, `task_id` and `inputs_fingerprint` (`_sibling_results`,
judged with the index it ran as), and a dispatch reusing a positional result name keeps another
batch's result beside it as `apply_batch_N.items-<sha8>.json` first (`_retire_foreign_result`; adversarial
review, 2026-10-02)) so the rollback point stays the bytes from before the batch's *first* dispatch
rather than the previous run's half-applied edits — measured 2026-09-15, four resumed runs each
re-snapshotted the last run's partial edits and `util restore-batch` could no longer reach the
pre-batch file — and `apply-task-suggestions` keeps `tasks-backup.md` across a resume for the
same reason; the human-task strategy resolved **once** (`--force-strategy` →
`state.json` on resume → `--yes`'s `skip-dependents`) then applied as
a graph walk (`skip-dependents` = transitive BFS over `depends_on`,
`skip-continue` = `record_dependency_override` per unblocked task, `pause-and-ask` = exit 3
naming both `--human-task <id>=done|skip`); `prior_changes.append` → `state.record_metric` →
`update_task_status` → one log line per batch **in that order** (the append must land before batch
N+1 substitutes, or the injected context lags a batch; every other `self.log` inside the
loop is exceptional, so that line is what makes a long run's log readable) —
`state.json`'s `task_status` is one flat map shared by all four write kinds, so
`implement`'s own `T###` ids sit beside
`apply-batch-N`/`apply-fix-batch-N`/`apply-task-batch-N`; a reader wanting implement's
task progress calls `state_manager.implementation_task_statuses()`, which filters to the `T###` shape, never
the raw `get_all_task_statuses()` (F4: `check_workflow_prerequisites` used to sum the whole map, so a run that
had only executed apply-suggestions batches read as completed implement tasks, and one failed apply
batch flipped `implement` to `in_progress` before any task existed); and
**finalization is part of the contract** — `driver.verify_command` (F4: the repo's own check, run
once with cwd at the pinned root before anything is marked completed, for the two code kinds only;
unset is *logged*, not silent, and a non-zero exit reverts nothing but costs the run its completion
at exit 3, recorded as `verification: {command, source, exit_code, duration_seconds, output_tail}` on `SchedulerResult`, in `write_batch_report.json`,
in the `.status.json` write mirror and as a `Verification` line in both summaries —
the command and its config layer are logged **before** it starts as well as after, which is what
`docs/configuration.md`'s safety section promises of the one config key that executes; it is spawned
with `Popen` in its own process group and a timeout kills that whole group through
`process_control.kill_tree`, because `subprocess.run`'s timeout SIGKILLs only
`/bin/sh` and left the real `pnpm`/`node` running while
the phase reported `exited 124` (F4/F3b, 2026-09-17), and its capture goes to a temp file
read back to the last `VERIFY_OUTPUT_CAP_BYTES` rather than buffering a whole build in memory, under
the environment `providers.base.build_child_env` composes for a spawn rather than a bare
`os.environ` — before it, seven code fixes were applied to one file and the phase
completed without any check, the runners' volunteered `node --check` being their initiative
and not the phase's guarantee) + `metrics.finish` + `mark_phase_completed` +
`finalize_tracking` + `state.reload()` + the implementation summary (with its
`## Manual Tasks (Deferred)` / `## Dependency Override Warnings` / `## Cancelled` sections) +
`insert_implementation_summary_reference`.

A run that dispatched every batch and produced none of those is a **failed** run.
**`state.json` has exactly one writer per run, and it is the scheduler's
`StateManager`** (F3/F4, 2026-09-16): `StateManager.save()` dumps its dict wholesale and
never merges, so anything a *file-level* writer added between two saves is erased by the second one.

Metrics therefore go through `state.record_metric`/`start_metrics_phase`/`finish_metrics_phase` —
the in-memory twins `metrics.py` now exposes beside its CLI-facing file-level pair
(`record_metric_in_state` and friends), with `start_metrics_phase` called once before the first
dispatch so `metrics_progress` and the `[ETA]` line `instructions/implement.md` step 5
describes exist on the executor path at all (the `batch_index`/`total_batches` that
call passes are this run's own progress and dispatchable count — `_eta_progress`: the
finished tasks of the set `_eta_task_ids` counts, and that set's size — never the plan's
global decomposition index or the bare materialized count including a deferred human task, which is
what made a resumed run's `[ETA]` line read `batch 7/6 … WARNING: batch_index > total_batches` from its first
batch on, N1 2026-09-16; **both halves count tasks**, because `len(batches_dispatched) + 1` was a batch
ordinal against a task denominator and an `implement` dependency level with k tasks is
one batch of k — every one of those k entries recorded the same ordinal, so a parallelizable plan
ended at `batch 9/14`, never reached `N/N` and never printed
`ALL BATCHES COMPLETE`, F3 2026-09-17 — and a resume skip counts too, the same way
`_mirror_status` counts it in `completed`).

`metrics.finish_phase_in_state` **returns** `(elapsed_ms, line)` instead of printing the closing
`[ETA] … phase complete` line, mirroring `record_metric_in_state`, so the CLI leg prints it and the
scheduler logs it into the phase log the rest of that run's ETA lines went to (F7).

`finalize_tracking.main`'s return code is no longer dropped: a non-zero census is a warning in the
phase log **and** a message on the result, since `reload()` would otherwise keep a stale
`tracked_files` silently and `review-code` would review it (F4, 2026-09-17); that
script's own state write is temp + `os.replace` now, as its comment always claimed — and
the one remaining second writer, `finalize_tracking`, is followed immediately by
`StateManager.reload()` (re-read the file, discarding anything unsaved — callers save first) so the
closing `save()` carries its `tracked_files` instead of overwriting them.

Before this, an implement run whose log said `Tracked 1 implementation files: index.html` finalized with the two stale
entries earlier phases had left, and three audited runs finished with no `metrics` key
at all after every batch had recorded one. CI missed both because `tests/test_write_executor.py`'s
`FakeState.save()` was `pass`; it writes for real now.

The four D3g outcomes are their own axis (`TaskOutcome.write_outcome`), because the model's claim is
never the verdict — a mandatory per-task pre-batch snapshot under `{phase}/pre_batch/{task_id}/` decides by
byte comparison (never `git`: the target file is frequently untracked, which every
`git diff` variant reports as unchanged): **applied**, **no-op** (`no-op`
claimed and no bytes moved — the claim and the verdict agree, so it is recorded as
`(no-op — …, verified no file changes)` and the loop continues; measured 2026-09-16, two verification-only implement
tasks were labelled clean failures and their prior-changes entries told the next batch the work had
*failed*; and a `no-op` claim whose bytes DID move — T006, 2026-09-16 — still records
`applied`, since that is what `util restore-batch` needs, but now sets
`reason` naming which of two things happened: the pre-batch snapshot was reused from
an earlier dispatch of this same batch, so the diff is that attempt's, or this dispatch's own fresh
snapshot disagrees with the claim, a foreign-write signature that also warns on the log — N2),
**clean failure** (`applied` claimed, no bytes moved anywhere → since fix round 14
never `ok`: the batch is sent once more, its prompt plus `subagent_prompts.build_nothing_changed_resend_prompt`'s
note that the last attempt changed no file — the ladder's rule 9, `RESENDS_BY_KIND`, asked
before `after_result` through the scheduler's read-only `nothing_changed` hook, never
for a check task, a crash, a jail denial or a tree the census could not see — and a second unchanged
`applied` is `STATUS_BATCH_FAILED` with `applied_without_changes`, recorded
`failed`, the loop stopping and `--resume` dispatching it again; OpenCode
racing run 5's T016 was recorded `completed` and listed "Done and kept" while the
function it described never existed, and a stamp from before the fix is re-dispatched, not resumed;
declared files unchanged while others moved is judged by the ids like any moved file, applied or
partial), the **failed batch** (`failed` claimed, or a non-zero exit, and no bytes
moved → `STATUS_BATCH_FAILED`, not `ok`: nothing to restore but the loop stops at
the boundary and `--resume` re-dispatches it, and `_already_done` accepts only an
`applied`/`no-op` result file, because a schema-valid
`{"status": "failed"}` used to read as a clean failure at `status: ok` and the phase
completed at exit 0 — and only one whose own `ids` intersect the task's
`expected_ids`, the second half of F1's defence: a resumed `implement`
re-materializes from `format_batches_for_output(skip_completed=--resume)`, so its list is SHORTER and a positional result name
re-pointed every survivor at a predecessor's file — measured 2026-09-16, T006 adopted
`implement_batch_0.json` (`ids: ["T001"]`), had no fingerprint of its own to compare, and five
never-dispatched tasks were reported `applied` at exit 0; the first half is
`subagent_tasks.implement_output_name`, which keys an implement result by the plan task's id the way
`human_decisions.verdict_output_name` keys a verdict by `group_id` (the executor's
`implement` pattern admits both spellings, and a resume skip now appends its own
`skipped` line to `subagent_audit.jsonl` naming the file it trusted and the ids that
file attests to — the branch returned before every audit site, so the artifact whose job is
answering "what did this phase do" said nothing about the five).

The rename itself leaves the OLD positional files sitting beside the new ones with nothing to clean
them up; `_warn_stale_result_files` (N6, scoped to `implement` — a positional kind's
filenames have never been renamed) logs one warning naming any that match neither a current task's
`output_name` nor any task id `state.json` has ever recorded a status for, and
never deletes them.

Resume reads the scheduler's **verdict**, never only the runner's claim (F1, 2026-09-16): every
`_classify` branch ends by stamping `verdict: {write_outcome, status, ok}` onto the result file
(`_stamp_verdict`; additive, the schema admits it), `_already_done` re-dispatches a
file whose verdict is not `ok` or whose task `state.json` records
`failed`, and the inputs fingerprint is recorded only for an `ok`
outcome — a runner that printed `applied` with every id and then died mid-edit left a
file resume skipped as done.

A resume skip also moves the task's `task_status` to `completed` (reason
`resumed from <file>`, no metric, state-tracking F5), implement's resumed materialization keeps
`failed`/`in_progress` tasks beside `pending` ones (F2: a
cleanly failed task was dropped from the resumed run and the phase completed without it), and
`driver.verify_command` re-runs on a resume that dispatched nothing whenever `state.json`
records an applied batch for the phase (`_phase_has_applied_batches`, F3: the `--resume` the
exit-3 message prescribes after a failed check used to stamp the phase completed without running
it).

That resumed materialization also keeps a **deferred human task** (status `skipped`,
not `pending`), or the run loses `skipped_tasks` and the summary's
`## Manual Tasks (Deferred)` section entirely), **partial application** (bytes moved but the batch
reported failure or only some `expected_ids` came back → stops the loop at the batch
boundary, snapshots kept, phase not completed, exit **3 even under `--yes`**),
**schema rejection** (rungs 5/6 — prose or the wrong object, never counted as applied; one whose
batch moved bytes, and any other failed dispatch that never reached `_classify` with
moved bytes — the report re-ask failing again or timing out — is a partial application,
`_partial_when_bytes_moved`, fix round 9).

Finalization hands every file an `ok` outcome touched to `finalize_tracking.py --touched`
(repeatable): that script subtracts `pre_existing_changes` from the current git diff, which drops a
file the run edited *and* that was already dirty when the phase started — measured 2026-09-16, nine
batches edited only an `AM index.html`, `tracked_files` named one stray session
export, and `review-code` would have reviewed that instead.

`git_utils.get_modified_files()` runs plain `git status --porcelain` (unquoting the paths git quotes, so a
filename with a space is stored as itself and not as `"sp ace.txt"` — F6, 2026-09-16), so an
untracked *directory* stays one compact `?? some_dir/` entry, and **the subtraction is
prefix-aware** (`git_utils.subtract_baseline` / `baseline_covers`, used by `finalize_tracking` and
by `review-code`'s changed-file filter): a baseline entry ending in `/`
covers every path beneath it.

Only what SURVIVES the subtraction — a directory the phase itself created — is expanded back to its
files (`git_utils.list_untracked_files`, called from `finalize_tracking`), because a directory path has no
per-file diff and `review-code` silently reviewed none of a new module on the native leg
(F2, 2026-09-16); a baseline-covered directory is never expanded, which is what keeps the 1
511-entry regression below from recurring. Censusing with `-uall` instead was tried
and reverted — it disagreed with the compact baselines already stored in every
`state.json`, so nothing was subtracted and all 1 511 untracked files in a repo landed in
`tracked_files` (295 KB of state, and `review-code` fed 41 audio thumbnails instead
of the plan).

An untracked file has no "modified" state in git, so a file-granular census cannot spot an edit
inside an untracked tree anyway; `--touched` is the authoritative signal.
`finalize_tracking` **replaces** `tracked_files` rather than unioning it, so re-running
`util finalize-tracking --touched …` repairs a bad list (each entry's `action` is
`created`/`deleted`/`modified` read off
`head_before_implement`'s tree, else `HEAD`'s (the empty tree when `HEAD` has no commit yet),
and the baseline —
`_file_actions`, 2026-10-05; it was always `modified`, so a created
`index.html` read as an edit), and it prints at most 20 paths (state.json holds the rest).

The declared set of an implement task comes from `task_decomposer._extract_files`, and its
`**Files to modify**:` / `**Files to create**:` lines (what `plan_updater` renders for every
generated task) win outright over the keyword heuristic, which is for a hand-written task without
them: measured 2026-09-16, the heuristic read `CONFIG.boost`, `Camera.update`,
`HUD.update` and `e.preventDefault` out of a task's narrative as files to *create*, so
the executor snapshotted, swept, prompted for and reported on four files that do not exist.
`update_plan_tasks` therefore renders **both** lines for every task, `none` when
the list is empty — an omitted line is read as "hand-written task, guess" and is the one way a
*generated* task reaches the heuristic.

`**Complexity**` is read the same way (`_explicit_complexity` before `_estimate_complexity`,
which now judges the stripped narrative since the word "Complexity" itself contains the heuristic's
first high indicator): it was the only field of the JSON → markdown → JSON round-trip that did not
survive, and all 14 tasks of an audited `tasks.md` came back `high`,
contradicting the colours of the mermaid graph in the same file. The sweep's "no rollback point"
warning names the files it could not retain and why (up to five, `_skip_reason`): a bare
count left the reader guessing whether the gap mattered — measured 2026-09-16, two runs printed
`3 file(s)` and it was the three under `.multi-llm/`, denied by design.

A file `driver.write_paths` denies is left out of that warning and named once on the closing line
below instead, since a write to it already fails the batch on the policy check (racing run 4: the
same `.multi-llm/providers.yaml` warning opened every write phase). Once per run, after the sweep is done
and before the store is torn down, one more line reports overall utilisation against the budget
(`rollback store peaked at N MB of the 64 MB budget (files, shards); discarded now — the restore points are the pre-batch snapshots in <safety dir>/pre_batch/`, the directory named in full because a jailed repo-wide batch keeps them
under the state home, racing run 4), so degradation toward the cap is visible before files start
being skipped rather than only once they are, and the wording no longer claims the bytes are still
there one statement after they are deleted (F12, 2026-09-16).

That total is honest across an inherited store too (F12, 2026-09-16): `_store_blob` used to
return early for a blob already on disk — this run's own earlier copy, or one a killed run's store
left behind — without adding its size to the running total, so a run that reused a killed run's
store reported `0.1 MB of 64 MB` for a directory that measured 47 MB and the cap went
unenforced for the rest of that run; each digest is now counted once, on whichever call finds it
first.

Adapters: `claude-code` `--permission-mode acceptEdits`, `codex`
`--sandbox workspace-write`, `opencode` `--agent build`, each with a non-argv
(`stdin`) transport and never a `--yolo`-class bypass;
`tests/test_providers.py` cross-checks `agent-write` against `supports_file_edits` in both
directions. `utils/write_phase_bridge.py` is the one conversion from each phase's existing batch payload
into materialized write tasks (stable `prior_changes_id`, declared file set, monotonic result
names).

**`prior_changes_id` has to be unique per batch as well as stable**, and `utils/prior_changes.entry_id`
is the one owner of both halves: every reader sees one entry per `(id, phase)`, the newest
(`latest_entries`; `append` skips an identical summary and appends a changed
one, which supersedes the earlier line — fix round 14: a check re-run to a pass kept its findings
line, and the task after it refused), so a *colliding* id silently DROPS an entry while an
*unstable* one silently DOUBLES it.

The three batched phases keyed it on the bare `file_key`/`section_key`, which
is a partition and not a batch — the batchers emit many batches per key (`MAX_FIXES_PER_BATCH`, the
char caps, one batch per security/HIGH fix, a section that overflows) — so only the first batch on
each key was ever recorded and every later one read a context missing its predecessors, at exit 0.
The id is now that key plus a short digest of the batch's own member ids (the same list the task
carries as `expected_ids`), computed **once in Python** and serialized into each batcher's
`to_dict()`, so the native invoker reads `batch.prior_changes_id` off the batch instead of
picking a key and both paths append under byte-identical ids.

`implement` is unaffected: it keys on `task_id`, which is already one per
batch. **How an apply phase ENDS is the base's job, not the scheduler's** (`utils/apply_orchestrator_base._write_outputs`).

Before the first batch it snapshots the single file it is about to overwrite as
`{phase}/backup.md` — `write_target_file()`, gated on `has_single_write_target`, so
apply-code-fixes writes none (no one target) and apply-task-suggestions answers with the
`tasks-backup.md` it already publishes — and an existing backup is KEPT only when the phase has
RECORDED PROGRESS (a processed item, `write_batch_report.json`, one of this write kind's own batch
results, or an explicit `--resume`), because that is what makes the target half-applied;
keying on the backup file's existence instead treated a *new review cycle* — same phase dir, no
progress in it — as a resume, so cycle 2 kept cycle 1's snapshot, printed "Backup kept", and saved
the bytes it was about to overwrite nowhere.

The same mistake survived one level down until 2026-10-09: the phase dir and `state.json`
outlive a review re-run, so cycle 1's records counted as cycle 2's progress. `_phase_has_recorded_progress`
now dates each signal against the review's latest completion (`_review_completed_at`), strictly, as
`_earlier_report_rows` does — a processed record by its `timestamp`, the report by its
`written_at` (else its mtime), a batch result by its mtime — and an older one is an
earlier cycle's; a missing date on either side keeps the signal, and `--resume` still
counts on its own (mid-cycle it is the native leg's only signal).

Any re-stamp over a backup whose bytes differ keeps the earlier one aside first
(`keep_backup_aside`, `backup.<sha8>.md`), so the plan from before cycle 1 survives cycle 2
(`tests/test_apply_backup_across_rounds.py`). `--fresh` re-stamps for the same reason, and it is the one
apply invocation whose scheduler does not resume (`run_write_executor`: every batch dispatched
again, no result file trusted as done, no earlier row carried, no snapshot reused — implement's run
without `--resume`; `--resume` beside it still resumes, and a plain re-run
still resumes so a landed batch is never applied twice).

It presumes the target is back at its pre-run bytes and checks: a target whose sha256 is a
`sha256_after` some completion of this review cycle recorded with moved bytes still carries
that run's edits, so `_fresh_target_refusal` is exit 3 at `_setup` step 10 — before
anything is cleared — naming `backup.md` and `util restore-batch` (measured 2026-10-02:
a completed run then `--fresh` landed every batch twice at exit 0 and re-stamped
`backup.md` and every `pre_batch/` slot with already-applied bytes, so nothing
held the pre-run file); and what it re-stamps never destroys a rollback point: a differing earlier
`backup.md`/`tasks-backup.md` moves to `<stem>.<sha8>.md`
(`keep_backup_aside`) and an occupied `pre_batch/<id>` slot is set aside before
`take` whenever the scheduler is not resuming.

After the loop, on the executor leg only, it calls `record_artifacts` + `detach.record_output_files`
with the file it edited, `orchestrator_output.json`, `write_batch_report.json` and `prior_changes.jsonl`
(a fully successful detached run reported `artifacts: []` and no `output_files`,
measured 2026-09-16), and replaces `print_text_summary` — a *plan of work*, printed in the future
tense 0.1 s after the work finished, 12.5 KB of it — with an outcome summary: batches ok/total,
items applied (the `expected_ids` of the `ok` outcomes), per-batch
status/verdict/duration/runner from `SchedulerResult.outcomes`, and every skipped item with its status,
`group_id` and reason (the skip's own when it was skipped, not the needs-human
validation text it superseded).

That block goes to **stderr** when `--output-format json` was requested: the machine payload is then
the whole of stdout, which is what the flag promises. F7 (2026-09-16) extended the hand-back for a
phase with no single write target: every `ok` outcome's own `files_touched`
also joins it as `touched_file_{n}` (resolved against `write_safety.resolve_repo_root`, the same way
`implement` resolves its own) plus the judge's kind-named report and the
decisions-overlay report path when either exists, and the RESULT block / `summary.md`
print that touched set as `Files changed:` instead of falling back to `self.plan_path` —
a phase like apply-code-fixes that never opens the plan had been reporting the plan as what it
changed.

That last part is also the only place the decision summary reaches this leg: `util decisions`
is step 3 of the apply instructions and `--exec-subagents` finishes the phase in one invocation,
so the step never runs. `skipped_items` carries `group_id` now for the same
reason `to_apply` always did — a skip that can only be matched back to the report by
title is not traceable.

**And the skip list is rebuilt after judging, not before it** (`_rebuild_formatted_skipped`, called from
`_apply_user_feedback` and again from `run()` past `_handle_human_decisions`): built
once, the list and the `skipped_count` published beside it disagreed by exactly the judged
items — measured 2026-09-16, `skipped_count: 3` over two `skipped_items`,
`Skipped: 2` in the RESULT block, and `util decisions` listing 19 of 20 groups, so
the only decision the run actually *made* was missing from every surface meant to report decisions.
A judge-sourced skip carries `status: "judge_skipped"` plus its `decision_source` and the judge's
own reason, and `display_decisions.py` buckets it apart from `invalid`.

The judge's persisted names (`judge_decide`, `judge_auto_decide[_salvage]`, `judge_skipped`,
`judge_decide_item_ids`, `judgeDecided`) live in `utils/judge_vocab.py`, which is also the
one map from the `claude_*` spellings written before 2026-10-02: writers use only the
new names, and every load boundary (`StateManager`, the selections loaders,
`util decisions`, the `report_data.json` sidecar) reads both.

`StateManager` also writes an upgrade that changed something straight back to
`state.json` (the upgraded document alone, skipped if another process saved since its
read), because the host reads that file raw and a native-leg `--resume` saves nothing
before `--mark-completed`; the reporting reads (`status`, `util next-command`,
`--status`, the tasks staleness probe) load it `read_only=True`, which never
writes and makes `save()` raise.

**Steps 7 and 8 of the apply instructions run on this leg too** (`write_apply_summary_report`,
`regenerate_decisions_overlay`, plus a `Decided by the judge agent` block in the RESULT output): the
`{phase}/summary.md` the instructions advertise and the decision badges on the review HTML lived
only on the native leg, which `--exec-subagents` never reaches, so the first run that judged
anything had no durable human-readable home for the decision. The overlay is called as a function,
not shelled out as `util report regenerate-decisions`, and is a quiet no-op when the review predates
`report_data.json`. The native leg keeps `print_text_summary` byte-for-byte: there it really
is a plan of work the host is about to execute.

F10 (2026-09-16) gave `write_apply_summary_report` the per-fix detail its own template promises — each
batch's own fix/suggestion titles and `file`/`line_range` from
`self.batches`, its runner's free-text `notes` read back from
`output_path`, and `Base ref`/batching-efficiency lines when the orchestrator
computed them — and F11 patches `write_batch_report.json`'s `summary_file` (serialized
`null` by the scheduler, which runs before this leg's own summary exists) once the
summary path is known.

Both use `item_noun_plural` (F8: defaults to the arithmetic plural in `__init_subclass__`,
overridden to `"fixes"` on `ApplyCodeFixesOrchestrator` since `"fix"+"s"` read
"Fixs" on every one of these surfaces) rather than pluralizing `item_noun` inline.

**`TaskOutcome` carries its own verification evidence on write outcomes**
(`expected_ids`, `returned_ids`, `sha256_before`/`sha256_after` per
file, F9, 2026-09-16), populated once in `_classify` right after the post-run
`observed` hashes are computed so every branch below it shares the same row —
`write_batch_report.json` (`SchedulerResult.to_dict`) and this run's own `subagent_batch_report.json`
(`BatchReport`, built from the identical `TaskOutcome` list) therefore both answer
"did every item come back?" and "is the hash chain contiguous?" without a reader re-deriving it from
`subagent_audit.jsonl` or the per-batch result file.

**A stop accounts for every task and names a way forward that works** (racing re-run 3, 2026-10-05):
`_record_stop` gives every materialized task the stop left undispatched a
`not run — …` reason on its `pending` status and lists it in
`SchedulerResult.not_run` (→ `write_batch_report.json`'s `not_run_tasks`/`stop`,
the mirror's `not_run_task_ids`, `summary.md`'s accounting line and
`## Cancelled` list, `status`'s `not run T012, T013`), and
`_stop_message` names the failed task, what a check reported (`TaskOutcome.runner_notes`, the
runner's `notes`), whether anything is half-applied (only when a failed batch moved
bytes, `stop.bytes_moved` — fix round 14; `status`'s resume line reads the mirror
rows' `files_touched` the same way), the remedy in the host's own command spelling
(`user_commands`, built by `implement_orchestrator._stop_user_commands` through `next_phase.phase_command`) and the
accept path; the one-line `stop["headline"]` is recorded as the result line's
`error` (`cli_common.record_error`; the key is gated on the line's
`state` word, `error`/`needs-human`, not the exit code, so a
`status --exit-zero` poll keeps it — fix round 8), which the detached close-out writes as
`run.error` and a pinned terminal `status` poll re-records, since both
package servers relay `payload.error` as "the last line it left".

`--human-task <id>=done|skip|retry` answers a code task the state records `failed`, on
`--resume` only (`done` → `HumanTaskPlan.accepted`, not dispatched,
dependents run; `skip` → the skip walk, under every strategy —
`skip-continue` satisfies only a manual task's edges; `retry` → one-shot,
never persisted); the native emit applies the same answers (fix round 8: it used to resolve the plan
only for a plan holding a manual task, and handed an accepted task out again as
`failed`) — an accepted task leaves the batches and is named in the handoff's
`accepted_tasks`, a skipped code task is materialized on `--resume` like a human
task so the resumed account keeps it (one the resolved plan no longer skips — `--resume --force-strategy skip-continue`
after `skip-dependents` — is handed out and recorded `pending` on both legs,
`mark_no_longer_skipped`, fix round 9), and `_applicable_task_decisions` keeps a `skip`
while the state says `skipped`; a human task answered `done` is the
`completed manual` bucket (an accepted failure and a manual task answered `done`
that a later `--resume` no longer materializes are read back from their recorded
reasons, `answered_done_on_record`, on both legs — the native handoff's `accepted_tasks` and
`completed_manual_tasks` — fix round 9), and a skip the user chose (`USER_DECLINED_REASON`,
`ANSWERED_SKIP_PREFIX`, their chain — `HumanTaskPlan.declined()`, twinned in `StateManager._deferred_to_user`) is
`skipped at your request` and `## Skipped at your request`, never a deferral (summary.md's
`## Task Results` gives a task this run's answers settled one row, the answer as its outcome
and the earlier run's outcome inside it, fix round 26); and `_unchanged_since_failure` refuses to
re-dispatch a failed verification-only task while `git_utils.worktree_fingerprint` (tracked +
untracked-not-ignored files by size and mtime, the plugin's own trees excluded; stamped on the
failed verdict as `tree_fingerprint`) still matches, producing a `not_rerun` outcome
with no metric and no new prior-changes line — the executor leg keeps stop-on-failure (continuing
past a failed check with its independent siblings was judged unsafe to add under the single-stop
resume contract; the native leg's instructions continue with independent tasks, a documented
difference).

## Retry after edits

**A write batch is never sent twice on top of its own edits** (fix round 8): before any retry or
rate-limit backoff would re-send a write batch's own prompt (a timeout, a runner error, a prose
reply, a 429), `dispatch_task` asks the scheduler's `changed_files` hook whether the
attempt moved bytes (T4.8, which asked only for a denied turn). One that did is a partial
application (loop stops, snapshot kept, exit 3, `util restore-batch` named), except a write kind's
prose reply (implement's too since fix round 14), which gets the rule-7 report re-ask instead; the
ladder's side is rule 8, `RetryLadder.record(..., repeatable=False)`.

Its converse is rule 9 (fix round 14): an `applied` reply that moved nothing is sent
once more, told so (`record_resend`); for a plan or task-list batch "nothing" means its
declared file, a stray write beside an unchanged plan included, and a code batch whose declared
files did not move is applied on its own stray writes only, never on a path a co-scheduled sibling's
window also saw move (`UndeclaredWrites.shared`, fix round 15). A stop's `bytes_moved` and
`status`'s resume cost read what the sweep observed, a moved file with no rollback
point (`TaskOutcome.unrestorable`, named on the stop) included (fix round 15).

A failed batch writes the native leg's `(task failed — …)` / `(batch failed — …, verified no file changes)` prior-changes
line (fix round 15; none for an apply batch that moved bytes, a task refused before dispatch or a
check not run again), which the resumed dispatch's real entry supersedes, since
`append` keeps the newest summary per `(id, phase)` (fix round 14). A
dispatch that persisted no result file gets no verdict stamp (`_stamp_verdict` returns
quietly).

## Declined items and partial applications

**An apply batch may decline an item, and a partial application its resume would only repeat has a
way out** (fix round 18, Codex racing run 7's 08/08b: a batch left a fix out on purpose, said so
only in `notes`, and stopped on every resume). The three apply kinds' result contract
carries `declined: [{id, reason}]` beside `ids` (`with_result_contract`, the report
re-ask's shape, `write_batch_result.schema.json`; `implement` has none — a task its runner will not
do is `failed`, and a reply declining its task is read so).

`subagent_executor.declined_items` reads it leniently (a reason is recorded as the batch wrote it, closing stop
included, fix round 26; a line that ends the sentence after it adds a full stop only where the
reason ends in no `.`, `?`, `!` or
`…` of its own, `text_clip.with_closing_mark`, fix round 27): an expected id declined
with a reason is neither applied nor missing, a batch whose every item is applied or declined is
complete (a no-op, never re-sent by rule 9, when it moved nothing, whatever its
`status` word — `failed` too, which a resume keeps,
`_declined_from_a_failed_claim`, fix round 19), and an id in both lists, an id the batch does not hold or an
entry without a reason is a missing id, i.e. a partial application with the problem named.

`TaskOutcome.declined` rides the row, the verdict stamp, a resume skip, the batch log line, the
mirror's `declined_items` (`status` names them; `items_applied` excludes
them), summary.md's own `## Declined by their batch (not applied)` / `## Skipped at your request (not applied)` sections (fix round 22:
never under an applied heading, on both legs — the executor's summary, the runbooks' template, the
`[BATCH_DECLINED]`/`[BATCH_SKIPPED]` lines) and counts, and `ApplyOrchestratorBase.run_notes` (the
notice).

On the executor leg a readable report that, after edits that moved bytes, leaves items out of both
lists first gets the one report re-ask (`incomplete_report` hook, `RetryLadder.record_report_reask`,
`REPAIR_INCOMPLETE_REPORT`; the attempt that drew it is recorded `incomplete_report`, with no failure
capture, fix round 19; a re-ask that cannot restate leaves the first report to decide, one that
moves a file is a partial application); a partial application whose readable report still attests at
least one item in `ids` (fix round 19) and leaves others out
(`unapplied_ids`) offers `--resume --human-task <batch>=skip` (way-forward lines, headline,
`stop.answers` → the tool routes' options), which the scheduler checks against the stopped
verdict, the items, the inputs and the bytes on disk (`_check_partial_skips`, else exit 3 with
nothing dispatched, recorded as an `answer-refused` stop whose headline is the result line's
`error` and whose mirror drops the refused answer, fix round 19) and then keeps the
batch as it stands (`_keep_at_request`: re-stamped verdict, audit completion line, prior-changes
append, the left-out items as `skipped_items`, never counted as applied); an apply phase
takes only `=skip`, only with `--resume` and `--exec-subagents`.

On the native leg `util batch-done --declined <id>=<reason>` / `--skipped <id>` (refused when no item of the batch
is left applied, fix round 19) record those items `declined` / `skipped` in
`processed_<phase>`, print `[BATCH_DECLINED]` / `[BATCH_SKIPPED]`, and the apply
runbooks give the user both ways forward; `status`'s native apply view names them in
the executor's words (`run_status.not_applied_item_clauses`, fix round 22; `1 declined by its batch (not applied): …`, after the
`items applied` count, fix round 26).

Since fix round 22 the scheduler writes the same `processed_<phase>` record per batch
(`state_manager.record_batch_items_in_state`, both legs:
`applied`/`no-op`/`declined`/`skipped`/`failed`,
with `task_id`, `prior_changes_id`, `grouped_sha256`, `leg`;
the native leg records a verified no-op batch as `applied`, `util batch-done`
having no flag for it, and no reader tells the two apart, fix round 23), and step 23 reads it
through `processed_record_counts`: on the executor leg its own records drop nothing (its resume keeps a
batch from the result file under its own id, so the ids a stop's `=skip` names do not
move), and an earlier review's declined or skipped record drops nothing on either leg (offered
again; an applied record still counts).

## Dispatch counts

**Implement's counts are what a run dispatches, on both legs** (fix round 18, run 7 #2/#3/#9).

The executor's `Organized into …` line is the scheduler's own (`WriteBatchScheduler._dispatch_preview`, printed
once the human-task plan is resolved, in the native emit's words, `dispatch_line`): the
loop's own read-only tests with its log muted — resume evidence and `_check_findings_rerun` (a check
after a task this run dispatched counts as run again), `_unchanged_since_failure` and the write policy,
either of which stops the run so later tasks are not reached (the tasks dispatched in the same group
beside it still count: the walk runs over `admit_disjoint_groups` under `write_ceiling()`, fix
round 19) — and a task after a check this run dispatches counts, named as waiting on it; the
orchestrator prints it only for `--dry-run`.

The `[ETA]` line counts tasks, one record per dispatched task: on the native leg
`util batch-done` takes out of `metrics_progress.implement.total_batches` (`withdrawn`, put back when
that task is recorded again unless a check with findings or a failed task upstream still keeps it
out, `_still_kept_out`, fix round 19) each handed-out task its record keeps from running — held
by a check's findings, after a failed task, refused at `util write-prompt` — and
`util metrics record` counts against that stored total whatever the host passes
(`_settle_eta_total`, `metrics.withdraw_from_progress_in_state`); the executor's total (`_eta_total`)
leaves out accepted, manual and held tasks and, since fix round 20 (run 8 #1: "5 task(s) to
dispatch" beside `[ETA]` 1/7 … 4/6), starts from the dispatch walk's own set
(`_dispatch_walk` → `_eta_scope`: a task refused before dispatch, the tasks after it
and a resume's kept tasks are never in it; a task that finishes without a runner leaves it and one
dispatched against the walk joins it, `_eta_mark`; a stop short of the total logs
`[ETA] implement: run stopped after task N/M · <what it left>` (fix round 22, live run 9 #1: implement's lines say `task`,
the N/N line of a run about to stop claims no completion, `_eta_ends_short`, and both legs name
each task left, `metrics.left_out_tail` over `_eta_left_out` / the native emit's
`metrics_left_out` record and `withdrawn`; a run that dispatched nothing closes on
`run stopped with nothing dispatched · …`, on the native leg through `util metrics finish` with or without a
`start`, and a native task recorded `failed` ends on
`run stopped` too, `metrics_progress.{phase}.failed`, written by `util batch-done` for every
write kind since fix round 23, whose apply batch refused before dispatch leaves the count on both
legs, `_eta_unrun`; an apply phase's closing words count `batch(es)`); since
fix round 21 every ending of a write phase logs one closing line, `_eta_close` — a stop at
N/N too, a refusal, the tasks a check holds (`· T004, T005 not run: check T003 reported problems`), a failed
`driver.verify_command`, an exception out of the loop, a foreground Ctrl-C, a detached worker's abort
(`metrics.aborted_eta_line`, printed by the signal handler) — beside a finished phase's
`phase complete`, and re-stamps `metrics_progress.{phase}.total_batches`; a check's held tasks leave the total
from its own line on, `_eta_held`, and on the native leg `util metrics finish` after
records that kept tasks out says `run stopped after task N/M in … · <what it left>`), and the write mirror's
`total`/`items_total` (`status`'s progress line, the OpenCode
poll) count the same tasks (`_eta_task_ids`, fix round 19: a held T007 read "6/6 … ALL
BATCHES COMPLETE" beside "6/7 batches ok"); the line still names a held task, with its check.

The write mirror names a resume's kept tasks (`kept_task_ids`: since fix round 22 every task
the `resume: kept` line names, `_resume_kept_rows`, where Codex 01b's
`status` said "1 kept" beside "kept 4"; since fix round 23 the ones kept on a
`=done` answer too, `_resume_answered_ids`, a manual one also in
`completed_by_hand_task_ids`), rendered `(<n> kept from an earlier run: <ids>)` on the progress line
(`<kind>: nothing dispatched (…)` when the run dispatched nothing, and implement's `<ok>/<total> tasks ok`; a
run record that ended with exit 3 heads its block `stopped and needs you (exit 3)`, `run_status._stopped_for_the_user`,
glossing `nothing crashed` only while no runner or model of it failed (`runner_failed_task_ids`,
`_a_dispatch_failed`, fix round 23), its `run.state:` line reading
`stopped (exit 3: …)` while the record keeps `failed` (fix round 26), and a stopped
write mirror's `stopped` step is terminal, heading a foreground run
`stopped in foreground (needs you; …)` since fix round 23), each with what it is in the `resume: kept`
lines' words on both legs (`kept_named`: a check whose findings stand reads
`KEPT_WITH_FINDINGS_LABEL`, and an implement run's accepted tasks fold in as `accepted with its check findings` /
`accepted after failing`, the `status` tally's words too — fix round 20, run 8 #6), and
its `ok` counts the same tasks as its `total`.

A first implement starts nothing over (`_has_earlier_implement_records`: no record, so no reset stamp and no
started mark), and completing or skipping a phase drops `phases_started` once it is empty.

## Prior-batch context

**An apply-code-fixes batch's prior-batch context is its own run's batches** (fix round 22, live run
9 #7: implement entries past the record's budget listed as "Batch 15-18", and a first batch's
fall-back to the whole log listed the planning phases' plan-edit batches).

The `## Implementation record` (`code_fix_batcher`) ends with `- and N more tasks: <ids>` for the
implement entries its budget left out; `bind_prior_changes` renders a phase in
`OWN_RUN_CONTEXT_PHASES` (`apply-fixes`) with `prior_changes.render(own_run_only=True, since=last_phase_end(...))`: only that phase's
entries recorded after its last completion or skip stamp in `state.json`
(`prior_changes.run_entries`; fix round 23: a completion does not count while `processed_<phase>`
holds a `failed` record of the batch's review, a phase closed over a failed batch, and
the run is then bounded by the skip stamp and another review's newest record), no other phase's or
their count, no fall-back; `util write-prompt`'s context count and `[WRITE_PROMPT_ORDER]` read
the same entries.

The raw prompt (and so `_task_fingerprint`) moves only when the record leaves an entry out. Fix
round 24 (live run 10 #6: "Batch 1/2/3" beside `apply-fix-batch-0/1/2`, "- (T002): T002: …"): every
list of earlier batches names each by the id of the batch that wrote it (`prior_changes.entry_line`,
`- **apply-fix-batch-0** (index.html#a4d7fa43): …`, the entry id alone when they match), never a list number; the id is the
entry's `task_id`, recorded by both legs' `append` (the scheduler passes
its own, `util prior-changes append` reads it off the phase's `write_tasks.json`), and no line prints
an id twice (`summary_without_ids`, the record's `record_line` too, which
`quoted_in` matches).

Fix round 25: each is one plain line, its entry id through `prior_changes.plain_entry_id`
(`?` when it is not a plain id) and its summary through `plain_line`,
in the record and its left-out ids too.

## Write reply extraction

**A write reply is read as its report object, and the records end a quoted sentence once** (fix
round 28, live run 12 #4/#6/#7/#9). `json_extractor.extract_json_from_text` adds `only_object`, the reply's
one JSON object outside every other document, whenever it serves something else, and both invokers
read a kind whose result is an object (`subagent_executor.OBJECT_RESULT_KINDS`: the write kinds,
`task_generation`, `consolidation`, the judge) through `the_reply_object`: Codex's
decline batch opened with prose and the array-preferring extraction served its report's
`declined` array ("<root>: expected type object, got list", a re-ask); a reply holding
two such objects names none and keeps its re-ask.

`status`'s native apply view counts its items before the clauses for the ones its
batches declined or skipped, in the executor's words (`run_status.native_items_count`:
`2/3 items applied[ to <file>]; 1 declined by its batch (not applied): …`, over the run's items and records (fix round 29; see [native apply counts](host-reporting.md#native-apply-counts-across-resumes)),
once one names an item; `--json`'s
`items_applied`/`items_total`/`applied_to`). `status` prints
each why on the line under its command, unbracketed (`run_status.why_line`; a why ending in a
bracket read `(T006))`), the `=done` answer's why gives a check's
findings after `; <check> reported:` without their closing stop, and the CHECK FINDINGS reason
joins findings with `write_batch_scheduler.findings_clause` and ends the sentence once.

`util prior-changes append` refuses an empty or blank `--id` (exit 3, nothing written,
naming the batches of the phase's `write_tasks.json`).
