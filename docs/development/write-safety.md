# Write boundaries and tracking

Read when changing write preflight, containment, backups, rollback, verification, or tracked-file scope.

Back to the [development guide](../../skills/multi-llm/AGENTS.md). These notes retain detailed contracts and regression history from that guide. Paths beginning with `docs/`, `packaging/`, or `plans/` are repository-relative; other implementation paths are relative to `skills/multi-llm/`. Dated incidents explain the rules; verify version-specific claims against current code.

## Contents

- [Write safety controls](#write-safety-controls)
- [Backups](#backups)
- [Code-phase tracking](#code-phase-tracking)

## Write safety controls

**The write path is auditable and reversible before it is trusted (D3l), and all of it ships with
the first edit-capable dispatch.** `utils/write_safety.py` owns five controls, all on by default: a
**pinned repo root** (the directory holding the `.git` entry above the anchor,
walked and `realpath`'d — `git_utils.pin_repo_root`; `git rev-parse --show-toplevel` must agree with
it and a disagreement, a `GIT_DIR`/`GIT_WORK_TREE` in the environment, or a
`core.worktree` git cannot confirm is exit 3 naming the cause, since a repo-wide batch can
plant `[core] worktree = ~` in `.git/config` and every later phase used to re-pin the root
from git's answer, where the next jail's one rule then landed — review fixes 2026-10-05;
`landlock_dispatch.repo_wide_root` re-checks at the grant and an explicit scheduler `repo_root=` is
checked the same way; the post-run artifact hand-back degrades to a warning through
`repo_root_for_report` — and `realpath` on **both** sides of every containment
comparison, per D1c; asymmetry is the bug); `driver.write_paths`
`allow`/`deny` globs enforced **twice** — pre-dispatch against the
batch's declared set (refused before the runner starts) and post-run against the set the diff
observed (batch failed, offending files restored from the snapshot, loop stops), with post-run
honestly labelled *detective* while the runner's own sandbox stays preventive; the mandatory
pre-batch snapshot, under the **phase** dir so concurrent or resumed batches cannot clobber each
other's rollback point — except for a `repo-wide` batch on the plugin's Landlock jail,
whose root grant makes the phase dir the batch's to write: its snapshots, rollback store and audit
log live under `$XDG_STATE_HOME/multi-llm/write-safety/<sha256 of the phase dir>/` (`write_safety.claim_external_safety_dir`, claimed by the scheduler before the
phase's first snapshot once the preflight recorded the rung; every reader resolves through
`write_safety.safety_dir`, which never falls back to the phase dir once that directory exists; the jail
restricts writes only, so a key the batch could not read does not exist and an HMAC would be
forgeable — review fixes 2026-10-05; every other runner keeps the phase dir); a dispatch-time
**pairwise-disjoint admission check** (`driver.max_concurrent_write_subagents`, default **1**; shared-artifact phases
pinned to 1 regardless; a task declaring no file set runs alone) checked executor-side rather than
trusted from the batcher; an `implement` task with **no declared files** is a
verification-only task (generate-tasks emits one for a syntax gate or smoke run, 2026-10-01: the D3h
refusal dead-ended implement at batch 12 and `--resume` hit it again): it is snapshotted
as an empty set, the undeclared-write sweep is its whole verdict (`_verification_verdict`: nothing
moved is `no-op`/`VERIFIED`, any path moved is a partial application
naming the fix — declare the files under `Files to modify` and `--resume`; its
rows, verdict stamp and mirror row add `verification_only` + `outcome_label`
(`verified` / `partial-application`, `subagent_executor.write_outcome_label`) and summary.md, the batch
log line and `status` show the label, while `write_outcome` stays
`no-op` for resume and older readers, 2026-10-05), and the apply kinds are still
refused with nothing to snapshot, and so is a verification-only task whose pre-run census is
unusable (the tree over `SWEEP_MAX_FILES`), because a sweep that cannot look must never read as
VERIFIED (adversarial review, 2026-10-02: a tree that outgrows the cap *during* the run is a failed
batch for the same reason, the sweep compares bytes and not stamps, so a same-bytes rewrite is not a
write, and under `--resume` the reused snapshot's adopted entries keep an earlier
dispatch's stray edit PARTIAL); the sweep leaves **gitignored paths** out (`git_utils.ignored_paths`,
2026-10-02: the host's own `logs/*.jsonl` reached `tracked_files` and review-code's
scope) except a path `driver.write_paths` **denies** (the deny side alone, `WritePolicy.denied`,
so a narrow `allow` does not re-admit them), and only while the ignore rules in force
at dispatch (`.gitignore`s, `.git/info/exclude`, `.git/config`, the global
excludes file, hashed by `_ignore_rule_sources`) are unchanged after the run — a runner can write
those, and a drop decided on post-run rules hid everything a runner-written `*`
covered (adversarial review, 2026-10-02); `finalize-tracking` drops ignored
`--touched` paths too; a `--resume`'s `write_batch_report.json` and
`summary.md` carry the earlier runs' task rows (`earlier_rows`, flagged
`from_earlier_run`) beside this run's — only rows of the current review cycle: a report whose
`written_at` (file mtime for an older report) predates the apply phase's review completion
belongs to the previous cycle and is not carried (`REVIEW_PHASE_BY_WRITE_KIND`, the `_apply_batches_predate_review`
rule; adversarial review, 2026-10-02) — an earlier row is the same batch as this run's only when
both name the same `expected_ids` (`_same_batch_row`: apply ids are positional, so a
resume whose selection changed hands `apply-batch-1` to other items), and the apply phases'
own `summary.md` and result block render the same `report_rows()`
(`ApplyOrchestratorBase._phase_rows`), marking `(earlier run)` and a reused resume skip apart, with counts
taken from those rows; and `{phase}/subagent_audit.jsonl`, one fsynced append-only line before dispatch and
one after classification, carrying the rendered argv with the prompt reduced to its sha256 + byte
length — **never** prompt bytes, the runner's environment or any auth value.

`--dry-run-subagents` runs the whole preflight and starts **no** child (`{phase}/subagent_dry_run.json`).

`multi_llm.py util restore-batch --phase-dir <dir> --task-id <id>` (also `--all-failed`) refuses a manifest whose `root`
is not the phase dir's own pinned repository (`TaskSnapshot.load(expected_root=…)`, the scheduler's resume reuse
too), writes a payload only while its bytes still hash to the manifest's `sha256_before` (and
`adopt` keeps a rollback blob only while it hashes to its digest), and restores
exactly the files the audit recorded as touched and **refuses** one whose hash moved after the run
unless `--force` — the record is found by the items the snapshot manifest names
(`completed` lines carry `expected_ids` since 2026-10-02; a set-aside
`<id>.items-<sha8>` snapshot's record sits under an id another batch now owns, and
`--all-failed` lists the stopped batch under that set-aside name), and a snapshot with no
completion record is UNVERIFIED and restored only under `--force` (it used to restore
everything at exit 0, wiping a later batch's edit and the user's own; adversarial review,
2026-10-02); a partial application never reverts itself — the exit-3 message names the command,
because an automatic revert would destroy the state a human needs to inspect.

**The native leg's per-batch snapshot is not a `pre_batch/` slot.** For apply-suggestions and
apply-task-suggestions, `util write-prompt` copies the batch's one artifact (the plan, the tasks
file) byte for byte into the workspace temp dir (`write_prompt.prepare_batch_snapshot`,
2026-11-08; the host used to copy it in a turn of its own), and the host's byte diff against it
decides the batch. It is a convenience copy: it lies in the repository, which the batch's subagent
can write, so its checksums catch an accidental change and prove nothing against a subagent. No
command restores from it; the apply instructions tell a human how to copy it back by hand, and
that code-fix batches have none. Trustworthy native rollback needs the executor's protected
storage, a larger design (plans/apply-phase-throughput.md; Codex review 2 of U31, #2). The rules:

- *The phase dir is pinned first.* A phase dir that is a link (or not a directory) is refused
  (exit 1) before anything is read through it or written — no commit re-check record, check-tree
  record, commit bind or snapshot — in every output mode, and the snapshot's root is the pinned
  phase dir's repository, taken from the pin (`git_utils.pin_repo_root` must agree with it), never
  found again through the phase path. `write_tasks.json` is read for the run stamp through the
  pin too, so a folder swapped after the pin is refused as a link (Codex review of U31, #1).
- *One slot per phase dir and artifact.* The slot is
  `.multi-llm/tmp/batch_snapshot_<plan|tasks>_<plan stem>-<key>`, the key 12 hex digits of the
  sha256 of the pinned phase dir and the artifact's absolute path, so two plans of one name in
  different folders never share one (review 2, #8). The host takes the paths from the
  `[WRITE_SNAPSHOT]` / `[WRITE_ARTIFACT]` markers; binding the same batch again prints them again.
- *A generation is published whole.* A generation is a copy `<slot>.<sha12>.md` and the binding
  `<slot>.binding.json` naming it (`task_id`, `expected_ids`, the phase dir and its
  `(st_dev, st_ino)`, the run — the sha256 of `write_tasks.json`, which only an emit writes — the
  artifact, the copy's name and sha256, a random `generation` id, the time it was taken). Every
  field is checked for presence and shape before a transition is chosen: a binding missing one
  (its `phase_identity`, say) is damaged evidence, never another run's (review 3, #1). The copy
  is written first, the
  binding renamed into place last as the commit point, and the previous generation's copy is
  removed only after; a failure at either write leaves the previous generation whole (review 2,
  #3). Both are written after the prompt file and before the markers hand the batch out, never
  through a link (`prompt_files.PromptFolder`).
- *One call at a time.* Every call holds `<slot>.lock` (opened without following a link,
  `process_control.wait_for_fd_lock`, 10 s) from reading the slot through publishing it, so two
  hosts binding at once cannot take one batch's edits as its baseline; a call that waits too long
  is refused (exit 3) with nothing written, and tells the host to run it again (review 2, #4). A
  lock the file system cannot take at all (`EOPNOTSUPP`, `ENOLCK`: any errno outside
  `process_control.LOCK_CONTENTION_ERRNOS`) is raised at once, never waited out, and refused
  (exit 3) naming the cause and saying a retry gives the same answer (review 3, #3).
- *Transitions.* The same batch bound again keeps its generation (refused, exit 3, when its copy
  no longer hashes to the binding: a new copy could carry the batch's own edits). Another batch of
  the run replaces it only once the bound batch has a `util batch-done` record made for this
  generation — a recorded `task_status` and, beside it, the binding's `generation` as its
  `snapshot_generation`: `task_status` is one map for every run, and a `--resume` numbers its
  batches afresh. A new emit of the phase (a `--resume`, a re-run) is a new run and replaces it.
- *Identities, not clocks.* `util batch-done` keeps `write_prompt.completion_stamp` for a
  shared-artifact batch in `state.json`'s `batch_snapshot_records[<phase>][<task id>]`, in the
  same locked write as its `task_status`: `run` (the sha256 of `write_tasks.json` as it records)
  and, when the slot's binding names that batch, `snapshot_generation`. The `processed_<phase>`
  records keep the one shape both legs write. Records used to be dated against the binding's and
  the emit's local wall-clock stamps, so a clock set back between binding and recording hid a
  real record (review 3, #2).
- *Damaged evidence is not a new slot.* A binding that cannot be read, is not JSON or not this
  version's, or names another phase dir or file than its slot's; copies no binding names (a
  deleted binding, a cut-short write); a `state.json` that is missing, unreadable or has no
  `task_status` object where a transition needs it — each refuses (exit 3), naming the files and
  the human's way out: compare the artifact with the copies, put it back by hand if a batch
  half-applied, delete the slot's files and run again (review 2, #5).
- *In order.* Before the slot is touched, a batch is refused (exit 3) while an earlier batch of
  its run (lower `batch_index`, not a human task) has no record from this run — one whose
  stamp names this run. The batches edit one file one at a
  time, and without this a batch bound early and the batch before it each waited on the other
  (review 2, #6). The `[WRITE_PROMPT_ORDER]` warning stays advisory for the other kinds.
- Outside a repository, or with an artifact it cannot read, the call refuses (exit 3).
  apply-code-fixes and implement take no native snapshot: their instructions send a human to undo
  a native batch by hand.

`driver.max_concurrent_write_subagents` and `driver.write_paths` are `driver.*` keys like any other:
closed set, fail-fast validation, typed accessors (`get_max_concurrent_write_subagents`, `get_write_paths`,
which fills each half of the policy per key so setting only `deny` is never an
accidental deny-all). The write ceiling (`WriteBatchScheduler.write_ceiling`, the minimum of that key, the
runner providers' `max_concurrent` and the host's ceiling, pinned to 1 for a shared-artifact phase) is
resolved once per scheduler over every task's runner, from one config snapshot (`_ceiling_legs`, beside
`host_max`). That number is an upper bound only. Each batch loads one config snapshot when it starts
(`_batch_runners_and_ceiling`), resolves its own tasks' runners from it once (`_resolve_batch_runners`) and
is admitted under the lowest of the phase's number, the driver key in that snapshot and the
`max_concurrent` it gives the providers those runners name (`_batch_ceiling`). Every config load asks git
for the project root, so a phase's ceiling costs one discovery per batch plus one for the phase, however
many tasks or providers a batch holds (`tests/test_write_ceiling_cache.py`). The
rollback-storage preflight and the dispatch use that same resolution (`_runner_for`). So a config edit, or
a CLI that went missing between batches, can lower a later batch's ceiling but never raise it, and a
batch's ceiling always binds the runners it actually runs. The batch report and the dry-run record give
the highest ceiling a batch was admitted under, with the tightest driver and provider values any read gave
and every provider a batch ran on. The declared-file admission under the ceiling still runs every batch.

The post-run sweep's "already declared, don't flag it" set is scoped to tasks still IN FLIGHT, not
the whole phase (F13, 2026-09-16): `_begin_sweep` used to union every dispatch's declared
set for the run's lifetime, so once any task declared a path a LATER task's undeclared write to that
same path stayed invisible for the rest of the phase — it only ever needed to cover a co-scheduled
sibling declaring its own files while THIS task is being classified, so
`_begin_sweep`/`_undeclared_writes` now add a task's declared set on dispatch and remove
it on classification instead.

## Backups

**Backup before modification**: Always backup files before any salvage or modification operation.

## Code-phase tracking

**`tracked_files` is the code phases' record, and the workflow's own files are never in it**
(2026-10-05, Codex racing run 3: apply-task-suggestions' census recorded the plan and
`tasks.md`, a partial implement recorded nothing, and review-code reviewed
`tasks.md` as code while the `index.html` implement created was out of scope).

Only `implement` and `apply-code-fixes` write it (`write_batch_scheduler.CODE_WRITE_KINDS`; the
planning phases' `_finalize` no longer calls `finalize_tracking`), each entry's
`task_id` names the phase that recorded it (`finalize_tracking.TRACKING_LABELS`:
`implementation`, `apply-code-fixes`; `util finalize-tracking --phase`), and `finalize_tracking.phase_tracked_files`
replaces only its own phase's entries — the other code phase's keep their label but have their
action read off the tree again (`_path_states`; a recorded file now on no disk and in no
base commit — created, then removed — is left out and said, fix round 8; one git's census names is
kept as `deleted` under the phase whose census found it, fix round 9), a path they hold
is not claimed again, any other label is replaced.

It is recorded as each code batch finishes (`WriteBatchScheduler._record_batch_tracking`, in memory through the run's own
`StateManager`; `util batch-done` on the native leg, census only), so a run that stops
part-way has recorded its finished batches, and a resume's `--touched` set carries the
earlier runs' `ok` report rows (`_touched_files`).

The plan, the task list its `TASKS_FILE` marker names (a Markdown file only) and what of
its output folder is the workflow's (`output_handler.workflow_file_entries`: the whole folder only when everything
in it is; beside code of the plan's name only `state.json`, the prior-changes log, the
phase folders and its own temp/lock files — fix round 8) are left out of the census, of
`--touched`, of `--only-touched` and of review-code's scope
(`resolve_review_scope`'s tracked branch and git-diff fallback; `--files` stays exact),
each drop printed once (`Left out N of the workflow's own file(s)`). Implement's first run (no `head_before_implement`
yet) re-takes the dirty-tree baseline an apply phase recorded at its own start, before it edited the
plan (`record_pre_existing_changes`).

**The diff base exists before the first commit.** `head_at_start` (a new `state.json`) and
`head_before_implement` are `git_utils.head_or_empty_tree()`: `HEAD`'s commit, or git's empty tree
when `HEAD` is unborn (`head_is_unborn`: a plain `git init`, or `git checkout --orphan`). Nothing is
committed then, so every file is new: `validate_git_ref` falls back to the empty tree rather than to
`HEAD~1` and then no base at all, `paths_in_commit` reads a tree (the empty one holds no path, so a
file implement made is `created`), and `review-scope.md` names its base as the empty tree. A first-time
user's project is exactly this repository, and every mode that opened the state used to exit 1 on it
with `GitError: git rev-parse HEAD failed`. It is fixed here, in the shared runtime, never by an
initial-commit prerequisite; `tests/test_no_commit_repository.py` runs every mode the tool exposes in
such a repository, and nothing in it commits. `get_current_head()` stays strict for callers that mean
a commit.
