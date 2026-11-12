# Write boundaries and tracking

Read when changing write preflight, containment, backups, rollback, verification, or tracked-file scope.

Back to the [development guide](../../skills/multi-llm/AGENTS.md). These notes retain detailed contracts and regression history from that guide. Paths beginning with `docs/`, `packaging/`, or `plans/` are repository-relative; other implementation paths are relative to `skills/multi-llm/`. Dated incidents explain the rules; verify version-specific claims against current code.

## Contents

- [Write safety controls](#write-safety-controls)
- [Plan output containment](#plan-output-containment)
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
as an empty set, the undeclared-write sweep is its whole verdict (`write_verdict.decide`'s verification-only rows: nothing
moved is `no-op`/`VERIFIED`, any path moved is a partial application
naming the fix — declare the files under `Files to modify` and `--resume`; its
rows, verdict stamp and mirror row add `verification_only` + `outcome_label`
(`verified` / `partial-application`, `batch_reply.write_outcome_label`) and summary.md, the batch
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
(`ApplyPhaseReport._phase_rows`), marking `(earlier run)` and a reused resume skip apart, with counts
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
decides the batch. It is a convenience copy: it lies in the repository (or the phase dir), which the batch's subagent
can write, so its checksums catch an accidental change and prove nothing against a subagent. No
command restores from it; the apply instructions tell a human how to copy it back by hand, and
that code-fix batches have none. Trustworthy native rollback needs the executor's protected
storage, a larger design (plans/apply-phase-throughput.md; Codex review 2 of U31, #2). The rules:

- *The phase dir is pinned first.* A phase dir that is a link (or not a directory) is refused
  (exit 1) before anything is read through it or written — no commit re-check record, check-tree
  record, commit bind or snapshot — in every output mode, and the snapshot's root is the pinned
  phase dir's repository, taken from the pin (`git_utils.pin_repo_root` must agree with it), never
  found again through the phase path. A phase dir no repository holds (a plan kept outside
  every repository) keeps its slots in its own `snapshots/` folder instead, reached through the
  pin as `prompts/` is (`write_prompt._snapshot_home`): nothing is chosen from the working
  directory, so every bind, record and resume of the run reaches one slot and one lock from any
  directory, and no git answer is asked for. Only a repository holding the phase dir that
  `pin_repo_root` cannot pin is refused (Fable review of 2026-11-10, F2; Codex reviews of F02:
  a slot under the working directory's repository, and then a kept record of the chosen one,
  let a bind from another directory take a second baseline). The folder lies inside the phase
  dir, outside every repository, so no census, porcelain check or sweep (all repository-scoped)
  can count it as a batch's write, and the check tree leaves the phase dir out anyway
  (`check_tree.check_tree_exclusions`). `write_tasks.json` is read for the run stamp through the
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
  (review 2, #6). A record with no stamp at all (made by a plugin version that kept none: a loop
  begun before the upgrade, or an earlier run) is refused with its own message unless the slot's
  binding names that batch of this run (this version bound it, so a record of it would carry a
  stamp): run that batch first if this loop has not, else `--resume`, which emits what is left as
  a new run that needs no earlier record.
  The general message's "record it with util batch-done" would re-record a batch already
  recorded (Fable review of 2026-11-10, F10). The `[WRITE_PROMPT_ORDER]` warning stays advisory
  for the other kinds.
- With a repository holding the phase dir that git cannot pin, or with an artifact it cannot
  read, the call refuses (exit 3).
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
`ContainmentSweep.begin`/`undeclared_writes` now add a task's declared set on dispatch and remove
it on classification instead.

**The rollback store reads each file once, through the descriptor the policy was asked about**
(Codex review of U57, 2026-11-08). The retention pass (`ContainmentSweep.store_blob`) keeps its resolving policy
check, then opens the file with `WritePolicy.open_permitted` and hashes and copies from that one
read, so a blob always holds the bytes its name is the digest of. Before, the resolving check asked
about the root on its first resolution only, and the default `allow: ["**"]` matches any path, so
a file or a folder above it swapped for a link right after that resolution was read through the
link; and each file was read twice by path, to hash and to copy. On Linux and macOS
(`OPENS_WITHOUT_FOLLOWING`) each component below the root is opened through its parent's
descriptor with `O_NOFOLLOW`, as `prompt_files` walks the phase dir, so the file opened is the one
at the census's repo-relative path and the policy is asked about that path unresolved
(`violation_relative`); a link anywhere below the root is refused, never followed (the census
never walks one). A path spelled with a `.` or `..` component or a trailing separator is refused
before either way of opening is tried: such a spelling can name another file than its lexically
normalised form, the one the walk opens. "src/a.py/" names no file, yet the resolving check drops
the trailing separator and permits it, and the walk would open "src/a.py"; "src/link/../a.py"
names what lies beside the link's target. The census never spells a path so, so this changes
nothing it retains; it keeps the answer from resting on the caller's spelling. The walk is sounder
than reading the descriptor's path back (`/proc/self/fd` is Linux-only and can be unmounted, and
macOS's `F_GETPATH` can spell the path differently from the pinned root) or comparing `fstat`
with a `stat` of the resolved path (a link put back between the resolution and the `stat`
passes). Windows has no `dir_fd` and no `O_NOFOLLOW`: the file is opened
by path, the resolving check runs again after the open, and the path's own entry must be a regular
file with the descriptor's `(st_dev, st_ino)` (`lstat` against `fstat`, as
`result_files.read_regular_text` checks). It reads once and only adds checks, so it is never
weaker than before; what it cannot see is a directory junction above the file that is there for
the open, gone for the resolving check and back for the identity check. Either way the open never
waits on a FIFO (`O_NONBLOCK`). On Linux and macOS the blob takes the source's times, extended
attributes, mode and flags from the descriptor, `fstat`-ed after the read as `shutil.copy2` stats
after copying, because restore copies them back onto the file (`_replace_with_payload`). The
pending copy is created exclusively and never through a link, its times and mode set through
its own descriptor: its name is predictable (digest, pid, thread), and `shutil.copy2` wrote through
a link a runner planted there. On Windows without `CopyFile2` (Python before 3.12, a refused
`CopyFile2`) `os.open`'s `O_CREAT | O_EXCL` is `CreateFile`'s `CREATE_NEW`, which follows a link to
nothing at the name and creates its target (the Windows CI showed it, 2026-11-12), so the copy is
made and held through one handle instead (`_written_through_handle`: `_HeldCopyCalls.create`,
`CREATE_NEW` with `FILE_FLAG_OPEN_REPARSE_POINT`, as the staged restore makes its file), and its
times and read-only attribute are set through that handle before it closes, never by path. On
Windows restore puts back the streams and attributes the store keeps too, for a file that is not
read-only, and its encryption where it can (below). Tests: `tests/test_write_sweep_sees_the_same.py`.

**The pre-batch snapshot reads each declared file once too, and an adopted write copies only bytes
that hash to its blob's name** (fable review F9, 2026-11-10). U80's rule covered the store only:
`TaskSnapshot.take` checked containment, then `shutil.copy2` and a second read for the hash, both by
path, so a declared file (or its folder) swapped for a link to an outside file in between, by a
sibling batch still running, put the outside bytes in `<phase>/pre_batch/<task>/files/` (the restore
then refused them; the copy was the exposure). Now `take` opens each declared file with the store's
walk (`_open_declared`: `_open_below_root` from the pinned root, never through a link; on Windows by
path, then `lstat` against `fstat`), reads it once (`read_capped`, the snapshot cap applied to the
bytes read) and writes those bytes, hashed as read, with the store's writer (`write_copy`, so the
payload keeps the file's mode, times and extended attributes, and on Windows with Python 3.12+ is
`CopyFile2`'s copy checked against them, as `copy2` gave it). The manifest's repo-relative path is
taken from the checked path, never resolved again. A link found where the check resolved none, or a
regular file that cannot be opened, refuses the batch (`WriteSafetyError`, nothing dispatched); an
absent path or a directory is recorded absent, as before. `TaskSnapshot.adopt` opens its retained
blob once without following a link at its name (on Windows, which cannot refuse one at the open, by
path, and the name must then be the file opened: `lstat` against `fstat`, as `_open_declared` checks;
Codex review of F01), and copies the bytes it read only when they hash to
the digest (now required with `existed=True`; the sweep always gives it). The payload's folder is
still made and reached by path, as the store's is (item 2 below). Tests: the snapshot cases at the
end of `tests/test_write_sweep_sees_the_same.py`.

**On Windows the blob is `CopyFile2`'s copy, checked against the bytes read; a refused copy is
removed, or named when it cannot be** (Codex reviews of U80, 2026-11-08). With Python 3.12+ a copy
from the descriptor's bytes would drop what base's `shutil.copy2` kept there through `CopyFile2`:
NTFS alternate data streams, file attributes and EFS encryption (a plaintext blob of an encrypted
file at rest). So `write_copy` calls `CopyFile2` itself, with `copy2`'s
`COPY_FILE_ALLOW_DECRYPTED_DESTINATION` plus `COPY_FILE_FAIL_IF_EXISTS` and `COPY_FILE_COPY_SYMLINK`,
and checks the result: the pending name must be a plain new file (a regular file with one link and
no reparse point), the source path must still name the file the descriptor read (`lstat` against
`fstat`), and the copy, read back, must hold exactly the bytes read and hashed through the
descriptor. Where `CopyFile2` refuses access it falls back to the descriptor copy, as `copy2` falls
back to `copyfile`; any other failure keeps nothing. Before 3.12 there is no `CopyFile2`, `copy2`
was `copyfile` plus `copystat` there, and the descriptor copy keeps the same. `shutil.copy2`
overwrites its destination, so an exclusively created placeholder could be swapped for a link before
it opened it; `CopyFile2` creates the copy itself instead. `CopyFileEx` documents what the flags do
with a link already at the destination: with `COPY_FILE_FAIL_IF_EXISTS` alone the copy fails "only if
the target of the symbolic link exists", so a dangling link is followed and its target created; with
`COPY_FILE_COPY_SYMLINK` as well it "fails in all cases". Windows does not do so for a regular source:
on the Windows CI (2026-11-12) a dangling link planted at the destination between its clearing and the
copy was followed with both flags, and its target created with the file's bytes. So nothing rests on
the flags for a link at the destination. The pending name is cleared (a link there removed, never
followed), but `CopyFile2` never copies to it, since it is predictable and a link could be planted
there again before the copy: it copies to a name beside it drawn at random for this copy
(`_copy_name`, `.copy.<64-bit token>.tmp`, cleared too), which nobody can know before the copy makes
it, and the checked copy is then renamed to the pending name (`os.replace`, the entry itself, never
through a link). The check after the copy still covers a copy that is a link: a copy that is a link or
another reparse point once made has its entry removed, nothing is kept, and the pass warns, naming
the link's target, since a copy that went through the link is there and a file at a path someone
else chose is not the pass's to delete. At the source the flag copies a link as a link and changes
nothing else ("If the source file is not a symbolic link, there is no change in behavior"); the
source was checked to be a regular file, so a link there is a swap and its link copy is refused. `CopyFile2` keeps the source's attributes, read-only included, and Windows will
not delete a read-only file, so removing a refused copy clears that attribute first: only when the
name still holds the copy read back (same `(st_dev, st_ino)`), a regular file with no other name and
no reparse point, and on the entry itself (`os.chmod(..., follow_symlinks=False)` on 3.13+ and 3.12's
`os.chmod` are both `SetFileAttributesW` on the path, the call CPython makes for `lchmod`). A refused
copy can hold bytes other than the file's, another file's from outside the repository included (the
junction case below), so one that still cannot be removed is never dropped like an ordinary failed
copy: `write_copy` raises `CopyNotContained`, the file gets no rollback point, the pass logs a
warning naming what is left for the user to delete, the retention warning gives that as the file's
reason, and its size counts against the store budget. The same holds, on every platform, for a
descriptor copy that fails part-way.

What these checks leave open, precisely:

1. `CopyFile2` follows a dangling link at its destination, so the copy's own name is all that keeps
   it from writing through one: planting a link there before the copy makes it needs that name's
   64-bit token, drawn just before the copy. Tests plant a dangling link at the pending name inside
   the copy call (the Windows CI's case: its target is never created, and the copy is renamed over
   the link) and, with the token pinned, at the copy's own name before the pass (removed, never
   followed). A link at the copy's name once the copy is made is refused and named, as above, but
   what the copy wrote through it stays. Between the check of the copy and its rename to the pending
   name, a writer of the store folder can swap the copy for something else, which the rename then
   moves there, never through a link; such a writer can rewrite a blob anyway (item 4), and
   `TaskSnapshot.adopt` copies only a blob that hashes to its name.
2. The folders above the pending name (`rollback_store/ab/`) are created and reached by path, so a
   junction planted at one redirects where the copy is created, as it redirected base's `copy2`
   (hardening the store's path is a design of its own).
3. While the descriptor is held, Windows refuses to delete or rename the source file (Python's
   `os.open` does not share delete), so `CopyFile2`'s reopen by path reaches that file unless a
   directory junction above it is redirected for the copy alone and put back for the identity check.
   A copy of another file's bytes then fails the bytes check and is removed as above; what the race
   can still carry into a blob is another file's streams, attributes and times, never its main bytes.
4. Between the copy and its check, only a writer of the store folder can replace the pending copy,
   and such a writer could read the copy anyway.

**On Windows a restore puts a payload that is not read-only back as `CopyFile2`'s copy, checked and
swapped in through one handle held on it alone** (Codex review 2 of U80 and reviews of U82,
2026-11-08). A snapshot's payloads are `shutil.copy2` copies of the files (an adopted write's payload
a `copy2` of its blob), so on Windows with Python 3.12+ they keep NTFS alternate data streams,
attributes and EFS encryption. Restore wrote a payload's bytes and `copystat`, so a rolled-back file
lost its streams and its hidden or system attribute, and a file encrypted on its own, in a folder
that is not, came back as plaintext. There `_replace_with_payload` now restores through
`_restored_by_copy_file2` and `_publish_held`:

1. The payload is opened and held, so it can be neither deleted nor renamed meanwhile. A read-only
   payload takes the staged restore below instead: `CopyFile2` would copy the attribute, and
   Windows opens a read-only file for writing to nobody, so its copy could not be held as in step 3,
   and reopening it to clear the attribute would give up the exclusion.
2. The payload's own name must be that regular file (`lstat` against `fstat`), and the bytes read
   through the descriptor must hash to the manifest's digest (`restore_path`, the commit write
   policy's restore, has no digest: the copy is checked against the bytes read). `CopyFile2` copies
   it to the temp name beside the file with `COPY_FILE_FAIL_IF_EXISTS` and `COPY_FILE_COPY_SYMLINK`.
   Any name already there, a link included, skips the file and is left as it is, never removed or
   followed, as the staged restore's exclusive creation refuses it: it is looked up first, since
   `CopyFile2` follows a dangling link there whatever its flags (the store's paragraph above), and
   the temp name's token is drawn for this attempt, so nobody can put one there between the look and
   the copy. A failed copy that left nothing there takes the staged restore; one that left something
   skips the file and names it.
3. The copy is bound once: opened by path as itself (`FILE_FLAG_OPEN_REPARSE_POINT`) to read, write,
   delete and set attributes, shared for reading alone (`_HeldCopyCalls.bind`). A handle already
   open on it to write or delete refuses the bind (a sharing violation): the file is skipped and the
   copy named. While the handle is held nobody can open the copy to write, rename or delete it.
4. Through that handle alone, wrapped as a C runtime descriptor: the copy must be a regular file with
   one name and no reparse point (`os.fstat`, which is `GetFileInformationByHandle`); it is flushed
   (`os.fsync`, `FlushFileBuffers`); the payload's name must still be the file held; the copy, read
   back after the flush, must hash to the digest; it takes the payload's attributes (those
   `FileBasicInfo` sets) and its access and write times, as `copystat` gave them
   (`SetFileInformationByHandle`); and it is renamed over the file's own entry, replacing it, while
   still held (`FileRenameInfo`: atomic and never through a link, as `os.replace`). Then the handle
   closes.
5. Any failure after the bind skips the file and deletes the copy through the handle
   (`FileDispositionInfo`): the name bound, never a link's target or another name of the file. The
   snapshot is kept. Nothing on this path changes, removes or renames the temp name by path, because
   between two calls a name can be swapped for a link to another file: a hard link to a file outside
   the repository put there before the bind is refused before anything changes, by its link count
   (the link alone is then deleted) or, when that file is read-only, by Windows itself (the link is
   left and named).

The refusal rules are unchanged (a file changed since the run needs `--force`, a payload that no
longer hashes to its manifest is never written), and every folder above the file is still revalidated
without following a link, where on Windows a directory junction, or any reparse point naming another
path, now counts as a link: `lstat` reports a junction as a directory, so a junction planted above the
file used to lead the restore, and the delete leg, outside the repository (other reparse points, a
OneDrive placeholder for one, hold the file itself and are not refused). Every other restore
(Linux, macOS, and on Windows a read-only payload, Python before 3.12 or a failed `CopyFile2` that
left nothing) is the staged restore below; every Windows restore refuses a junction above the file:
the refusal is in the revalidation.

What a Windows restore with `CopyFile2` keeps, for a file that is not read-only: the payload's bytes,
checked; its alternate data streams; its attributes (hidden, system, ...); its times; and the
payload's EFS encryption, though not unconditionally. `COPY_FILE_ALLOW_DECRYPTED_DESTINATION`
(`copy2`'s flag, also passed when the snapshot was taken) lets `CopyFile2` write a plaintext copy
where the destination cannot be encrypted (a volume or share without EFS) instead of failing, and a
payload that was itself written as plaintext (the snapshot's `copy2` under the same flag) has no
encryption to give. In each case the file comes back as base's restore wrote every file: plaintext
unless its folder encrypts new files. A read-only file comes back through the staged restore: its
bytes, times and attributes (read-only, hidden, system, ...), without its streams or encryption.
What it leaves open:

1. The folders above the file are revalidated by path and then reached by path by the copy, the bind
   and the rename: a junction planted at one in between redirects them, as it redirected the plain
   restore (Windows has no descriptor walk).
2. A junction above the payload redirected for `CopyFile2`'s reopen alone and put back for the
   identity check: the bytes check fails such a copy; only another file's streams, attributes and
   times can come back with the payload's own bytes (the store's item 3).
3. Sharing does not exclude a handle opened only to read or set attributes, so while the copy is held
   another process can still change its attributes or give it another name (a hard link), and a
   writable mapping of the copy made before the bind (its handle closed since) can still write it.
   Each needs a writer of the file's own folder, which could write the file itself once it is back.
4. The file put back is encrypted when its payload is. A payload made in an encrypted folder (a phase
   folder inside one) is encrypted whatever its file was, so a file that was not comes back encrypted
   when it lies outside that folder; the manifest records no attributes to tell.

**Every other restore stages the payload's bytes itself, in a file bound to one descriptor (one
handle on Windows) from its creation to its rename, and never reached by its name** (Codex review 3
of U82, code health U87, 2026-11-08). The plain restore wrote the payload into a temp file beside
the file, closed it, then ran `shutil.copystat(payload, tmp)` and `os.replace(tmp, target)` by path.
Whoever can write the file's folder (a provider CLI running in the repository) could swap the temp
name, just before `copystat`, for a hard link or a symlink to a file outside the repository: that
file then took the payload's mode and times (`copystat` follows a symlink), and the link was renamed
over the repository's file, reported `restored`. A rewrite of the temp file before the rename put
back bytes nobody hashed. Linux and macOS took that path for every file, Windows for a read-only
payload since U82. Now (`_restored_through_descriptor`, `_stage_and_rename` and the
`_publish_staged` it shares with the [plan outputs](#plan-output-containment), Linux and macOS):

1. The payload is opened without following a link and read once; its folder's revalidation is
   unchanged. The target's folder is reached from the pinned root one component at a time, each
   opened `O_DIRECTORY|O_NOFOLLOW` relative to the one above (`_open_below_root`, the walk the
   write policy's reads take), and held by that descriptor (`_HeldFolder`; both in
   `utils/held_folder.py`, below `write_safety`); every name in it is
   reached relative to that descriptor (`openat`, `linkat`, `fstatat`, `renameat`). A link
   planted at any folder on the way after the revalidation is refused, never followed.
2. The staged file is made with no name where Linux can (`O_TMPFILE`, mode 0600; ext4 and tmpfs
   make one, measured on kernel 7.0.0), so nothing can reach it by a name while it is written.
   Where the kernel or the folder's filesystem refuses `O_TMPFILE`, or `/proc/self/fd` (the one way
   an unprivileged process names such a file: `linkat` with `AT_EMPTY_PATH` needs
   `CAP_DAC_READ_SEARCH`) is missing, it is made as the temp name with `O_CREAT|O_EXCL|O_NOFOLLOW`,
   mode 0600: a name already there, a link included, is left as it is and named.
3. Through its descriptor alone: the payload's bytes are written, flushed (`fsync`) and read back;
   the hash of the bytes read from the payload must be the manifest's digest (`restore_path`, the
   commit write policy's restore, has none: its caller checked its copy), and the bytes read back
   must be the same. Then the times (`futimens`), extended attributes (descriptor to descriptor, as
   the store copies them), mode (`fchmod`) and, on BSD and macOS, file flags, in `copystat`'s
   order. Python has no `fchflags`, so the flags go through libc's (`ctypes`); where it cannot be
   called (not loadable, or a stub failing with `ENOSYS`) the file goes back without its flags and
   a WARNING names them: set by the temp name, as the first round did (`lchflags`), they reached a
   hard link swapped in there (Codex review 1 of U87).
4. An unnamed file must still have no name (another one given it through `/proc` is refused), and is
   named the temp name with `linkat`, which never replaces a name: one there is left as it is and
   named. Then, for both ways, the temp name must still be the staged file (`fstatat` without
   following against `fstat`), a regular file with no other name (`st_nlink == 1`), and it is
   renamed over the file. Before the `linkat` and again before the rename, the folder held must
   still be the folder at its path: a fresh walk from the root reaches the same `(st_dev, st_ino)`
   as `fstat` of the descriptor held. A writer of a folder above can move the folder away and put
   another at its path; the first round then published the file in the moved folder and reported
   `restored` (Codex review 1 of U87). Once renamed, the folder must still be in place and the
   file's name the staged file; otherwise the restore says so, naming where the file went (the
   folder's path now, through `/proc/self/fd` on Linux and `fcntl(F_GETPATH)` on macOS, which
   spells it as the kernel holds it: `/private/var/…` for `/var/…`), instead of `restored`.
5. A refusal puts nothing back and keeps the snapshot. Once the staged file has a name (made by it,
   or an unnamed one once linked) it is never removed: POSIX removes a name, not a file, and
   between any check of the name and `unlinkat` another process can swap its own file in, which the
   removal would delete, its only name perhaps (reproduced on base and in the first round). It is
   left, named and recorded ("... was left at <path> ...; check it and delete it by hand
   (`rm -- <path>`): until it is gone, no write check in this process counts it as a batch's file
   and no rollback in it replaces or deletes it; another process checks it as any other file"; see
   the leftover set below). A name found where the staged file was to be is
   left and named too, but it is not the restore's, so it is not recorded. An unnamed file refused
   before it is named leaves nothing. A directory at the file's path is refused before anything is
   staged (`_refuse_a_directory`, through the held folder; `restore_path`'s link leg alike): a file
   or link renamed over a directory fails once its staged name exists, so each attempt used to leave
   one more copy of the pre-batch bytes beside it (fable review F4). A directory put there after
   that look still fails the rename, and the staged name is then left as above.

**The delete leg removes the name relative to its folder held** (fable review F3, 2026-11-10). Both
rollbacks' delete leg (`TaskSnapshot.restore` for a file the batch made, `restore_path` with no
source) revalidated the path, then called `os.unlink` by path, so a folder swapped for a link to an
outside folder in between sent the removal there and deleted the user's file of that name.
`_delete_entry` now reaches the folder from the pinned root without following a link and holds it
(`_HeldFolder`, the restore's), looks at the name relative to it (`fstatat`: gone is "deleted
(already absent)", a directory is "skipped: a directory is at the path now" on both legs), checks the
folder is still the one at its path, and removes the name relative to it (`unlinkat`). A link met on
the walk, something that is not a folder, a folder that cannot be opened, or a folder moved meanwhile,
skips the path and says why. A folder that is gone (another process removed it after the
revalidation) is "deleted (already absent)", as the removal by path said: when the walk fails, a
second walk that finds a component missing (`ENOENT`, never following a link: `_folder_gone`)
confirms there is nothing to delete (Codex review of F01). What stays open: between that
last check and `unlinkat`, a writer of the folder above can move the folder away; the name removed is
then the batch's file in the moved folder, never one a link leads to. Windows removes by path, the
window a removal by path always had.

`restore_path`'s symlink (the commit write policy putting a link back) follows the same rules
(`_restored_link`): the link is made at the temp name relative to the held folder (`symlinkat`,
never over a name there), and just before the rename (`renameat`) that name must still be the link
made (`fstatat`: a link, its identity) reading the saved text (`readlinkat`), with the folder still
in place; once renamed, the target must be that link and the folder still in place, else `skipped`.
Once made, it is never removed. Base made, renamed and removed it by path, unchecked: a regular file
swapped in for it was installed and reported `restored`. Windows reaches the names by path with the
same checks, without the folder's.

On Windows (`_restored_through_handle`) the staged file is created through the handle U82's
`CopyFile2` restore holds its copy by (`_HeldCopyCalls.create`: `CreateFile` with `CREATE_NEW` and
`FILE_FLAG_OPEN_REPARSE_POINT`, to read, write, delete and set attributes, shared for reading alone),
so a name already there, a link included, fails the creation and is left, named, and from its
creation nobody else can open it to write, rename or delete it. The payload's own name must be the
file read (`lstat` against `fstat`), its bytes are written through the handle and must hash to the
digest, and U82's sequence does the rest through that handle (`_publish_bound`): one name and no
reparse point, flushed, read back and hashed, the payload's attributes and times (the read-only
attribute included, which a read-only file needs), renamed over the file. Windows deletes no
read-only file, through a handle neither, so when the rename fails after the attribute was set it is
cleared through the handle first, then the file is deleted through it. Where the handle calls cannot
be loaded nothing is put back (base restored by path there). The Win32 surface is U82's: one more
disposition to the same `CreateFile`.

What the staged restore leaves open, precisely:

1. Between the last check of the temp name (and of the folder) and `rename(2)`. POSIX has no rename
   that takes a descriptor, and `linkat` never replaces a name, so even an `O_TMPFILE` file needs the
   temp name for the rename (`renameat2` exchanging names would only move the window to removing the
   old file). What is swapped in there is what the rename moves over the file, which the folder's
   writer could move there itself; nothing is set by a name, so the file it links to keeps its bytes,
   mode and times, and the check after the rename reports the file not restored. A folder replaced
   there gets the file in the moved folder, which the check after the rename reports too.
   `O_TMPFILE` removes every window before the name exists: the swap before the metadata, the
   rewrite by name before the flush, another name given by name.
2. POSIX has no mandatory exclusion: a writer that opened the staged file before its read-back (by
   the temp name, or an unnamed file through `/proc/<pid>/fd`, as any process of the same user can)
   and still holds it can write it after the hash, before the rename. It could write the file once
   it is back anyway. Windows' sharing excludes it.
3. A refused restore can leave its staged file, or link, beside the file for the user to remove,
   named in its result: the price of never removing a name another process may have swapped in. It
   shows in `git status` until the user removes it.
4. The pinned root itself is opened by path (a link above the repository is the user's). On
   Windows the folders are reached by path as before (the `CopyFile2` restore's item 1).
5. The flags call is untested on BSD and macOS: it runs emulated here (an `st_flags` the payload
   reports, a `fchflags` that records the inode it reaches). A file whose flags include
   `UF_IMMUTABLE` refuses the rename there, as base's `copystat` then `os.replace` did, and is left.
6. `restore_path` has no digest: a copy changed between its caller's check and the restore's open is
   put back (the bytes read are what is checked).

The identity checks compare `(st_dev, st_ino)` of a name and of the descriptor, so a filesystem
whose inode numbers are not stable across lookups (some FUSE mounts) fails them: the restore is then
skipped and says why, never done unchecked.

**A staged file kept for the user is recorded in the restoring process, and that process's write
checks leave exactly it out** (Codex review 2 of U87, 2026-11-08; fable review F1, 2026-11-10). The staged name was `.<name>.restore.<pid>.tmp`. A file kept
there after a failed rename was, to another batch whose containment sweep was open, an undeclared
new file: the sweep adopted it into that batch's snapshot and its rollback deleted it, the very
removal the restore refuses; and a second restore of the same file in the same process found the
name taken and was refused. Now:

- Each attempt stages under a name of its own, `.<name>.restore.<pid>.<token>.tmp`, the token
  random (`_staging_name`): a kept file never blocks a retry, and the name cannot be known before
  the attempt makes it.
- The restore raises its own kept entry as `_StagingLeft` with its identity, `(st_dev, st_ino,
  st_ctime_ns)`, read through the staged file's descriptor (the link's `lstat`, once it is still the
  link made; on Windows the handle's, or for `CopyFile2`'s copy an `lstat` of a regular file with
  one name that is still the file the copy made). `_replace_with_payload` and `_restored_link`
  record it, path and identity, in the restoring process's memory (`_LEFTOVERS`, a registered
  process cache, per repository root, under its lock), never on disk. The change
  time is part of the identity because a filesystem can give a removed file's inode number to the
  next file made, and no call sets it. A name swapped in for the staged file, or a link found at
  its name, is not the restore's and is never recorded.
- **Only the restoring process's own record is trusted** (fable review F1, 2026-11-10). The first
  version kept the record in `$XDG_STATE_HOME/multi-llm/write-safety/restore-leftovers/<key of the
  repo root>.json`, trusted because a jailed batch cannot write the state home. That held on the
  Landlock rung only: on the uncontained rung, exactly where the porcelain check runs, and for any
  write batch on a provider with no jail, the runner has the user's rights, and a record it wrote
  naming its own edit and its own new file, with their real identities, took them out of the
  sweep's census, the porcelain check and both rollback legs (`--force` included). No file is out
  of such a runner's reach, a key file included (it can read whatever the plugin can), so no
  on-disk record or signature can be trusted on that rung; the parent process's memory is the one
  place it cannot write. An entry is added only from the descriptor of a file this process staged,
  so a runner cannot add one; it could only change the file, which changes its identity and makes
  it an ordinary file again. Nothing on disk is read as a leftover. The set lasts as long as the
  process: one command, or every run a process serves when it runs several in turn.
- `restore_leftovers(root)` reads that set; `RestoreLeftovers.holds(path)` is true only for a
  recorded path whose entry, never followed, has the recorded identity. The scheduler's census
  (`ContainmentSweep.census`) skips those entries and hands back the paths it skipped, and the
  differences between a batch's first and final census (`_swept_moves`) leave those paths out on
  both sides: a file the first census saw before the restore recorded it is then neither a new nor
  a deleted file, never adopted (with `existed=True`, which would send it to the replace leg) and
  never a deletion attributed to the batch (Codex review 3 of U87). Both legs of a rollback
  (`TaskSnapshot.restore`, `restore_path`) refuse one, `--force` included: neither replaces nor
  deletes it, and its record stays, even for a snapshot that adopted it before it was recorded. The
  porcelain write check (`porcelain_snapshot`) drops one too; it runs in a review run, which
  restores nothing, so there it drops none (below). There is no name-pattern
  exemption: a file a batch makes with a staging-like name, or at a kept file's path once that is
  gone, is swept as any other.
- An entry that is gone, or whose path now holds another file, leaves the set at its next read
  (any census, porcelain snapshot or restore in that repository). An entry whose identity could not
  be read, or whose folder was moved, is not recorded, and the restore's message says it is
  unprotected and how to remove it.

What the set does not close: a first census taken in the instant between the staged file's
naming and the failed rename's record sees the file; if the user deletes it before the batch's final
census, that batch reads the deletion as its own, and its rollback puts the file back from the bytes
the first census retained. The porcelain write check, taken in the same instant, reads the file's
later absence from its snapshot as a write of the slot's (detection only, nothing is removed). A
`CopyFile2` copy, made by its name, has no identity from before it was made. And the price of
trusting the restoring process alone: a file a restore in another process kept (an earlier `util
restore-batch`, another run, the commit write policy's restore in its own process) is an ordinary
file to this one. It is in both of a batch's censuses, so it is no write of the batch's, unless it
appears or goes during one: the user deleting it while a batch or a review slot runs reads as that
batch's or slot's write (the batch's rollback puts it back from the bytes its first census
retained), and a staged file another process kept while a batch's sweep was open is adopted by that
batch and deleted by its rollback. Each is the checks reporting, or undoing, a change they cannot
attribute, never a write they miss. Restore keeps no hard link: the file put back is a new file, so
another name of the old one keeps the batch's bytes.

**The store's discard removes it whole, or names what it cannot.** `ContainmentSweep.discard_store` used
`shutil.rmtree(..., ignore_errors=True)`, and Windows will not delete a read-only file, which a blob
is when its file was (`CopyFile2` keeps the attribute), so there the store outlived the run, silently.
`write_safety.remove_tree` removes each entry itself, never what a link there leads to
(`shutil.rmtree`), and returns what is left; the run logs a WARNING naming it (a folder left only
because something in it is left is not named again). A refused removal is never tried again, on any
platform (Codex reviews of U82, 2026-11-08): a retry by path follows whatever link or junction is at
the entry's folder by then (`rmtree` on Linux and macOS removes through each folder's descriptor,
where a link swapped in leads nowhere), and holding every folder against that on Windows took Win32
steps that cannot be verified off Windows. What is left stays in the store and costs disk space only,
as the warning says: a blob becomes a rollback point only once its bytes hash to its name
(`TaskSnapshot.adopt`), as a killed run's leftover store does. Base ignored the refusal; the discard
now names it. What it leaves open: on Windows `rmtree` walks by path, so a junction put at a store
folder's name once listed redirects `rmtree`'s own removal, as it redirected base's (a read-only file
there is refused and named, never cleared). Tests: `tests/test_write_sweep_sees_the_same.py` (the
restore and discard tests at its end; the held copy's sequence, the hard link swapped in before the
bind, the writer holding the copy, the copy changed before the bind and the calls that never name the
copy run emulated everywhere and natively on Windows; the streams, EFS, junction, read-only blob and
held-copy sharing cases run on Windows CI, the EFS one skipping where EFS is unavailable) and
`tests/test_restore_staged_file.py` (the staged restore's adversaries, on both POSIX staging ways:
a hard link or symlink swapped in before the metadata, a rewrite before the flush, another name
given to the staged file, a swap between the last check and the rename, the staged file reached by
no path; the folder replaced during the flush or just before the rename, a link planted above it; a
failed rename leaving the staged name, which a removal would have lost to a swap; the flags through
the descriptor, or left off and named, emulated; `restore_path`'s symlink swapped once made or at
the rename, its folder replaced, a failed rename, a name already there, by descriptor and by path;
the Windows staging over the emulated handle calls; Codex's U82 round 3 repros; a directory at the
path, the delete leg's folder swapped for a link before and after it is held, and a leftover record a
runner forges in a separate process, from fable review F1, F3 and F4). The Windows staging
runs natively on Windows CI as the sweep tests' `staged copy` restore. A policy violation's reason
says "no rollback point exists" only for a path the snapshot holds nothing for; any other path left
on disk is named with the restore's own reason (`PolicyRestore.not_restored`), so a refusal with
the snapshot kept is never reported as a missing one.

## Plan output containment

**A file the plugin writes beside a plan resolves inside the plan's directory and is written through
its folder held there, or is not written; the plan itself is the one file written through a symlink
wherever it leads.** The plan is the user's own file, and a plan kept elsewhere and linked in is a
supported setup (`update_plan_tasks.replace_file`, the only `follow_symlinks=True` write in this
flow). Its outputs are not: a repository symlink named like one (the task list, its `tasks/` folder,
the plan's output folder) carried `util update-tasks`' write of `tasks.md` to any file the link
named, outside the repository included (U55's Codex review, 2026-11-08). Two steps keep every
output in: a check, then a write that cannot be redirected after it.

- **The check.** `update_plan_tasks.contain_plan_output` resolves the path (all of it, or the folder
  alone for a rename of the entry) and refuses with `TasksOutputEscape`, exit 1 from
  `util update-tasks` and exit 3 from `generate-tasks`, before anything is created: the output
  paths are computed (`output_handler.output_file_path`, `phase_dir_path`), not made, until it has
  passed. The comparison is `write_safety.contain_local_path`: both sides resolved, as in
  `contain_under_root`, without that check's refusals for a model-supplied path (drive-relative,
  UNC, an NTFS stream colon), which a real POSIX directory with a colon in its name would trip.
- **The boundary is the directory the plan is reached through, resolved** — not the plan file, so
  a linked-in plan's outputs stay beside the link — and it is the one implement reads the list from:
  `tasks_file.resolve_tasks_marker` refuses a list that resolves anywhere else, so a link elsewhere
  in the repository is refused too (it could name any tracked file, and the list it produced could
  never be implemented). A link that stays inside is resolved by the check, and the write lands
  where it leads; the link survives.
- **The write goes through the folder, held** (`write_safety.write_below_root`; the move aside and
  the restore's removal are `rename_below_root` and `remove_below_root`). A check of a path
  followed by a write by that path can always be redirected in between, by a link swapped in for
  the file or for any folder above it (Codex's review of U86a, 2026-11-09: both swaps, made just
  before the write, overwrote a file outside). On Linux and macOS the checked path's folder is
  reached from the plan's directory one component at a time without following a link, a missing
  one made on the way relative to its parent's descriptor, and held (`_HeldFolder`, the restore's,
  [above](#write-safety-controls)); the file is staged and renamed in by the restore's rules
  (`_publish_staged`, which the restore's `_stage_and_rename` uses too): made exclusively, never
  through a link, unnamed where Linux can, flushed with its mode, and renamed over the target
  relative to the held descriptor only while that folder is still the one at its path (`fstat` of
  the held descriptor against a fresh walk). A link met on the walk is refused, and so is anything
  but a file at the target's own name: the check resolved it, so a link there now was put there
  since, or leads nowhere (a loop is refused alike on every Python). An existing file's mode carries
  over and a new one gets the umask's, as `atomic_io` gives them. A staged file that was named but
  not renamed is left and named, never removed, as the restore leaves its own.
- **Windows** has no `dir_fd` and no `O_NOFOLLOW`, so the write is `atomic_io`'s at the checked
  path, which replaces a link at the name and never follows one: a folder above it swapped for a
  link or junction between the check and the write is still followed there. That window is the
  one a write by path always had, no wider than base's.
- **`generate-tasks` checks every destination before it writes anything**
  (`_contain_phase_outputs`, ahead of the prerequisite gate, the scope policy's record in
  `state.json`, the prompt sidecar's cleanup and the task file): its own folder, the executor's
  failure-capture folder in it (`failed/`), the task file, the prompt sidecar, the result document
  and the task list. A planted link costs no dispatch and leaves the whole tree, `state.json` and
  prompts included, byte for byte and inode for inode as it was. Each writer then checks its file
  again and writes through its held folder: the task file (`materialize_task_file`), the result
  document the consumer rewrites, the task list, the archive `--force` and a stale result move
  aside (`_move_result_aside`), and the consumer's restore, which puts each file back the way its
  writer writes it and only when its bytes changed: the plan through its link, `state.json` over
  the entry, the task list through its held folder.
- **The following writers under a plan's directory are held from it too** (the ones still
  written by path are listed under "Not covered here") (`utils/held_folder.py`, `write_in_plan_dir` and `remove_in_plan_dir`): `state.json` (every
  write goes through `state_file.write_document`: `StateManager`, the scope policy's record,
  `util metrics`, `util batch-done`, `util finalize-tracking`, `util write-prompt`,
  `commit_apply`; `StateManager`'s move from the old location, below, writes through its
  `write_bytes`, and `StateManager.delete` removes the file through `remove_document`, under the
  state lock), `.status.json` (`status_file.write_status`, the heartbeats included), the prompt
  sidecars and host views (`validation._write_one_sidecar`), the executor's JSON records
  (`failure_capture._atomic_write_json`: a result, `task_result.persist_result`; the executor's
  batch report; the write scheduler's `write_batch_report.json`; the task files and reports that
  helper writes), the write scheduler's rewrites of a result it persisted (the verdict stamp
  `_stamp_verdict`, the own check's re-record `_restamp_own_check`, the tree's re-stamp
  `CheckHolds.restamp_check_trees`) and its inputs binding (`ResumeEvidence.record_fingerprint`,
  `write_fingerprints.json`, its folder made held) (fable review F5, 2026-11-10: they rewrote by
  path the file `persist_result` writes held, so a phase folder swapped for a link between the two
  had an outside JSON file of the result's name rewritten), a pre-batch snapshot's manifest
  (`write_safety._atomic_write`, which writes held below a plan's directory and by path in the
  out-of-grant directory) and the generate-tasks restore's `state.json`, written or removed. The
  write audit log, `subagent_audit.jsonl`, is appended through its folder held the same way
  (`write_safety._open_to_append`): opened `O_APPEND|O_NOFOLLOW|O_NONBLOCK` relative to the folder,
  which must still be the one at its path, and the entry opened must be a regular file its name
  still holds, so a symlink at its name (a dangling one included, which the append used to create),
  a FIFO, or a phase folder swapped for a link refuses the append (`OSError`, which stops the run
  as any failed audit write does) instead of appending to an outside file (fable review F6,
  2026-11-10); in the out-of-grant directory, and on Windows, its folder is made and reached by
  path with the same checks on the name. A hard link at the name is still appended through:
  refusing a second name would stop every run in a tree a backup hard-links, and with the
  protected-hardlinks sysctl (on by default on Linux) only a process that owns a file, or can read
  and write it, can link it. Nothing a writer stated changes: its mode, `fsync`, temp-file prefix, lock, and whether it made its folder;
  only the publication is held, so a normal run writes the same bytes with the same mode and the
  same flushes (`tests/test_held_folder.py`). Codex's re-check of U86a (2026-11-09) found these
  writing by path after the run's start check, so a link swapped in during the run redirected
  them.
  - **`StateManager`'s move from the old location** (`_migrate_from_old_location`: a state file an
    old version kept in the skill's own `state/` folder, named by a hash of the plan's path) is
    the one writer whose write changed. It was a `shutil.copy2` by path, unlocked, so an output
    folder swapped for a link after its check overwrote the `state.json` the link led to, and the
    old file was then deleted (Codex's re-check of U89, 2026-11-09). It now holds the state lock
    from its check that `state.json` is absent to the old file's removal, and writes the old
    file's bytes as they are through `state_file.write_bytes` (`write_document`'s write, for bytes
    already in hand). The old file is removed only after that write succeeds; a refusal, an OS
    error, or an old file that is not UTF-8 JSON keeps it and is logged, as a failed move always
    was. The copy is owner-only, as
    every other write of the file is, rather than taking the old file's mode, and it carries the
    time of the move rather than the old file's. The next save rewrites both anyway, and the
    file's time is read only as "when the state was last written" (`status_native`).
  - **The root is the plan's directory, resolved once**: the run's first look at its outputs
    (`output_handler.output_dir_path`, which every producer and `StateManager` compute their folders
    through) anchors it (`held_folder.anchor_plan_dir`), and every later write takes the anchored
    directory its path is below by spelling (`plan_dir_of`). A command handed `state.json` alone
    (`--state-file`) anchors its grandparent, the output folder's parent, when it takes the state
    lock (`state_file.locked`), before the read its write is based on, and keeps it through the
    write. Once recorded, a link swapped in for the plan's directory or a folder above it never
    moves the root. A swap made before it is recorded (before the run locates its outputs, or
    before such a command takes the lock) is the directory the run starts from, as a link already
    there would be.
  - **The write**: the file's folder (not its own name) is resolved and must be the root or below it
    (`held_folder.OutsidePlanDir`, an `OSError`, otherwise: a static link leading out is refused, one
    that stays inside is written where it leads, as above). On Linux and macOS that folder is
    reached from the root without following a link, a missing one made on the way relative to its
    parent's descriptor where the writer made its folder, and held (`_HeldFolder`, the same walk as
    `write_below_root`'s and the restore's, moved below `write_safety` so the L1 writers can use
    it); `atomic_io` then writes relative to the held descriptor (`dir_fd`): its exclusive temp file,
    the rename over the name (a link at the name is replaced, never followed, as these writers
    always did) and the folder's flush. Just before the rename (`atomic_io`'s `before_replace`) and
    once after it, a fresh walk must reach the folder held: a phase folder or file folder swapped for
    a link after the walk is refused and never written through. A refusal is the writer's own
    failure: `state.json` and a result raise, the checkpoint and a failure capture are skipped as
    any failed write of theirs is, a sidecar falls back to the inline prompt.
  - **The failure captures** (`failure_capture._write_failure_capture`, every phase) stay checked
    below the phase folder, and are checked below the plan's directory as recorded too, then retired
    and written through `failed/` held from that recorded directory, the phase folder and `failed/`
    both walked. The recorded directory is walked from as recorded, never resolved again
    (`rename_below_root` and `write_below_root` with `resolve_root` off): a phase folder, the plan's
    directory or a folder above it swapped for a link before the capture, which its own check
    resolves on both sides, is refused too, and the capture is skipped (it used to anchor on the
    phase folder, which then moved with the link).
  - **Overhead** (measured 2026-11-09, ext4, a plan seven folders deep): a held write is about 53 µs
    against 15 µs for the plain one (`.status.json`, 0.8 KB) and 54 µs against 23 µs (`state.json`,
    33 KB); one walk is 2.7 µs, resolving the file's folder 6.6 µs, the rest is the extra
    descriptors and path objects. A mocked full chain makes about 30 such writes, 4 ms in all, and
    a heartbeat every few seconds adds well under a millisecond a minute. The folder's descriptor is
    not cached across writes: it would save one walk of the three (the two re-checks around the
    rename are the safety and stay), a few microseconds, against a descriptor held for the whole run
    and a reopen path for a folder legitimately replaced meanwhile (an archive, `--force`).
- **Not covered here.** `StateManager` resolves the plan, so a linked-in plan keeps its state beside
  the file the link leads to while its task list sits beside the link (each is held from the
  directory it is in, both anchored), and `util update-tasks` resolves the plan first, so its
  outputs land beside that file too. Kept exactly as they were, and so still by path: the lock files (`.state.lock`,
  `.status.lock`, made by `FileMutex` at their path, so a folder swapped for a link before the lock
  is taken gets an empty lock file where the link leads), and in `validation.write_prompt_sidecars`
  the phase folder's `makedirs`, the removal of the previous round's sidecars and the all-or-none
  cleanup. A pre-batch snapshot's payloads and their `files/` folder (`TaskSnapshot.take` and
  `adopt`) and the rollback store's blobs and folders are made and reached by path, each payload or
  blob created exclusively and never through a link at its name (`write_copy`); what they hold is
  read through a descriptor (the store's reads, above). The other files in the output folder are
  written by path; holding one is the same call.
  A sweep of the raw writes (`open(..., "w")`, `shutil.copy*`, `os.replace`, `os.rename`) on
  2026-11-09 found these: the reports, the grouped, consolidated and validation documents and the
  HTML report (`consolidation`, `apply_report`, `apply_output_helpers`,
  `write_reports.write_summary`, the code review's issue files); the review slots' results, error
  files, logs and salvage requests (`review_orchestrator_base.write_model_result`,
  `save_model_result` and `_write_salvage_request`, `code_review_orchestrator`'s own,
  `llm_client._save_log` and `_adopt_misnamed_result`), and their moves aside
  (`result_files.set_aside_*`, `retire_salvage_artifacts`, `retire_unselected_results`,
  `retire_result_from_another_scope`, `write_resume.retire_foreign_result`,
  `scope_apply.retire_task_file`); the task files `subagent_tasks.write_task_file` writes
  (`write_tasks.json`, `salvage_task_*.json`, `human_decision_tasks.json`; the ones
  `failure_capture._atomic_write_json` writes are held, above), and the fix, consolidation and
  revalidation task files; the copies made before a change (`backup.backup_before_write`, the
  apply phases' target backups, a review round's `backup.md` or `tasks-backup.md`,
  `apply-task-suggestions`' task-list backup); the ask mode's answers and logs; the code review's
  diff files (`review_diff`); the changelog (`output_handler.append_to_changelog`), the record of
  earlier batches (`prior_changes.append`), and a detached run's rotated log and copy of its
  question (`detach`); and `util write-prompt --output`, a path the caller named. A file below no
  anchored directory (a tool or a test handed a folder directly) is written by its path. The root
  itself is opened by path, so the plan's directory swapped for a link in the instant between a
  write's check and its walk is followed (the re-check before the rename walks the same way); a
  swap after the root is recorded and before the check, or after the walk, is refused, and a swap
  before it is recorded is the directory the run starts from (above). These root guarantees hold
  for the writers in this list; the task list and the other `write_below_root` writers above
  resolve the plan's directory at each check instead. Windows writes by the checked path, the
  residual above.

Tests: `tests/test_update_plan_tasks.py` (`TestTheTaskListStaysInThePlansDirectory`,
`TestALinkSwappedInAfterTheCheckIsNeverWrittenThrough`),
`tests/test_generate_tasks_orchestrator.py` (`TestTheOutputsStayInThePlansDirectory`,
`TestEveryWriteGoesThroughItsHeldFolder`), `tests/test_plan_dir_writers_held.py` (each writer, a
link swapped in just before its write) and `tests/test_held_folder.py` (the root, the refusals, the
fallbacks, and a normal write against each writer's own).

## Backups

**Backup before modification**: Always backup files before any salvage or modification operation.

## Code-phase tracking

**`tracked_files` is the code phases' record, and the workflow's own files are never in it**
(2026-10-05, Codex racing run 3: apply-task-suggestions' census recorded the plan and
`tasks.md`, a partial implement recorded nothing, and review-code reviewed
`tasks.md` as code while the `index.html` implement created was out of scope).

Only `implement` and `apply-code-fixes` write it (`batch_prompt.CODE_WRITE_KINDS`; the
planning phases' `_finalize` no longer calls `finalize_tracking`), each entry's
`task_id` names the phase that recorded it (`finalize_tracking.TRACKING_LABELS`:
`implementation`, `apply-code-fixes`; `util finalize-tracking --phase`), and `finalize_tracking.phase_tracked_files`
replaces only its own phase's entries — the other code phase's keep their label but have their
action read off the tree again (`_path_states`; a recorded file now on no disk and in no
base commit — created, then removed — is left out and said, fix round 8; one git's census names is
kept as `deleted` under the phase whose census found it, fix round 9), a path they hold
is not claimed again, any other label is replaced.

It is recorded as each code batch finishes (`VerifyStep.record_batch_tracking`, in memory through the run's own
`StateManager`; `util batch-done` on the native leg, census only), so a run that stops
part-way has recorded its finished batches, and a resume's `--touched` set carries the
earlier runs' `ok` report rows (`ApplyPhaseReport.touched_files`).

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
