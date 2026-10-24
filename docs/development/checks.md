# Checks and human task gates

Read when changing check findings, user answers, manual-task dependencies, accepted findings, or batch verification reports.

Back to the [development guide](../../skills/multi-llm/AGENTS.md). These notes retain detailed contracts and regression history from that guide. Paths beginning with `docs/`, `packaging/`, or `plans/` are repository-relative; other implementation paths are relative to `skills/multi-llm/`. Dated incidents explain the rules; verify version-specific claims against current code.

## Contents

- [Check findings and dependent tasks](#check-findings-and-dependent-tasks)
- [Answerable stops and resumes](#answerable-stops-and-resumes)
- [Manual tasks that block implementation](#manual-tasks-that-block-implementation)
- [Declining or completing manual tasks](#declining-or-completing-manual-tasks)
- [Accepted checks and batch failures](#accepted-checks-and-batch-failures)
- [Known repository checks](#known-repository-checks)
- [Repeated accepted findings](#repeated-accepted-findings)
- [Attributing accepted findings](#attributing-accepted-findings)

## Check findings and dependent tasks

**A check that reported problems is `check-findings`, never `verified`, and holds
back only the tasks that depend on it** (racing re-run 4, 2026-10-05: Codex T007's
`npx html-validate` found two problems, the runner kept them in `notes` and the
task read VERIFIED; in run 3 the same kind of check, worded "must exit 0", claimed
`failed` and stopped implement for good).

A check task's result contract (`write_batch_scheduler.CHECK_RESULT_CONTRACT`, appended by `with_result_contract` on both
legs when `is_check_task`) asks for a structured `check: {passed, findings}` (optional and untyped
in `write_batch_result.schema.json`: an older reply stays valid and a near-miss shape is read, never a schema
failure — fix round 11, when implement had no report re-ask; it has one since fix round 14, and the
lenient reading stays so a near-miss costs no re-ask); `subagent_executor.check_findings_from` reads it leniently,
never `notes` (only a clear pass with nothing listed is a pass;
`{}`, `check: null` or an unreadable `passed` is a
`CHECK_RESULT_UNKNOWN` finding, and a bare
`"none"`/`"n/a"`/`"no problems found"`/`0` beside a
pass is no finding — fix round 12), and `_verification_verdict` records `passed: false` or any
finding as `no-op`/`ok` labelled `check-findings`; a crash,
or a `check` saying `could_not_run: true` (`read_check_reply`; the contract's
word for a check that could not run, on the native leg `util batch-done --status failed --check could-not-run`), is a failed task
whatever its `check` lists, and so is `failed` listing no finding;
`failed` whose `check` lists findings without `could_not_run` is
`check-findings` (fix round 12, a coordinator decision: racing run 3's runners answer
`failed` for a check that ran and failed, and round 11 read every
`failed` as could-not-run, which stopped implement on all of them).

The findings reach the rows' `check`, the verdict stamp, summary.md's
`## Check Findings` and accounting line, the batch log line, `[CHECK_FINDINGS]` lines (on a
stop as at the end of a run, since fix round 14; on the native leg from `util batch-done --check findings`,
`check_findings_line`, fix round 26), the mirror's `check_findings`/`check_findings_task_ids`,
implement's result `notice`, the prior-changes line and `state.json`'s
`task_check_findings` (`state_manager.set_task_check_findings_in_state`; on the native leg `util batch-done --check passed|findings --finding …`), which
`status`'s trailer (while implement is unfinished, then until code-review is completed
or skipped), a terminal poll's `notice` and review-code's implementation record read
through `StateManager.implementation_check_findings`.

Fix round 11: its transitive dependents are not run (`SchedulerResult.blocked_by_check`, each
`pending` with `blocked_by_check_reason`), and once every other task ran implement ends
needs-human with a `check-findings` stop record (`_record_check_block`, summary.md's
`## Not Run: Upstream Check Reported Problems`, `blocked_by_check` in the report and mirror); a check with findings on
record, completed and not accepted (`rerunnable_checks`), is materialized again on
`--resume`, answered with `--human-task <id>=done|skip|retry`, and re-dispatched on retry or when the
tree changed since the run ended (`_check_findings_rerun`; `_restamp_check_trees` stamps the tree the
run left); on the native leg `util batch-done --check findings` records the held dependents and prints
`[CHECK_BLOCKED]`, `util write-prompt` refuses them (`[WRITE_PROMPT_BLOCKED]`) and
`util finalize-tracking --mark-completed` refuses while one is left.

Every one of those walks, and the skip walk from a code task answered `skip`, runs
over the whole decomposition (`decomposition_graph`: the materialized tasks plus
`state.json`'s `implement_task_depends_on`, which `record_task_titles` records), never a
resume's subset, and the refusal is computed from it (`held_back_by_checks`), never from a recorded
reason alone (fix round 12: a task whose chain to the check ran through a finished manual task was
dispatched on `--resume`, and with two checks holding one task, the first passing
cleared the hold).

## Answerable stops and resumes

**A stop names every answer it needs, and a task refused before dispatch can be answered** (fix
round 13, the rerun-5 check-gate fixture). A check whose reply said `failed` while
listing findings counts as done on `--resume` when its verdict stamp reads
ok/`check-findings` (`_findings_from_a_failed_claim`; the runner's `failed` stays in the
file), so `_check_findings_rerun` decides its re-run as for any check.

A task refused before any runner started (the write policy, the grant, the snapshot, the census, the
result contract) is recorded `failed` with its reason through `_record_batch_effects`
(`_refuse_before_dispatch`: no metric, batch line `refused before dispatch`), so `--human-task <id>=done|skip|retry`
answers it and `--resume` checks it again; its stop says `refused before it was dispatched` and
offers skip/done (a policy refusal also offers allowing the file, worded as the user's own call and
naming it, `write_safety.policy_allow_clause`/`policy_skip_clause`, on both legs and in
`WritePolicy.check`'s message; every host is told `modes.STOP_POLICY_RULE`: never edit
`driver.write_paths` itself, relay the ways forward the stop names as it words them, add no policy
advice — fix round 20, replacing round 18's "never suggest loosening"; fix round 21: worded per rule
from `PolicyViolation.hits` (`write_safety.PolicyHit`, recorded beside `policy_paths` as the
stop's `policy_rules`): a deny hit names each deny rule to remove or narrow, since deny wins
and "allow it" resumed into the same refusal, an allow miss says to add the file to
`allow`, and a file outside the repository, named as the task declared it, gets the
skip alone, `policy_outside_note`), and every stop that leaves tasks held by a check with findings
appends that check's `done|skip|retry` answers and one combined resume command
(`_held_by_checks_lines`, `user_commands["answers"]`; `stop.blocked`/`check_task_ids` and
the headline).

On the native leg `util write-prompt` refuses a task the policy denies (`[WRITE_PROMPT_REFUSED]`,
`policy_refusal`) and records the files git sees when it binds a check task
(`prompts/<id>.tree.json`, `git_utils.worktree_file_stats`), and `util batch-done --status completed` refuses that check
while a file changed since that no other task of the phase declares (one record verifies one
dispatch; no record, no guard: a host that skipped `write-prompt` is not checked, and like
the executor's sweep it cannot tell the check's edit from another process's).
`check.passed`/`could_not_run` read `"1"`/`"0"` as
the numbers, and `record_task_titles` stores each dependency list sorted.

Fix round 14: the native emit's `batches` hold only what the host dispatches
(`_native_withheld`): skipped, accepted and manual-done tasks, a task the policy refuses
(recorded `failed` at emit, `refused_tasks`) and its dependents
(`not_run_tasks`), and the tasks a kept check holds (`held_tasks`) are named in
top-level maps, `write_tasks.json` is written without them and the counts are recomputed (fix
round 16: every count the emit prints and hands over is read off those batches,
`_handoff_counts` — `Organized into N execution batches` on both legs, `Batches:`, and
`tasks_to_dispatch`/`Tasks to dispatch:`, which the core passes as `--total-batches` — and
a handed-out task that depends on a check handed out with it is named in `waiting_on_checks`,
`_waiting_on_checks`); a check with findings is handed out again on a native
`--resume` only when `write_prompt.check_tree_fingerprint` (the executor's exclusions) differs from
`state.json`'s `task_check_trees` (`util batch-done --check findings` stamps it,
`util finalize-tracking` re-stamps it last — also when it refuses `--mark-completed`, fix round 15
— the emit drops it when it hands the check out; a check the `--exec-subagents` leg recorded is
read off its verdict's `tree_fingerprint` while the findings match and adopted,
`_executor_check_stamp`, fix round 15), or on `retry` — else `[CHECK_NOT_RERUN]`
and `not_rerun_checks`.

## Manual tasks that block implementation

**A manual task that every code task left waits on is a stop, never a finished implement** (fix
round 28, live run 12, Codex racing 06: T010 gated nine code tasks, `--yes`'s
`skip-dependents` skipped them, and implement read `done`, exit 0, "Organized
into 0 execution batches", review-code next on an empty tree).

One pure rule, `state_manager.manual_task_gates` (over `implement_task_depends_on`; `StateManager.manual_task_gates` from
`state.json`, `write_batch_scheduler.plan_manual_gates` from a resolved plan): `closable`, the
owed manual tasks whose own dependencies are done (a later manual task that comes after code not run
yet, `after`, is never in a close command: `manual_close_ids` is
`manual_tasks_to_close` over it), `waiting`, the code tasks a resume runs once those are
recorded done, and `gated`, every code task left waiting on a manual task. The
workflow payload carries `manual_waiting`, each later manual task's `after`
(`ManualTask.after`, its line says what it comes after) and, while implement is not finished,
`manual_gate`.

An implement run that started no runner on a gate stops: exit 3, implement recorded started (so
every reader offers its resume), a `manual-gate` stop record (`manual_task_ids`,
`waiting`, `resume_flags`, `resume_why`; headline
`manual_gate_head`/`manual_gate_way_on`, both legs), the closing `[ETA]` line
and summary.md's `## Not Finished: Waiting On A Manual Task`; the native emit prints `[MANUAL_GATE]` and returns 3
(runbook step 3).

Every reader's way on is `--resume --human-task <gating id>=done` only (`next_phase.manual_gate_flags`,
`_resume_flags`: `util next-command`, `--blocked`'s
`flags`/`why`/`markdown`, `status`'s
`resume with:` on both legs), no review-code beside it; a resume of a completed implement
that runs nothing stays completed. A run that ran some tasks and left others waiting stays done, and
`next_phase_commands(after implement)` leads with that resume (`why`), the next phase kept as
`user_next_step.instead` ("Or review the code implement has run so far:"); the note and command under
the tasks are dropped when they repeat the command leading the section.

`manual_close_lead(count, waiting=…)` never says "No later phase waits" when a code task waits (all four callers).
A task no runner started (refused before dispatch, a check not run again, a batch kept at your
request) logs no `batch N:`, so the batches shown are the ones `Organized into …`
counts.

## Declining or completing manual tasks

**A manual-task gate offers the user's decline, gives one record-done command, and every reader
leads with its resume** (fix round 29, the Fable review of round 28, gate #1-#4).

Under a skip strategy an explicit `--human-task <manual>=skip` is the user's decline
(`resolve_human_task_plan`: `ANSWERED_SKIP_PREFIX` + `MANUAL_DECLINED_TAIL`, so `HumanTaskPlan.declined()`
and `StateManager._deferred_to_user` read it as not owed; under `skip-dependents` its walk runs first and
takes over a later manual task a deferral reached, while a failed code task's `skip`
keeps fix round 8's rule; under `skip-continue` the code after it still runs); the gate's
stop records it as its `answers` (`next_phase.manual_gate_decline_answers`, the combined one first over
several gating tasks), `manual_gate_way_on` names it after the resume, `util next-command --blocked` and
`--after implement` (`gate_decline_steps`) and the native `status` give it on
both legs, and a run that then dispatches nothing completes, its `phase complete` line naming
what the plan set aside.

While a gate holds, the note under the deferred list records the gate's tasks only
(`manual_note_ids`; `manual_close_lead(named=…)` names them beside a free manual task, which stays
listed), so `gate_resume_closes_the_manual_tasks` holds and the lookup drops the note as the section's own command.
The native emit counts no handed-out code task as done: a manual task after one is left out of
`[MANUAL_TASK_NOTE]` and listed with `after_code_note`, the report's lookup giving its
command. After a completed implement whose code still waits on a manual task the user can do now,
`status`'s `suggested next` and a lookup with no `--after` lead
with that resume (`next_phase.gate_resume_suggested`).

## Accepted checks and batch failures

**A task released by an accepted check is told so, and a code batch whose own check fails is never a
clean success** (fix round 14, rerun 5).

A task with a check upstream of it (transitively, over `decomposition_graph`) whose findings the
user accepted (`--human-task <check>=done`: `task_check_findings` on record, `completed`, an
`accepted —` reason; `accepted_checks_on_record` / `accepted_upstream_checks`) gets a
`## Check findings the user accepted` section after its prior-changes context (`accepted_checks_section`, bound by
`bind_prior_changes(accepted_checks=…)` on both legs: the scheduler reads its state at dispatch,
`util write-prompt` the plan dir's `state.json`), naming the check and its findings —
at most 5 per check, each one line quoted whole as recorded (fix round 18: the record's own bound,
`CHECK_FINDING_MAX_CHARS`, was 200 here and cut a native finding), quoted as JSON strings under a line
saying they are data — and telling it to go ahead and say in its `summary` what it
could not do honestly; every stop text offering `<check>=done` says the held tasks then run
told so.

Every batch that edits code (`CODE_WRITE_KINDS`, not a check task) is asked for the checks it
ran on its edits in the check task's `check` shape (`WRITE_CHECK_CONTRACT`; the last
run of each check, an attempt it corrected is not a finding — fix round 16, Codex racing run 6's
T006 reported `passed: false` for a corrected check that passed; the corrected run is the
same check over the same scope, never a narrower one such as one test file or `--passWithNoTests`
— fix round 17; since fix round 26 required in every reply and shown in the result shape, with
`ran` naming each command and how it ended or an empty `ran` and a
`not_run` reason, which is no check and never a pass — `ran_no_check`, natively
`--own-check none` — after live run 11's replies that ran `npm test` sent no
`check`), read by `subagent_executor.batch_check_findings` (no usable report is `None`,
never a finding; never `notes`; unlike a check task's reading, a clear
`passed: true` is a pass whatever it lists beside it, and `could_not_run` is one
`BATCH_CHECK_COULD_NOT_RUN` finding every surface words "its check could not run", fix round 15; fix
round 27: one rule on both legs, stated once in `references/own-check.md`, which both runbooks point to
for `--own-check` — beside a `ran` that names nothing it ran, a
`passed: true` is no verdict, so a finding it names counts and `passed: false` keeps
`CHECK_FAILED_WITHOUT_FINDINGS`, and only a reply that names no finding, does not say it failed and ran
nothing (a `not_run` reason with no `ran` included) is no check; a
`ran` that names no failure never clears a `passed: false`, which stays
`listed`'s rule alone) into `TaskOutcome.batch_check` — its own axis, never
`check_findings`, so it holds nothing back — and a failing one rides the row and verdict stamp
(`batch_check`), the batch log line, `[BATCH_CHECK_FAILED]` lines (`batch_check_line`,
both legs; since fix round 16 every one-line surface says it is the batch's own report and quotes
it, `quoted_findings`), summary.md's `## Check Findings` (`batch_check_section`, both
summaries; the outcome cell/heading reads "applied, its check fails"), the mirror
(`batch_check_task_ids`, per-task `batch_check_findings`), `status`'s progress line
(`own check fails …`) and the result `notice` (`write_result_notice`; a terminal
poll rebuilds it from the run's mirror, `run_status._write_result_notice`); on the native leg
`util batch-done --own-check passed|findings|none` records a code batch's and prints `[BATCH_CHECK_FAILED]` (fix round 18:
`--check` is a check task's verdict alone and is refused on a task that declares files,
naming the `--own-check` call); fix round 24, live run 10: Claude Code left
`--own-check` out of all ten implement records while every reply said its check passed, so
both runbooks' command block carries `{check_flags}`, explained on the line under it, and a
`completed` record given no `--own-check` (a check task, no
`--check`) prints `[OWN_CHECK_NOT_RECORDED]` / `[CHECK_NOT_RECORDED]` naming the same call,
every flag kept, plus the result — advisory, nothing recorded in its place, `batch_done.check_not_recorded_line`);
an implement task's failing own check is also on record in `state.json`'s
`task_batch_check` (both legs, never `task_check_findings`; `status` and
review-code's implementation record read it through `StateManager.implementation_batch_check_findings`, fix round 15).

A code phase that finished with no `driver.verify_command` — or, since fix round 16, stopped after
one of its tasks applied an edit (`_note_unverified_on_stop`, every stop path; run 6's stop 01 said
nothing; fix round 17: an edit, so a check or a task answered `=done` alone does not
count, and `_run_verification` asks the same `_phase_has_applied_batches` question) — carries
`SchedulerResult.unverified` (`unverified_line`: nothing ran the tests, add `verify_command`
under `driver:`, naming `verify_command_candidate`'s command — package.json
`scripts.test` by lockfile, a Makefile `test:` target, Cargo, Go, pytest
config — never run or written; since fix round 24 led by what did run when anything did, then "The
plugin itself ran no tests": implement's tasks' checks by outcome, `implement_checks_ran` over
`state.json`'s `task_own_check_ran` (both legs) and the completed check tasks, a task
whose own check only repeated accepted findings in its own `repeated the accepted findings of T002 in T007` clause, never
under `reported problems` (fix round 25), apply-code-fixes' tally when a batch ran a known check,
`checks_report`) in its messages, `write_batch_report.json`, the mirror, the Verification line
and the notice — and, on a stop, at the end of the stop's headline, the result line's
`error`, which the hosts relay where they never relayed `notice` (fix
round 18; fitted to `cli_common.ERROR_MAX_CHARS` by `write_batch_scheduler.fit_stop_headline`: the model's quoted words give
way first, then the not-run list and the task titles, never the remedy, the answers or this line —
fix round 19), as an apply phase's stop now records its headline as `error` too.

## Known repository checks

**An apply-code-fixes batch is told the repository's known checks, and the plugin still runs none of
them** (fix round 20, live run 8, Codex racing 08: batch 0 added `rankFinal` assertions
to `scripts/check-race-logic.mjs`, batch 1 removed `rankFinal`, no `driver.verify_command` was set
and nothing ran the check).

`utils/known_checks.py` lists them (`known_check_commands`: `driver.verify_command`, then the
backticked commands implement's `completed` agent tasks name — a check task's description
and criteria, a code task's criteria, `commands_in_text` — then `verify_command_candidate`'s; at
most six, each once; fix round 21: a task-named command only from the task list implement ran, whose
bytes still hash to its `write_tasks.json` `inputs` (`implement_task_list_text`; a
batch can edit `tasks.md`), and only a plain check command, `is_check_command`'s
strict allowlist: no shell operator, substitution or redirection, a known runner form, no
deploy/migrate/seed/release/watch/serve/`--fix`-style word; `go test ./...` is
kept) and both legs bind `known_checks_section` after the code batch's `## Checks you ran` at
dispatch (`with_result_contract(checks_section=…)`: the scheduler's `_known_checks_section`, `util write-prompt`),
apply-code-fixes only, nothing when none is known: run each after the edits in its own sandbox,
never as root, skipping one it judges unsafe as `not run`, report each in
`check.listed` as `passed`/`failed`/`not run` (with
`passed: true` when none failed; a `passed: false` with no finding beside a
`listed` that names no failure reads the same, `with_listed_failures`, both legs),
never weaken a check, plus the earlier batches of this run whose own check failed, quoted as data.

A command `listed` as failed is a failing own check (`with_listed_failures`, the
`[BATCH_CHECK_FAILED]` path, non-gating). Each batch's report is in `state.json`'s
`apply_fix_batch_checks` with its `prior_changes_id` and `grouped_sha256`
(`review_stamp`: a fix a later review reports again makes the same `prior_changes_id`,
so only both keep another cycle's record from being read as this run's; `--fresh` drops
the map), written by the scheduler and by `util batch-done --listed-check <command>=<outcome>`; native `status`
names a failing one.

With no `verify_command`, the tests-not-run line ends with `checks_tally` (per
command: passed in / failed in / not run in / not reported by which batch), on the executor's finish
and as `[TESTS_NOT_RUN]` from the native `apply-code-fixes --mark-completed` (`native_apply_fixes_unverified_line`); not
on a stop, whose headline budget it would eat. Implement's tasks are not told: each carries its own
criteria, and an accepted check's findings would come back as every later task's failure.

## Repeated accepted findings

**A released task's re-run of the check the user accepted is the accepted findings, not its own
failure** (fix round 22, live run 9, OpenCode accept 02: T007 re-ran T002's `node scripts/check.mjs`,
reported the accepted failure in its own `check`, and the phase read
`[BATCH_CHECK_FAILED]` / "not a clean finish"). `accepted_checks_section` tells the task the problems
are known, not its to fix and not a failure of its own check, that it may cite them and need not run
the check again.

`write_batch_scheduler.split_accepted_findings` takes out of an implement task's own-check findings each one that names the
same command as an accepted finding (since fix round 23 `accepted_findings_for`: only the accepted
checks upstream of the task, which its prompt's section named, and none whose command the task's own
title, `## Task` section or criteria name or whose findings they ask it to fix or make
pass, `task_owns_the_check`; the text before its `:` or before its first
`exited`/`failed`/`printed`-type word, quotes and case
aside) and whose words about what failed are one accepted finding's, all of them or its first words
in order (fix round 23: never a middle slice or several findings read together; a whole exit clause,
never a count, and `_FINDING_FILLER` wrapper words aside; fewer than `ACCEPTED_MATCH_MIN_WORDS`
words must equal it); no command, no words, another number or one more problem stays the task's own.

Both legs (the scheduler's classification, `util batch-done --own-check findings`) record those as
`batch_check_accepted` (row, verdict stamp, resume skip, and since fix round 23
`state.json`'s `task_batch_check_accepted`, which `status` names as
`completed, its check repeated T002's accepted findings: T007` on both legs; never `task_batch_check`), print `[BATCH_CHECK_ACCEPTED]` and
name them under the accepted check in summary.md's `## Check Findings`; apply-code-fixes is not
matched.

## Attributing accepted findings

**A released task reports a repeat of an accepted finding one exact way, and a code task's run of a
check task's failing check is that check's finding** (fix round 28, live run 12: stop accept 02 on
Claude Code and OpenCode wrote "(known problem the user accepted in T002)" / "— the accepted T002
problem (…)" into T007's finding, and OpenCode accept 01's T001 ran T002's release check unprompted;
each read as the task's own failure). The strict match (`split_accepted_findings`, the user's F3) is
unchanged.

`accepted_checks_section` asks for the accepted finding copied exactly, nothing added, and any comment
in the `check`'s `note`; `WRITE_CHECK_CONTRACT` asks every code batch
for findings that are the command and what it reported, nothing else, and a `note`
for its own words (`subagent_executor.batch_check_note`: the row, the verdict stamp, `state.json`'s
`task_own_check_note` or an `apply_fix_batch_checks` entry's `note`, the
`[BATCH_CHECK_*]` lines and summary.md; natively `util batch-done --note`, refused without
`--own-check`; never read as a finding).

The backstop is a re-ask, never a classification: a finding the match left the task's own that holds
an accepted finding verbatim with words added (`accepted_findings_with_words_added`) gets the executor's one
report re-ask (the `restate_findings` hook, the incomplete report's allowance, the re-ask
prompt's `restate_findings` form), whose restated `check` is taken only through
`merge_restated_check` (the first report's
`passed`/`ran`/`listed` kept, its findings replaced
only when each changed one was cut down to an accepted finding word for word and the words cut are
in its `note`), and natively a `[FINDING_RESTATE]` line naming the
`util batch-done` call that records the task again with the accepted finding and the added
words as `--note` (`words_added_to`), to run only when they are a comment.

An implement code task's finding that is the same finding (that strict comparison) a check task of
the run reported (`reported_checks_on_record`, `repeated_findings_for`: never its own check, any check the
user accepted — fix round 29: upstream of it or not, so a task the acceptance never released keeps
the accepted finding as its own failure, round 23's scope — or one `task_owns_the_check` gives it)
is recorded as repeating it — `batch_check_repeats` on the row, the verdict stamp and
`state.json`'s `task_batch_check_repeats`, never `task_batch_check` — by whichever of the
two is recorded second: the scheduler under its effects lock (`_split_check_repeats`;
`move_check_repeats_in_state` through `_move_repeats_of`, which also moves the code task's outcome,
earlier row and verdict stamp; a task the run did not materialize is judged from
`state.json`'s `task_own_text`, the words both legs keep beside a failing own
check, and left as it is with none — fix round 29: a `--resume` rebuilt it from its
title, and a task asked to make the check pass had its failure moved) and `util batch-done`
both ways; `[BATCH_CHECK_REPEATED]`, the batch line, summary.md under the check's heading (fix round
29: quoted as what the check reported then once the check, run again, no longer lists it; the record
stays, what the task's run saw), `status`'s `completed, its check repeated T002's findings: T001`
(`run_status.repeated_check_clauses`, wherever the accepted clause is), the mirror's `batch_check_repeats` and
the tests line's `repeated the findings of T002 in T001` name it.

An implement code task is told which checks are its to run (`IMPLEMENT_WHICH_CHECKS`), and every write
batch not to stage, commit, stash or push (`with_result_contract`'s `## Git (required)`, after
OpenCode racing T012's `git add`; nothing watches the index after a batch).
