# Dispatcher and result protocol

Read when changing CLI parsing, mode registration, utility commands, exit codes, or the machine-readable result line.

Back to the [development guide](../../skills/multi-llm/AGENTS.md). These notes retain detailed contracts and regression history from that guide. Paths beginning with `docs/`, `packaging/`, or `plans/` are repository-relative; other implementation paths are relative to `skills/multi-llm/`. Dated incidents explain the rules; verify version-specific claims against current code.

## Contents

- [Entrypoint and result protocol](#entrypoint-and-result-protocol)
- [Mode registry](#mode-registry)
- [Shared CLI parser](#shared-cli-parser)
- [Repository-root path resolution](#repository-root-path-resolution)
- [Utility commands](#utility-commands)
- [Import safety](#import-safety)
- [uv project selection](#uv-project-selection)
- [Shared flags](#shared-flags)
- [Mode and help updates](#mode-and-help-updates)

## Entrypoint and result protocol

**`multi_llm.py` is the single entrypoint.** Every mode is `uv run --project <skill-dir> -- python <skill-dir>/multi_llm.py <mode> [<positional>…] [--host <id>] [mode flags…]`, so a host
shim resolves one path once and routes nothing itself. `<mode>` is a bare word and
must be `argv[1]`; unknown/missing is exit 3 listing the valid modes. Positionals are
normalized into the flags the target already parses (plan modes
`<plan>`→`--plan-file`; `ask`
`<plan> <question>`→`--plan-file` + `--question`, falling back to a
dispatcher-written `--question-file` past `MAX_ARGV_QUESTION_CHARS`); an extra positional is exit
3.

Everything after the positionals is forwarded **verbatim as a Python list** to that mode's
`main(argv) -> int`, imported lazily and called **in-process — never `subprocess`**
(one `uv` resolution per invocation, a plan path with a space survives by
construction, `shlex` never runs on this path). `multi_llm.py <mode> --help` prints the
target parser's help. Exit codes are uniform and never remapped: `0` done,
`1` unexpected error, `2` nothing to do / phase already
complete, `3` needs a human. **What a phase *found* is never an exit code** —
findings are artifacts and bracketed stdout markers, so a review that completed is
`0` however serious its findings are.

`review-code` used to return `1` when it found HIGH-importance issues,
which claimed an error that never happened: it stopped `full` at
`review-code`, one phase before `apply-code-fixes` (the phase that fixes what the
review just found), and put `state: "error"` on the result line of a successful run. It now
prints `[HIGH_PRIORITY_ISSUES] <n>` and returns `0`; `--fail-on-high` is the
opt-in for a standalone CI gate and is exit `3`, never `1`.
Every invocation ends with exactly one machine-readable stdout line, `MULTI_LLM_RESULT: {"mode","phase","state","exit","artifacts":[…],"next":…}`, emitted
even on a crash (`next` is `null` when no follow-up is owed).
**Optional** keys join them.

`consolidation_recommended` (`{valid_groups, threshold, command}`), present only on a review phase that crossed
`CONSOLIDATION_RECOMMENDED_THRESHOLD` — written to `{phase}/.status.json` at the same site, carried through
`detach.classify_run` and re-emitted (with one rendered line) by `status --view run`, because
the `[CONSOLIDATION_RECOMMENDED]` marker is a bare `print` and a detached caller never reads
the log: measured 2026-09-16, a 29-group run printed the marker and the session model told the user
no consolidation had been recommended. Deliberately **not** `next` —
`--consolidate` is one of the dispatcher's `mechanical_continuations`, which the opencode plugin
auto-follows without asking, and following a recommendation would run a phase nobody requested.

Two more ride that line on the poll path, both recorded by `status` when the caller
named exactly one run (one view plus
`--phase`/`--question`/`--run-id`): `progress` — the
watched run's own `.status.json` `progress` object, carried through **whole and
unreworded** (a test pins that no key is added to it) — and **`progress_lines`**, its sibling:
the same run *already rendered*, an array of opaque human-readable strings a host may print
verbatim.

In order, each present only when it applies: the fan-out roster, the current-state line (the
ticker's `[PROGRESS]` while a slot is live, the human view's `progress:` once
every slot is terminal) and `just finished:` (what crossed into terminal during this call's
own `--wait`). `detach.launch` records the roster line the same way, so a
`--detach`ed fan-out names its models on the first result line a supervising host can
read rather than a poll later.

Opaque strings rather than named fields on purpose — the consumer's rule stays pure identity, "show
every supplied line I have not shown", and cannot grow an opinion about which line is which — and a
sibling rather than a key inside `progress`, which stays what the status file wrote. It
exists for the OpenCode **package** route, the one route with no agent in its poll loop
(`src/index.mjs`'s `execute` supervises the run itself and the model makes one
blocking tool call), and that file composes no sentence of its own: the wording is composed in
Python by the renderers that already own it and relayed verbatim. Empty entries are dropped, so an
invocation with nothing to show grows no key.

`answer_origins` (`{answered, file, stdout, fallback, accepted, unrecorded, kept, warn_all_fallback}`) is ask's: present on every ask run that reaches its
closing report (all zeros and `false` on a zero-answer run) and on no other mode,
written to the question dir's `.status.json` in the same `update_status` call as
`state`, carried by `detach`'s run view and re-recorded for a pinned
`status --view run` — the route `validation_counts` takes.

Its integers are the `answered: …` counts line's, read from each answer's
`origin_<model>.txt` record, and `warn_all_fallback` is the `[WARN]`'s own
predicate (every file-capable answer was recovered from stdout), so a host that reads only the
result line sees the condition; it never moves `state` or the exit
(plans/ask-answer-recovery-and-false-success.md D7.2).

**`user_next_step`** is the workflow's next phase as the command the *user* types
(plans/next-phase-command.md): `{"phase", "command", "optional_before": [{"phase", "command", "why"}], "complete": false}` (`optional_before` omitted when empty,
and every rendering puts the primary first; since fix round 16 it also carries
`manual_tasks`/`manual_tasks_note`, as below, whenever implement deferred tasks the user
still owes — Codex racing run 6's 06r/07 named T009 from the notice but never its closing command —
and `modes.TOOL_USER_NEXT_STEP_LINE` and every route's relay rule say to give them on any result; every block
renders them, after a command block's commands too — fix round 17), or `{"phase": null, "command": null, "complete": true}` once
no required phase is left (every answer ends with `markdown` since fix round 22: the
step as the report's closing section, each command fenced alone, which the tool and file routes copy
as it is — `next_phase.render_next_step_markdown`; `manual_tasks_note` is then the sentence leading into
`manual_tasks_command`, printed natively as `[MANUAL_TASK_COMMAND]`, which the Stop hook holds the
report to) (plus `manual_tasks: [{id, title}]` when implement deferred tasks the user still owes — a task
skipped because it depends on one also carries `waits_on`, that manual task, and its
`[MANUAL_TASK]` line and `status`'s trailer say implement runs it once that one
is recorded done (`next_phase.waits_on_note`, `StateManager.deferral_root`, fix round 17) — and beside it
`manual_tasks_note`, the one line saying how to record them done — `--implement <plan> --resume --human-task <id>=done` in the
host's spelling, printed as `[MANUAL_TASK_NOTE]` and on `status`'s manual-tasks
line, fix round 14; such a `done` completes a manual task under every strategy, and
a resume that dispatches nothing keeps implement's completion date on both legs — read from
`state.json`'s `skipped` statuses and the `task_titles` implement
records, never from tasks.md, scoped to the decomposition implement ran last
(`implement_task_ids`) and to deferrals — a pause-and-ask "Skip" and its dependents are not owed;
additive, 2026-10-05; the native leg's emit writes those statuses itself through the executor's own
`write_batch_scheduler.persist_human_task_plan`, and a manual task of a completed implement with no answer on record at all
is owed too, via `implement_human_task_ids` — fix round 7) — a complete workflow is an answer, so the
key is present then and **absent always means "no answer"** (not a workflow phase, the gate found
the phase unfinished, a failed run, or an excluded invocation), never "complete".

One builder, `utils/next_phase.next_phase_commands`, makes it from `get_workflow_status`'s `phases`
map and the host's `next_command` template; the command is the mode flag, the plan path
exactly as its source supplied it and, only when that phase is recorded `in_progress`,
`--resume` — never `--models`/`--quick`/`--yes`.

`multi_llm.main` sets it for a workflow-phase mode on exit `0` or
`2` (a re-run of a finished phase is the caller who most needs it) and for
`full` on exit `0`, only when the finished phase is on record
(completed, skipped, or `in_progress`: exit 0 at a `[…_PENDING]` marker or the
native first call of an apply phase is not a finished phase), never beside a non-null
`next` (which is what excludes a real `--detach` launch or
`--if-not-running` re-attach: `detach.launch` records the run's poll command as
`next`), never on `1`/`3`, never for a
write-phase invocation without `--exec-subagents` (it hands its batch loop to the invoker and
records the phase `in_progress` at emit time — `StateManager.mark_phase_started`, since 2026-10-05;
the builder would answer `--resume` for the phase being run), except the apply phases'
closing `--mark-completed` call, which records the phase completed and carries the key like any
finished phase (2026-10-05); implement's closing step is `util finalize-tracking --mark-completed`, a util that carries
none, so that leg's answer comes from the report step's `util next-command` (2026-10-01: there is
no `--detach` test, so a `--detach` invocation that launches nothing — the
completed-phase guard's exit 2, which returns before `maybe_detach` and which the tool
routes hit on every re-run because they always append `--detach` — carries the key like
any foreground exit).

A builder failure omits the key and the block, prints one stderr line and never moves the exit code.
Deliberately **not** `next`, for the reason `consolidation_recommended` is not: a tool
route auto-follows `next`, and following this would start a paid phase nobody asked
for. Beside the key, the emitter prints the `[NEXT_COMMAND]` block (see
[runtime](runtime.md#detach-boundary-and-worker-lifecycle)) above the result line — on stderr when the invocation asked for JSON output
(`--output-format json`, `status --json`).

## Mode registry

**`modes.py` is the one owner of the mode table** — mode words, the closed alias table
(`review`→`review-plan`, `code-review`→`review-code`,
`tasks`→`generate-tasks`; no prefix or fuzzy matching), positionals, lazy
dispatch targets, phase names, core instruction files, and which modes are user-facing.
`SKILL.md`'s `argument-hint`, `multi_llm.py --help` and `check_workflow_prerequisites.py`'s
`--mode`/`--phase` choices are all **rendered** from it; adding a mode is
a row, and `tests/test_instruction_portability.py` fails if the dispatcher and `argument-hint` disagree. Note
the one place mode word ≠ phase name: mode `review-code`, phase `code-review`.

**`sequence` is the composite column** (2026-10-09): the mode words a mode runs, in
order — `full`'s eight (a literal, which `tests/test_mode_sequence.py` holds to the
`is_workflow_phase` rows and to `FullWorkflow.steps()`), `improve-plan`'s and
`improve-code`'s two, empty elsewhere — and no reader uses the words raw:
`is_composite`, `sequence_phases` (phase words, `review-code` →
`code-review`; a word that resolves to none raises) and `last_phase`;
`utils/detach.py` maps each phase through `output_handler.PHASE_DIRECTORIES` for its dir
(`apply-code-fixes` → `apply-fixes/`), and `packaging/common/surface._writes_worktree` gives a composite
`writes_worktree` when **any** of its phases writes.

`runs_workflow` now means only "the whole workflow" (`full`), read where that
is the point (the `suggested_next` fallback of a finished `full`, which has no
single next phase); a reader meaning "a mode that runs other modes" reads `is_composite`.

## Shared CLI parser

**`utils/cli_common.py`** owns the cross-cutting argparse **parent** (`--host`,
`--yes`/`-y`/`--non-interactive`,
`--interactive`/`-i`, `--detach`, `--if-not-running`,
`--max-detached N`, `--exec-subagents`, `--skip-prerequisites`, plus the suppressed
`--internal-worker`/`--internal-run-id` re-entry flags), included by every
`*_orchestrator.py` parser plus `check_workflow_prerequisites.py` and `init_config.py`, so a direct
invocation accepts the same flags with the same semantics.

It also owns skill-dir resolution — `os.path.abspath`, **symlinks preserved, never
`realpath`** (`link-skill.sh` symlinks the plugin cache at a dev checkout;
resolving would move `uv run --project` off the `pyproject.toml` the host loaded) — plus
the printed-command builders, the `MULTI_LLM_RESULT` emitter and the per-invocation record (mode
word + normalized argv) a detached child re-executes. It also owns `--dry-run-subagents` (D3l:
render every command, start no child). `--exec-subagents` is implemented for the read/analyze
phases **and** the four write phases (see [write scheduling](write-scheduler.md)); `--detach` covers
every long-running mode (see [runtime](runtime.md#detach-boundary-and-worker-lifecycle)).

## Repository-root path resolution

**A command typed in a subdirectory of the repository runs from its root** (fix round 26, live run
11: Claude Code ran `generate-tasks plans/racing-game.md --reaggregate` from inside the phase folder and got
`Plan file not found`).

Before a mode's target or a `util` command runs, `multi_llm._run_from_repository_root` looks at its
relative input paths (a mode's `--plan-file`, `--question-file`,
`--tasks-file`, `--files-from`; a util's `--plan-file`,
`--state-file`, `--phase-dir`, `--output-dir`, `--tasks-file`,
`--output-file`, `--project-root` and the path positional of `next-command`,
`check-tasks` and `backup`), and the paths it creates (review-plan's
`--out-dir`, implement's and `util write-prompt`'s `--output`, fix round
27: one need not exist yet, but it keeps the command where it is when it or the first folder it
names exists from the cwd, and outputs alone never move it): when none of them names anything from
the cwd and each names something inside the root `git_utils.pin_repo_root` pins for the cwd (never
through a `..` or a link out of it, never while `GIT_DIR` and the
like redirect git), the whole command runs from that root, the paths still spelled as typed, with
one `multi-llm: running from the repository root …` stderr line; `main` moves back to the caller's cwd on its
way out.

A path that names something from the cwd, a mixed spelling, or one missing from the root too runs as
typed. Both tool servers name such a plan by the root's file for their polls and abort once the
launch's result line says the command ran there (`ran_from_root`, set only when it moved;
`plan_from_root` / `planFromRoot`, fix round 27: their own walk to the nearest
`.git` named the root's plan where the dispatcher refused to move, under
`GIT_DIR`, a planted `core.worktree` or a planted nested
`.git`), while the launch still carries the plan as typed.

## Utility commands

**`util` namespace, not a user-facing surface**: `multi_llm.py util metrics|prior-changes|report|decisions|update-tasks|prereq|finalize-tracking|backup|restore-batch|runs|report-check|next-command|check-reply|check-tasks|write-prompt|batch-done|improve-steps|instructions …` routes the
utility scripts the same in-process `main(argv)` way, flags forwarded verbatim. Excluded
from `argument-hint` and from `--help`'s mode list, documented only under
`multi_llm.py util --help`, and — like the reserved hidden `--internal-*` prefix — never
rendered into a `MULTI_LLM_RESULT.next` value. `util instructions <mode>` prints a mode's core
instruction file with every skill-dir placeholder replaced by the running skill dir, so the text a
host agent reads carries no unexpanded token.

`util finalize-tracking` also inserts the implementation summary's back-reference into the plan when
`--plan-file`/`--summary-file-relative` are passed (it replaced an inline
`python -c` in `instructions/implement.md`, which was a second skill-dir substitution site and
a block of code for an agent to retype). `util next-command <plan> [--after <mode>] [--host <id>]` is the Claude Code report step's one
lookup (plans/next-phase-command.md §2.2(3)): on that host the last dispatcher call of a phase is
often a util step that cannot know which phase finished, so the report runs this read-only,
sub-second call and copies its `[NEXT_COMMAND]` block.

It prints the block and sets `user_next_step` on its own result line; `--after`
names the mode the user ran and only sets where the walk starts (the gate stays off, so it answers
from whatever `state.json` holds — which is why the instructions run it after the phase's
completion write, and never while a `next` is owed); without `--after`
it answers plain `suggested_next`. The plan positional is resolved only to find
`state.json` and printed as received. A failed lookup is exit 3 with one stderr line and
no block, and the report then prints no command.

When it answers implement `--resume`, this lookup alone also prints
`[NEXT_COMMAND_ANSWERS_WHY]`/`[NEXT_COMMAND_ANSWERS]` (`user_next_step.answers`, `next_phase.implement_answers`): one
resume carrying `=done` for each check that holds tasks and `=skip`
for each failed task, the native stop's combined answer (fix round 14), then each answer alone, the
same list and whys the tool routes' `resume_answers` give (`write_batch_scheduler.native_stop_answers` over
`compose_stop_answers`, the executor stop's own composer, fix round 20; fix round 21: a failed task
`util batch-done --partial` recorded half-applied, `HALF_APPLIED_PREFIX`, gets none, and one recorded
`said applied, no file changed` gets `skip` before `done`, as the executor
answers a partial application and an unlanded reply); `util report-check` holds the report to
each like the other commands.

Above them it prints `[TESTS_NOT_RUN]` when no `driver.verify_command` is set and a code task is
recorded completed (`write_batch_scheduler.native_stop_unverified_line`, the executor stop's line, which the native leg never
printed — fix round 18); the report gives its text, and the Stop hook does not check it.

Fix round 24: below the block it prints the answer's `markdown` between
`[NEXT_STEP_MARKDOWN]` and `[NEXT_STEP_MARKDOWN_END]` (a stopped implement's under
`resume_lead`), which SKILL.md says to copy as it is and in which `report_check.find_blocks`
reads no marker line; `--blocked` carries the same section as `user_resume.markdown`
(`render_resume_markdown`), which both servers lift to `resume_markdown` with
`needs_human_message.resume_markdown` in place of the resume and answers clauses; a text line of it whose first
character could open a block is escaped (`markdown_text`).

Since fix round 22 it also prints `[NEXT_COMMAND_WHY]` above that resume (`user_next_step.why`,
`write_batch_scheduler.native_resume_why`): what the plain resume does on an unchanged tree, from the stop's facts
(`compose_resume_why`), which the executor stop records as `resume_why`,
`--blocked` hands the tool routes (`user_resume.why` → `resume_why` and
the `needs_human_message.resume_why` clause) and `status` prints under `resume with:`.

Fix round 23: it ends with what the resume keeps (`keeps the N tasks already done` since fix round 24, a kept
check task having applied nothing, and the half-applied clause status used to append), so status and
the tool routes say it word for word; it is cut at a word to `next_phase.MAX_RECORDED_WHY_CHARS`, as each
answer's why is; a check is said to stay unrun only when the resume keeps it (a verdict
`tree_fingerprint` equal to the tree the stop leaves, a `task_check_trees` stamp or
`executor_check_stamp`, or `not_rerun`), else it runs again; every task id
`rerunnable_checks`/`decomposition_graph`/`held_back_by_checks` read off `state.json`
is `T###` (`_IMPLEMENT_TASK_ID` ends in `\Z`), every why and
note the block and `markdown` print is one `plain_line`, and a stop's recorded
answers are relayed only in the shapes the plugin composes (`next_phase.recorded_answer_flags`,
`recorded_why`).

Every command answer (not only `--after implement`'s, fix round 17: `next_phase_commands`'
`command_answer` is the one writer) also prints the deferred manual tasks
(`[MANUAL_TASK]`) and `[MANUAL_TASK_NOTE]` below the commands, carried as
`user_next_step.manual_tasks`/`manual_tasks_note`, as the native emit prints the note under its deferred
list (the task JSON's `manual_tasks_note`) and both legs' summary.md end `## Manual Tasks (Deferred)`
with it; the note's command answers only the manual tasks themselves (`manual_close_ids`,
`StateManager.manual_tasks_to_close`), never a code task skipped with one, which implement refuses (fix round 16).
`util report-check` is the Claude Code plugin's `Stop` hook check (see
[host result delivery](host-reporting.md#claude-code-headless-transcripts)).

`util check-reply --tasks-file <phase>/validation_tasks.json [--task-id ID]` is what a native host runs right after writing each validation batch's reply
(parses, status vocabulary, covers exactly the task's `group_ids`; `[REPLY_INVALID]`
= re-ask the subagent once, never hand-repair — the 2026-10-01 Claude Code run lost a closing
`]` on 3 of 14 Sonnet batches; since fix round 24 the validation reply lists
`metadata` first and `groups` last, ending `}]}`, because
every lost `]` was the one before `"metadata"`; since fix round 26 it
also says to check each `group_hash` before printing and to print the object once, and a
reply holding several objects is named as such and still re-asked, never cut down to one — live run
11's draft had two hashes swapped, then a corrected second object; fix round 27: the count looks
past a `{` in the prose between them).

`util check-tasks <tasks.md>` is the advisory graph check apply-task-suggestions runs after its last batch
(`validate_generated_tasks` plus a task listed before a dependency), on the graph `implement`
parses: both read dependencies through `task_decomposer.declared_dependencies` alone — a `**Dependencies**:` /
`**Depends on**:` / `- Depends on:` line, never the narrative — and on that line only
the ids before the first `(`, `—`, `;` or
`:` (`T001 (see Step 2)` declared `T2`), a leading
`None`/`N/A` declares none whatever follows, `T001-T003`
is the range, an empty label reads the id-only bullets under it, and a bare number is an id only in
a Step-numbered document (adversarial review, 2026-10-02); the executor leg prints the same findings
as `[CONSISTENCY_WARNING]` and in `summary.md`, and (fix round 14) hands them, with every
item it left unapplied and why, to the result as `progress.notes` (`ApplyOrchestratorBase.run_notes`):
the run's own `notice`, `status`'s `note:` lines and
`progress_lines`, and a terminal poll's `notice`, so a host relays them beside
the counts.

`util write-prompt --phase-dir <phase> --task-id <id>` (2026-10-05) is the native write leg's per-batch prompt: the four write
phases' emit path now materializes the executor's own `{phase}/write_tasks.json`
(`[WRITE_TASKS_FILE]`), and this binds one task's prompt through the scheduler's two module-level
seams — `write_batch_scheduler.bind_prior_changes` (late, from the **plan dir's** `prior_changes.jsonl` as it is at
that moment, phase-scoped) and `with_result_contract` — writes it to `{phase}/prompts/<id>.prompt.txt` and
prints `[WRITE_PROMPT] <path>` / `[WRITE_PROMPT_LINES] <n>` / `[WRITE_PROMPT_CONTEXT] …`
(`--stdout` prints the text alone); `tests/test_write_prompt.py` pins it byte-identical to
`WriteBatchScheduler`'s dispatch text for the same state.

The executor leg keeps the same file (fix round 16, run 6 #10b: T007's prompt with the accepted
findings was in no artifact on either tool route): `WriteBatchScheduler._keep_dispatched_prompt` writes each dispatch's
prompt to `{phase}/prompts/<id>.prompt.txt` (`dispatched_prompt_path`) just before the child starts, overwritten
by a later dispatch as a `--resume` rebinds the native one, and an attempt that sends
something else (a report re-ask, rule 9's re-send) to `<id>.attempt-<n>.prompt.txt`, which the task's next
first attempt clears; never for a task refused before dispatch, unredacted like the native file
(plan and task text, prior-changes summaries, accepted findings, all already on disk), inside the
phase dir the sweep never attributes to a runner, and checkable against the audit line's
`prompt_sha256`.

Both legs write these files without following a link (fix round 17: a jailed repo-wide batch, or a
native host's subagent, can replace the phase dir or `prompts` with a link out of the
repository): `utils/prompt_files.py` reaches the phase dir from the repository root one
`O_DIRECTORY|O_NOFOLLOW` component at a time, the scheduler pins its identity before any batch runs
and `util write-prompt` at the start of the call, and every prompt and `.tree.json`
record is created, renamed into place and unlinked relative to the folder's descriptor, so a link at
a file's name is replaced, never written through.

An earlier batch the log does not hold yet — recorded by `util batch-done` but not appended, or
not finished — draws one `[WRITE_PROMPT_ORDER]` stderr line naming it (advisory, the prompt is still
written, so a `--resume` is never refused; fix round 7). The host hands the subagent
only "read that file and follow it" and reads the closing JSON object's `summary` for
the prior-changes append, whose `--output-dir` is now the plan dir on every native
instruction (it used to be the phase dir, a second log the executor never read). Measured
2026-10-04: with the `{prior_changes_context}` token left to the host, every Task subagent spent 3-6
calls guessing the JSON shape, and two skipped the context or read `tasks.md` instead.

**`util batch-done --phase-dir <phase> --task-id <id> --status completed|failed` is that leg's per-batch progress record** (fix round 6, 2026-10-05): run
right after the host verified a batch and before its prior-changes append, it writes the same
`task_status[<id>]` the scheduler writes plus, for the apply kinds, `processed_<phase dir word>` for
every `expected_ids` member (the scheduler writes that too since fix round 22) (a raw
read-modify-write of `state.json` through `state_manager`'s module-level helpers,
never a `StateManager` over the plan the batch just edited, held under
`metrics.state_file_lock` — `<state dir>/.state.lock`, which `util metrics record|start|finish` and
`util finalize-tracking` hold too, so parallel same-batch records no longer drop each other's
`task_status`; a lock not free in 120 s is exit 1 with nothing written, fix round 8); the
apply base's `--resume` (step 23) drops a group recorded there unless
`failed`, implement's drops a `completed` task, and both native emit paths
refuse a `--resume` (exit 3, `[RESUME_REFUSED]`, `batch_done.unrecorded_logged_batches`) that would
re-emit a batch the phase's prior-changes log already names and no record says finished — before it,
a native `--resume` printed `[resume] 0 already processed` and re-emitted every batch under the
same `prior_changes_id` (reviewer repro).

**`state.json` exists from a phase's first run** (2026-10-05): the review orchestrators'
completed-phase guard saves the fresh `StateManager` at pre-flight (`review_orchestrator_base._ensure_state_file`,
and `code_review_orchestrator`'s guard), because the instructions run `util metrics start --state-file {plan}/state.json` between the
fan-out and the validation subagents and nothing wrote the file before the phase completed — on a
plan's first phase it failed "state file not found" and the host shrugged it off (racing re-run,
2026-10-04).

## Import safety

**Import must be side-effect free.** Every dispatch target keeps argparse and all work inside
`main(argv)`, with `if __name__ == "__main__": sys.exit(main(sys.argv[1:]))` — no module-level argparse, no module-level
`sys.argv` read, no I/O at import. Two targets can run in one interpreter, and
`tests/test_instruction_portability.py` enforces it (AST guard + a one-process import probe). Direct invocation of
`*_orchestrator.py` stays **supported but unadvertised** (the test suite and
`--internal-*` re-entry rely on it); the instruction core and shims name only
`multi_llm.py`.

## uv project selection

Use `--project` not `--directory` with `uv run` (path duplication
bug)

## Shared flags

Never redefine a cross-cutting flag locally: `--host`,
`--yes`/`-y`/`--non-interactive`,
`--interactive`/`-i`, `--detach`, `--exec-subagents`,
`--skip-prerequisites` come from `cli_common.build_parent_parser()`. Adding one to an orchestrator parser that
already includes the parent raises `argparse.ArgumentError` at import of the parser, not at first use.

## Mode and help updates

A new mode is a row in `modes.py`, never an edit in twelve places — but the
`argument-hint` **trailer** (everything after the mode alternation) is a literal, so a mode
that introduces a new *flag* still needs that constant updated. Regenerate `tests/golden/*_help.txt`
whenever the shared parent's help text changes.
