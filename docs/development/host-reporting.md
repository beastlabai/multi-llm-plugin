# Host result delivery

Read when changing tool-call supervision, stop hooks, result relay, or native apply counts.

Back to the [development guide](../../skills/multi-llm/AGENTS.md). These notes retain detailed contracts and regression history from that guide. Paths beginning with `docs/`, `packaging/`, or `plans/` are repository-relative; other implementation paths are relative to `skills/multi-llm/`. Dated incidents explain the rules; verify version-specific claims against current code.

## Contents

- [Interrupted host calls](#interrupted-host-calls)
- [OpenCode stop visibility](#opencode-stop-visibility)
- [Claude Code headless transcripts](#claude-code-headless-transcripts)
- [Claude Code report checks](#claude-code-report-checks)
- [Single delivery of result text](#single-delivery-of-result-text)
- [Stop headlines and notices](#stop-headlines-and-notices)
- [Native apply counts across resumes](#native-apply-counts-across-resumes)

## Interrupted host calls

**A host that leaves mid-call stops the run cleanly, and the next `--implement` picks it
up** (fix round 16, Codex racing run 6): under codex 0.160.1's code mode the model calls this tool
from a JavaScript `exec` cell that yields back to it after `yield_time_ms`
(default 30 s) with the call still in flight, and nothing stops it ending its turn there;
`codex exec` then closes the server's stdin and SIGTERMs it in the same millisecond (no
`notifications/cancelled`; `notifications/progress` never reaches the model either) and kills it about
0.5 s later (measured offline against a stub provider).

The server cannot tell that from a user who quit codex, so a claimed run is aborted as on any
cancellation, never left running with nobody to receive its result (the budget handoff differs:
there the host is still listening for the re-attach `next`); the abort is recorded
`aborted`, not a crash (`install_worker_signal_handlers` ignores every signal after the first,
including the copy `uv run` forwards to its child, and the handler prints through
`stream_bootstrap.streams_for_a_signal_handler`'s lock-free writers), the finished tasks stay recorded,
`status` and `util next-command --blocked implement` name `--implement <plan> --resume`, and a plain
`--implement` on an implement that started and did not finish continues it as
`--resume` with one `resume: implement started earlier …` line (an explicit native
`--resume` prints `resume: kept N task(s) an earlier run finished (…)`, naming an accepted check as
`T002 (accepted with its check findings)`, as the executor's own line does — `implement_orchestrator._native_kept_line`, fix round 18)
(`--fresh` starts over; a finished implement, `--task`, and records older
than a regenerated task list start over as before).

Fix round 17 (review of round 16): that holds in every cycle, not only the first, because both legs
record implement started — a start-over in `implement_orchestrator._start_over`, a resume that hands a task to a
runner in `WriteBatchScheduler._mark_implement_started` — and a start-over sets aside what earlier runs recorded for each
code task (`pending` again, its check records dropped, its result file kept as
`<name>.rerun-<stamp>`, its pre-batch snapshot set aside, `prompts/*.prompt.txt` removed fd-relative
through no link, `.tree.json` kept), so a later `--resume` never keeps a task
on an earlier cycle's evidence; it stamps `implement_task_reset_at` (a `--task` rerun
stamps its task too), which `batch_done.unrecorded_logged_batches` reads so a native `--resume` does not
refuse a task on the log line of the run before.

A native `--task` rerun the host recorded but never closed leaves implement started: a
plain run continues it, hands out nothing when every task is done and only closes the phase.

To make the leave rarer, the codex tool description and plugin skill tell the model to start its
cell with `// @exec: {"yield_time_ms": <the tool budget in ms>}` and to call `wait` rather than end its turn
(`modes.TOOL_LONG_CALL_RULE`, filled by `surface.long_call_rule` from `packaging/codex/mcp.json`'s
`tool_timeout_sec`; that pragma held a 70 s call to its result offline, and in fix round 17 an
870 s one: codex 0.160.1 forwards the pragma's value to its code-mode host unclamped and sends it no
`cellExecutionLimits`, and `features.code_mode.default_exec_yield_time_ms` sets only the 30 s default, never a cap; a
`wait` with its own `yield_time_ms` held 369 s the same way); whether models
follow it is for a live run to show.

## OpenCode stop visibility

**An OpenCode stop is on the tool's own line, which shows without the model** (fix round 16,
OpenCode racing run 6: the free model never wrote its reply after implement's needs-human result, so
the stop and its resume command reached nobody): the package's `tool.execute.after` hook titles a
finished stop (a failed or needs-human result carrying `resume_command` or
`no_command_reason`) with the surface's `stop_title` (`modes.STOP_TITLE`, additive
within schema v5), the run's own last error line and `resume_command`; `opencode run`
prints a plugin tool's title on the tool line the moment the call returns, `--format json`
carries it as `state.title`, (measured offline on 1.18.34, stub provider, a model that
never answers; the TUI's tool header reads the same title, not measured).

`execute` still returns one JSON string. Since fix round 17 the title shows no control
character (`plainTitleText`; the result line's `error` is already a
`cli_common.plain_line` at the source, since the stop headline carries a task title from tasks.md and
the runner's summary, and the codex tool result and `status` show it too) and cuts the
error line at `STOP_TITLE["max_line"]` characters, never the resume command after it. A batch killed
at its time limit after its edits landed (`TaskOutcome.ran_out_of_time`) says on that line that its edits
are kept and that a resume dispatches it again on top of them.

## Claude Code headless transcripts

**The Claude Code plugin's hooks read a headless run whole, and no phase schedules its own way
back** (fix round 26, live run 11 #2/#3). Claude Code 2.1.293 runs a phase's subagents in the
background, so a headless phase spans several turns, each opened by a `task-notification` entry
(`origin.kind` not `human`, not `isMeta`) or a
`ScheduleWakeup` prompt (`isMeta`).

`util report-check` now reads everything since the prompt the user typed
(`report_check.TYPED_ORIGIN_KIND`), takes the reply after the run's last user entry of any kind as the final
message, and holds it to every command the `[NEXT_STEP_MARKDOWN]` section fences beside the marker
lines' (`Block.section`), its reason asking for the section printed as given: racing 04 ended
on a wakeup's reply that carried none of them.

A `PreToolUse` hook (`hooks/wakeup-check.sh`, matcher `ScheduleWakeup|CronCreate`) runs
`util wakeup-check` (`utils/wakeup_check.py`), which denies both tools (`permissionDecision: deny`)
while the same transcript shows a phase that has not printed its next step: a Bash call ran the
dispatcher with a `DETACH_MODES` mode and no later tool result printed the next-step block
(or that call's own result exited `1`/`3`, or
`ask`'s answered, or, since fix round 27, the `task-notification` naming a
background call says it failed or was stopped; a subagent's report quoting the block, an
`Agent`/`Task`/`SendMessage` result, ends nothing, and the
Stop check holds the report to no block one quotes, `report_check.SUBAGENT_TOOLS`), read once the call it
decides, and the results of the phase calls its message made before it, are on file
(`wakeup_check.call_on_file`, fix round 27).

Racing 03's wakeup re-ran `--generate-tasks` and kept `claude -p` alive 16 minutes.
Every error allows, a session that never ran the dispatcher pays the wait for its own call to be on
file and one grep, a prompt the user typed ends the phase's hold, and `state.json` is not
read (only the transcript says which phase this session started). SKILL.md tells the host the same,
and to send a batch back to its subagent (`SendMessage`) rather than edit its files.

## Claude Code report checks

**A Claude Code host's copies and checks are read as they were meant** (fix round 28, live run 12,
Claude Code racing). A Stop-hook block with a ready section says once how the report ends: the whole
report again, ending with that section in SKILL.md's words (`report_check.SECTION_REQUEST`, its
`READY_SECTION_RULE` held to `modes.READY_SECTION_RULE`), the commands it holds not listed a second
time, and no reason names a `### Next Steps` section (03's asked for both, and the resent
report added the heading).

A task-generation or judge reply whose only schema error is a left-out `schema_version` is
read as the one version its schema's `const` allows, with one plain line
(`schema_validator.supply_schema_version`, through `subagent_executor.supply_result_version` on the executor's parse lane and in
`task_is_complete`, in generate-tasks' consume, which writes the key into the file, and in the
verdict ingest); a present key keeps its value for the schema to judge (03's host dropped it copying
the subagent's JSON).

`util check-reply` prints `[REPLY_MISSING]` for a reply not on disk yet, no re-ask and exit
0 on its own, and the runbooks check the whole file only once every batch's subagent returned (01
checked while wave 2 ran). apply-code-fixes' step 7 hands the report the `[TESTS_NOT_RUN]` line
first and whole (08 paraphrased it).

## Single delivery of result text

**A result hands the host each sentence once** (fix round 26, live run 11: hosts pasted the notice
above the ready section, wrapped the section in a fence of their own, printed "Workflow complete."
twice and listed a stop's options themselves). A terminal poll that carries `user_next_step`
leaves out of its `notice` each workflow line the step's `markdown` says
(`run_status._workflow_trailer_lines(carried=…)`: an optional entry's `why`, the manual-tasks line), and so
does its text when a tool route reads it (`_read_by_a_tool_route`; a shell's text keeps every line);
a non-zero terminal poll carries no `suggested next` line, and a stop's notice drops a part
its `error` says word for word.

Both servers move `user_next_step` out of `result` rather than copy it, and drop
from that result's `output` the block lines the step carries (fix round 27: a line
that is, or starts with, one of the surface's `dispatcher.next_step_markers`, `next_phase.CARRIED_BLOCK_MARKERS`;
`[RUNNER_WARNING]` stays, and a shell's text and the Claude Code leg keep every marker).
`modes.READY_SECTION_RULE` (copy word for word, in no fence of your own, say none of it twice) and
`modes.NOTICE_PLACEMENT` are on every relay surface, and a stop's `resume_markdown` is the
`options` report, in place of any list of the host's own.

The manual-task note and lead read in the singular over one task (`next_phase.manual_close_lead(count)`), and a
single-target apply phase names its file: the mirror's `applied_to`
(`… items applied to <file>`), the notice's left-out line and the native `--mark-completed` message.

## Stop headlines and notices

**A stop's line beside its section names no way forward the section carries, and every notice join
is one rule** (fix round 28, live run 12 #5/#7/#10/#11).

The executor records beside a stop's `headline` the same line without its
`--human-task` answers and bare resume (`stop.headline_beside_section`, `WriteBatchScheduler._set_stop_headline`; only
when every answer it leaves out is one of the stop's `answers` or its own
`resume_flags`, `answers_cover`; a manual-gate stop's is what stopped and why, handed
back only while the section's resume is those flags, and the native `[MANUAL_GATE]` line says
the same); `util next-command --blocked` hands both as `user_resume.error_line`/`error_beside_section`
(`next_phase.stop_headlines_on_record`: one plain line each, every recorded answer valid), and both servers, only
when they lift `resume_markdown` and while the result's `error` is that
`error_line` word for word, put the second in `error` and
`result.error` and compose `message` around it (`relay_stop_beside_section` /
`relayStopBesideSection`); the shell's text, `status`, the file routes and Claude Code's
exit 3 keep the full line, and the native report step leaves a held check's answers to the section.

Every notice join is `text_clip.join_clauses` (a clause another follows gives up one closing
`.`/`;`, the last keeps its mark). A finished Codex result's
top-level `progress_lines` is its terminal child's own (`settle_progress_lines`), never the
last relayed snapshot. A re-run of a completed review phase prints `review_orchestrator_base.already_completed_lines` (done,
nothing ran, `--force`/`--reaggregate`), exit 2 and the same words as
`notice`, never `ERROR:`.

## Native apply counts across resumes

**A native apply view counts its run, not its file** (fix round 29, review of round 28: a native
`--resume` re-materializes `write_tasks.json` with only the batches still to run, so
the count vanished mid-resume and then read `1/1 items applied` for a phase that applied 3 of 4,
and a bare re-run of a completed phase showed the earlier run's `4/4` before it
recorded anything).

The native apply emit writes `native_run` beside the file's `leg: native` stamp
(`write_phase_bridge.native_run_record`: `started_at`, every item the run handed out,
`emitted_at`); a `--resume` of the review the previous native file applied
(`inputs.grouped_sha256`) carries `started_at` and adds its items, a `--resume`
after any other file records `started_at: null` (every record of the review counts, as before),
and any other emit starts a run.

`run_status._native_apply_not_applied` counts the run's items and the `processed_<phase>` records written since
`started_at`; a batch recorded `failed` since `emitted_at` is
named after the count with the reason its record gave (`failed_batch_clause`:
`1 batch failed: apply-batch-0 (<reason>)`, `--json`'s `failed_batches`), never a bare
`0/4 items applied`.
