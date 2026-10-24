# Status views and progress

Read when changing workflow or run status, progress counts, follow-up commands, ETA, or bounded output lines.

Back to the [development guide](../../skills/multi-llm/AGENTS.md). These notes retain detailed contracts and regression history from that guide. Paths beginning with `docs/`, `packaging/`, or `plans/` are repository-relative; other implementation paths are relative to `skills/multi-llm/`. Dated incidents explain the rules; verify version-specific claims against current code.

## Contents

- [Status views and run selection](#status-views-and-run-selection)
- [Printed follow-up commands](#printed-follow-up-commands)
- [ETA outcomes and denominators](#eta-outcomes-and-denominators)
- [Status parity across execution paths](#status-parity-across-execution-paths)
- [Clipping model text](#clipping-model-text)

## Status views and run selection

**`status <plan>` is a real command (`utils/run_status.py`), not the retired
instruction-only `--status` mode**: one word, two named views over two files, never
merged. `--view workflow` reads `state.json` through `check_workflow_prerequisites.get_workflow_status` and is
**byte-identical** to `check_workflow_prerequisites.py --status` (which survives unchanged as that view's
implementation); `--view run` reads `{phase}/.status.json`; `--view all`
(default) prints both as labeled sections, two keys under `--json`. Three state fields
— workflow phase, in-phase `state`, `run.state` — print on three lines and
are never collapsed, and a finished run under an unmarked phase is reported as the normal post-run
condition, not a conflict.

The workflow view's `suggested next:` line also names any *optional* phase it stepped over
(`review-tasks` / `apply-task-suggestions`, still `pending`, sitting before the
suggestion in phase order) rather than leaving them invisible — the bare word used to read as
"nothing else is outstanding" (F8, 2026-09-16); the machine-readable `suggested_next` itself
is unchanged.

The run block also prints the `host:` the run was **launched** with, and
`render_runner_text` answers for *that* host rather than the invoking one: it prefers the runner
specs the run's own executor mirror recorded (`progress.subagents` on the read path,
`progress.write_subagents` on the write path — a write phase that never recorded one used to fall
straight through to a fresh resolution, and a *listed* view with an empty mirror is skipped rather
than returned outright, falling back to its host only if no view ever recorded a spec — F6,
2026-09-16).

Among several views that each recorded one, the most recently `updated_at` wins rather
than whichever phase sorts first alphabetically (a `generate-tasks` run's
pre-`--variant low` record used to outrank `review-tasks`' own `:low`
record beneath it), and the printed line names the run the winning spec came from once there is more
than one candidate to name — `_run_runner_facts` (F4, 2026-09-16). Else it resolves for the run's
host with the session rung skipped when the run was launched elsewhere (that rung reads this
process's `MULTI_LLM_HOST_SESSION_MODEL`, which is the reader's session, not the run's).

`detach.status_command_for` carries `--host` into the poll command it emits whenever the
run recorded one, so running the emitted `next` verbatim reports the same run the
same way. Measured 2026-09-16: three polls of an `opencode` run whose eleven batches all
dispatched to `opencode:openrouter/…` each printed `claude-code:sonnet`, and no line on screen
named the run's host. Selectors: `--phase` (mode word *or* phase name, resolved through
`modes.py`), `--question`, `--run-id`, `--all`.

Exit codes follow the run view and **are** the poll-loop condition: `0`
terminal-done, `2` non-terminal (including "no run recorded" — never
`0`), `3` terminal-needs-human, `1`
unreadable. A host prints any non-zero exit as an error, so the two read steps that are not polls —
`util instructions status` and each fan-out core's one non-waiting roster call — say exit
`2` is not a failure (2026-10-05); the code never moves, and no
`|| …` is chained onto the call, which would cost Claude Code's permission allow
rule.

`--exit-zero` (fix round 7) is the opt-in for a call whose loop never reads the code — each
core's roster read, and the poll a background-mechanism host composes (Claude Code's *run detached*
sentence): a view that was read exits `0` and the verdict's word stays on the
result line's `state` (`cli_common.keep_state_word`); `1` and the usage
`3`s are unchanged, and the default never moves.

With several phases listed, the `next` this hands back on exit 2 is built from the
first **non-terminal** view — falling back to the alphabetically-first (`discover_phase_dirs`' sort
order) only when none is live — rather than whichever phase dir happened to sort first (F3,
2026-09-16) — and "live" is non-terminal **and holding a run record**, because a checkpoint with no
`run` (verdict `no_run`: every foreground phase, which on the Claude
Code host is every phase) is non-terminal only in the sense that there is nothing to be terminal
about, and counting it live reported `running` for a plan nothing had launched with
`next` pointed at a phase that never had a run (F1, 2026-09-17); that exit 2's
`MULTI_LLM_RESULT.state` also reads `"running"` rather than `"nothing-to-do"` when a
live run is why the exit code fired, via `cli_common.STATE_RUNNING`/`record_state` (F11) — the
exit code itself never changes, and the genuine "already complete" / "no run recorded" exit-2 cases
keep the old word.

A live view whose recorded pid the identity probe says is **gone** is rendered
`died (pid N gone, last beat …)` rather than `running (stale, …)`, and gets the failure block's
`reason` line and `resume with:` command one heartbeat window early; its
result-line word is `cli_common.STATE_DIED`, taken only when *every* live view is dead (a dead run
beside a live one is still, on the whole, running).

The close-out still waits out `HEARTBEAT_STALE_AFTER_S` and the exit code is still 2 — only the word
and the timing of the guidance change — because a SIGKILLed worker reported as
`running` is a lie the poller cannot act on (F2, 2026-09-16: an `implement`
write subagent's `killall -9 node chrome google-chrome python3` killed the detached orchestrator that had dispatched it, and
the record stayed `state: running, exit_code: null`).

For a TERMINAL run, `--view run` — the only channel the opencode plugin polls — also
carries the workflow view's `suggested next`/`notice` lines into its own
rendered text and into `MULTI_LLM_RESULT`'s new `notice` key (never
`next`, which the plugin auto-follows unasked): before this, that view never fetched
the workflow at all, so the real follow-up for a finished `review-tasks` lived only in
`--view all`'s text and a weak model, reading `next: null` on the only view it
polled, said "nothing further to run" and then invented `--apply-task-suggestions` from general
knowledge (F7, 2026-09-16). A still-running poll carries neither.

**A terminal poll pinned to one run that exits `0` also carries
`user_next_step`** beside F7's `notice` (plans/next-phase-command.md): built for
the run record's `mode`, its `plan_arg` when `next_phase.recorded_plan_spelling`
passes it (else the poll's own absolute plan, never the record's `plan_file`), the poll's
own `--host` and route, from a fresh workflow read taken after any
`--wait`.

Unlike F7 it does not depend on `--view`: it rides the result line and the
`[NEXT_COMMAND]` block prints under `--view run` and `--view all` alike
(on stdout in the text view, on stderr under `--json`, whose payload gains no key
beside `runs[]` or inside `workflow`); `--view workflow` never
carries either and stays byte-identical to `check_workflow_prerequisites --status`. Never on a poll's exit
`2` (still moving, or nothing recorded) or `3`; a workflow
read or builder that raises omits the key and the block with one stderr line and leaves the exit
code and every other key alone.

**`get_workflow_status` knows when a review left nothing to apply**: `apply-code-fixes` is
marked `skipped (no findings)` when `code-review` completed and `_grouped_has_suggestions(code-review/grouped.json)` — the
one predicate `full`'s `step_apply_code_fixes` also calls, moved into
`check_workflow_prerequisites.py` — is false (any non-empty group counts, whatever its verdict; a missing or
unreadable file is false), and `skipped (prerequisite skipped)` when `code-review` was skipped,
without reading the file; `apply-suggestions` is marked `skipped (no valid findings)` when
`review-plan` completed and `check_apply_suggestions_prerequisite`'s new `open_count` (groups
whose status is anything but `invalid`) is 0, so `needs-human-decision` groups keep it
pending.

An apply phase that is neither completed nor skipped but has one of its own write-batch ids
(`apply-batch-N`, `apply-task-batch-N`, `apply-fix-batch-N`, matched whole) recorded
`completed` now reads `in_progress`, as a partly implemented plan does — but
only if the phase's last dated run (`metrics_progress.{phase}.started_at` or the newest `metrics.{phase}`
timestamp) is not older than its review's latest completion, because nothing clears
`task_status` when a review is re-run (2026-10-01; `TestStaleApplyBatchesAfterReviewRerun`) — and only while
that review is `completed`: a review read as pending (stale after implement) takes its
cycle's batches with it (2026-10-02).

Since 2026-10-09 that rule is one predicate, `check_workflow_prerequisites.apply_phase_in_progress(state, phase)` (a started record counts as
progress too), which the post-loop branch calls and the composites read to infer an apply's
`--resume`; its completed-side twin is `StateManager.completion_predates(apply, review)`, true when the newer of
the apply's completion and `skipped_at` stamps is strictly older than the review's
completion (phase words only; a date that does not parse keeps the apply done).

**The same rule one level up** (2026-10-02, `TestReviewStaleAfterImplement`): `StateManager.is_phase_completed` reads a
`code-review` or `apply-code-fixes` completion older than `implement`'s
latest as pending (`phase_completion_is_stale`; the raw `phases_completed` record stays, an
unparseable date keeps the completion), flagged `stale` with both dates on the
workflow entry, which `status`'s run view renders as a plain re-run and never the
`--reaggregate` that would re-stamp it (2026-10-02), so `status`, the next-step
builder and the completed-phase guard all agree, and `code_review_orchestrator` takes its
`--force` path on a stale review instead of auto-healing it back from the artifacts it
left.

**The next-step walk never skips a started phase** (`next_phase_commands`): a required phase before
the finished one that is `in_progress` or has a seeded `tasks_total` (an implement
that ended `needs-human`) is the answer, so `user_next_step` and the result's
`suggested next` notice cannot disagree and a workflow with an unfinished implement is never
`complete`; a phase never begun still stays skipped over by a standalone start.

**An early warning for the refusal that waits at phase 6** (`check_workflow_prerequisites.repo_wide_sandbox_warning`): whenever
`implement` / `apply-code-fixes` would be refused for the runner
`driver.subagent_runner` resolves to on this host — computed from the dispatch's own decision
(`subagent_executor.repo_wide_write_refusal`: `_detect_sandbox` with the `agent-write` profile and a
repo-wide `JailRequest`, the adapter's model-free probes once per process through the
shared probe cache; since 2026-10-02 never a guess from an earlier phase's recorded rung, which
after step 3b would say the opposite) — `implement` and `apply-code-fixes` carry a
`warning` (also in `notices`, so `status` and a terminal
poll's `notice` show it), a next step that points at one prints it as a
`[RUNNER_WARNING]` line above the `[NEXT_COMMAND]` block, and `init`
prints the same line after its smoke-test command (`init_config._runner_warning_lines`), each naming the
refusal's own remedy; it stays silent while the repo-wide rung works and for a runner with no
sandbox probe (nothing spawned).

It resolves config for the reader's `--host` (threaded through `get_workflow_status`;
the servers declare the host nowhere else, so the codex session rung is only reachable that way,
2026-10-02). A missing or unparseable date on either side keeps the mark, and a review re-stamped by
`--reaggregate` makes older batches look stale, so the next command is bare rather than
`--resume`.

`suggested_next` is picked **once, after** those marks (`_suggested_next`: a started,
unfinished required phase first — `in_progress`, a seeded `tasks_total` or
`stale` — else the first open required phase after the furthest completed phase,
never one before it, so a plan whose review-plan never ran is not sent back to it; fix round 14), so
after a clean review-code `--view workflow` names nothing further where it used to name
`apply-code-fixes`; and the unapplied-task-suggestions notice is also recorded on
`phases["apply-task-suggestions"]["notice"]` (the `notices` list is unchanged), which is where the builder
reads it as that optional entry's `why`.

It takes no lock and writes nothing in the user's source tree outside `.multi-llm/` (the
early warning's probes create and remove `.multi-llm/tmp` and a seed file there, under an
interprocess mutex so concurrent polls leave nothing behind; since 2026-10-02 nothing else — the
repo-wide canary used to create a file in the first sorted directory, `.github/`, on
every poll), so it works while another run owns the dir; its one write elsewhere is the deliberate
`failed(heartbeat_timeout)` close-out. `--wait` (default 90 s, under OpenCode's ~120 s cap)
compresses the loop into one turn; hosts must never spell the wait as `sleep <n>; status` in one
shell call.

`--abort` (alias `abort`/`cancel`/`stop <plan>`)
is the **only** kill path — identity probe → process-group SIGTERM → 10 s → SIGKILL →
`run.state: aborted`, idempotent (exit 2 when nothing was running). `--sweep` closes
out orphans under a plan; `util runs [--sweep]` does it across the `{project_root}/.multi-llm/runs/` registry,
which also enforces the `MULTI_LLM_MAX_DETACHED` (default 4) ceiling.

**`doctor` is the one command allowed to spawn provider CLIs outside a phase, and only
their model-free probes** (`utils/doctor.py`, D6 step 5): one line per provider in the resolved
config — present, version against `min_version`, auth through the adapter's own
`auth_check`, sandbox through its `sandbox_probe_command` (codex's `codex sandbox -c 'sandbox_mode="workspace-write"' -- sh -c …`,
~30 ms) or, for any adapter naming bubblewrap, the 2 ms `bwrap --unshare-net` self-test — every
probe concurrent inside one budget (`--budget`, default 1 s; a probe that outlives it is
`unprobed`, never a blown budget), exit 0 or 3, `--warn-only` always 0,
`--json` one object, and when a sandbox row fails the AppArmor profile that grants
`/usr/bin/bwrap` the `userns` right is printed verbatim as the repair (the
`sysctl` alternative named and not recommended; the plugin never installs either).

The BROKEN row also names the `providers.codex.sandbox_fallback` value in force and the fallback rung a second,
read-only-profile probe selected — e.g.

`(fallback: auto → landlock — codex runs read-only phases under a real sandbox; writes: see the agent-write row; measured at codex 0.155.1: `[features].use_legacy_landlock` is deprecated and will be removed soon.)` (the rung's
`deprecated_at`; until 2026-09-30 that clause read "agent-write still refused"),
`(fallback: auto → plugin-landlock — codex runs read-only phases on the plugin's Landlock jail, which stops stray writes, not the model; …)` when the walk took the plugin's own jail (2026-09-30), the unsandboxed
sentence when the walk reached the last rung, a `bypass disabled` refusal under
`contained`, `refused by config` under `off`, and
`rung unprobed: the report budget was exhausted` when the budget ran out first (a rung measured unusable is named too:
`rung landlock is not usable here (<outcome>) — measured at codex …`, which is doctor's stateless stand-in for the run-time retirement notice; a
rung whose probe printed codex 0.156.1's exit-101 panic reads `→ landlock unusable since codex 0.156.1 (panics; measured)` (not "removed":
codex 0.160.0 still lists `use_legacy_landlock` as deprecated, 2026-10-02) beside the read-phase
choice — `sandbox_fallback: contained` refuses read phases like writes, `auto` runs them
under `danger-full-access` — while a timeout or any other probe failure is shown as itself; doctor
always runs the real rung probe, never skipping by `removed_at`, and flags a pass on a
build at or above it as a mismatch, 2026-09-24) — so the operator learns what the next run will do
on this machine; the exit stays 3 and the repair is still printed, because codex's own sandbox
cannot start whatever a rung serves (D6, 2026-09-17; ladder 2026-09-22; until 2026-09-30 the reason
given was that no rung served the write profile at all).

Since 2026-09-30 (plan codex-sandbox-unavailable-findings-and-fixes §6 F1, steps 2–3) two more rows
sit under a BROKEN codex line: `plugin-landlock` — its probe result, its grade in words (stops
stray writes, not the model; never "contained" or "sandboxed"), the codex version its allow-list was
measured through beside the installed one (a newer installed codex flagged), the resolved base
allow-list, and in every state the one switch that skips it alone, `MULTI_LLM_CODEX_SKIP_RUNG=plugin-landlock` (while
that is set the row says the rung is skipped and no probe runs; a bad value is reported on the row,
since `doctor` alone is exempt from the start-up rejection) — and
`agent-write`: the write rung and its probe result, the `plan-file`,
`tasks` and (since 2026-10-02, plan step 3b) `repo-wide` scopes it serves
under `auto`, that a repo-wide batch may write the whole repository
(`.git/`, `.codex/`, `.multi-llm/` and the output folder
included — told to leave them alone, checked afterwards, nothing backed up; root grant 2026-10-05),
the network it leaves open, and F5's clause.

When the jail probes ok under `auto`, the AppArmor/sysctl block is still printed in
full but headed `System-level alternative (needs /etc and sudo; …)` instead of `Repair (…)` (`REPAIR_HEADING_SYSTEM_ALTERNATIVE`);
the verdict and the exit code do not move. It also WARNs when codex's own config has
`[features] use_legacy_landlock` enabled globally, which makes every `workspace-write` dispatch panic. It
takes no lock and writes nothing.

The Claude Code plugin's `hooks/hooks.json` runs it `--warn-only` from a
`SessionStart` hook through `hooks/doctor.sh` (silent exit 0 when uv is missing) —
advisory, never blocking; beside it, a `Stop` hook runs `hooks/report-check.sh`
(plans/next-phase-command.md T11), which exits 0 at once when `stop_hook_active` is set or when
the transcript holds no `[NEXT_COMMAND`/`[WORKFLOW_COMPLETE]` text and no Bash call that ran
`multi_llm.py` (a `grep` before uv is even looked up; the call is enough
since fix round 27, because the block the turn's last call printed may not be on file yet), and
otherwise runs `util report-check`: when the turn's last `[NEXT_COMMAND]` block names a
command that is not a line of a fenced block in the final message byte for byte (or the message
lacks "Workflow complete." after `[WORKFLOW_COMPLETE]`, or a `[MANUAL_TASK]` id listed under
either kind of block, after a command block too since fix round 17 — ids only, as a word anywhere in
the message), it prints one `{"decision": "block", "reason": …}` naming the missing commands and asking for the
whole report again (the reply becomes the run's final message; racing run 5 got back the next-steps
section alone, fix round 14), which gives the model one more turn to copy them; on a native
`--improve-plan`/`--improve-code` run the `[IMPROVE_RERUN]` line of a Bash call
that ran `util improve-steps` is a block too (`report_check.planner_call_ids`, `Block.continuation`), and
while it is the block held a later block with no ready section, an inner phase's standalone next
step, does not replace it, while a `util next-command` lookup's (every one prints a section)
does: before that, review-plan's `--reaggregate` block (`--apply-suggestions <plan>`) blocked every
correct report of a stopped apply and the forced turn handed out a bare apply, which re-emits every
batch on that leg (Fable review 2026-10-09); a run that never ran the planner is read as before, and
a finished composite whose closing lookup failed is held to the continuation (it prints
`[ALREADY_COMPLETE]` and the next step); the "Stop hook error occurred" notification a block draws
is Claude Code's label for a block (that run's hook record: exit 0, empty stderr, JSON stdout).

It is advisory and exits 0 with no output on every error path, as `doctor.sh` does; it
runs only on the plugin route (a plain-skill install loads no plugin hooks), and hooks load from the
plugin root alone: a run that links only `skills/multi-llm` into an older installed cache
(`link-skill.sh`) runs that cache's hooks, none at the marketplace's 7ef543f, which is why
the 2026-10-05 racing run 4 never ran either hook (load a checkout with `claude --plugin-dir <checkout>`; a
`multi-llm doctor` SessionStart `hook_response` in the stream shows the hooks loaded, and
Stop runs show only under `--include-hook-events`). Measured live that day (2.1.289, headless
`-p`): it blocked once and the model's next reply carried the command.

Claude Code writes the transcript in batches (2.1.293, a localhost stub, fix round 27: the reply
lands on file 0.1-0.2 s after `Stop`, and a tool result the reply followed at once
lands with it), so the check reads it once the reply `last_assistant_message` names is on file with no
`stop_hook_summary` after it, or once the file stopped growing, at most `report_check.SETTLE_BUDGET_S`
(`settled_turn`); `install-host --check` never reaches any of these probes, and
`status` reaches only the model-free probes behind the early warning above — codex's
standard `workspace-write` probe and then the write rung's — once per process and under
`subagent_executor.ADVISORY_PROBE_BUDGET_S` (1 s: a hung or slow probe draws no warning, never a slow
`status`, and a measurement the budget cut short is cached nowhere, so the dispatch's
own 10 s probe and verdict do not move), and `provider_registry.py` stays subprocess-free.

`sandbox_unavailable` is the same machine condition as a dispatch-time refusal:
`preflight_runner`'s D6 leg runs the adapter's probe once per host/spec/profile (decision in
`_SANDBOX_DECISIONS`; the measurement in `subagent_executor._SANDBOX_PROBE_CACHE`, the one probe cache the fan-out
shares, keyed by codex version and policy, one rung probe per version across both lanes —
2026-09-24) before the first dispatch of that profile — **the read-only one included** (F2,
2026-09-17): `codex exec --sandbox read-only` is the same bubblewrap, so codex answers a probe for
`agent-readonly` too (the same `codex sandbox` argv under that policy, with a command
that writes nothing — what fails here is bwrap *starting*), where before it declared none and every
read kind (validation, revalidation, consolidation, salvage, task generation, the human-decision
judge) dispatched codex unprobed, and the classifier's `SANDBOX_MARKERS`
(`bwrap:` lines, consulted ahead of `permission_denied`) give the same verdict to a
probe or a dispatch that died on it.

**A poll pinned to exactly one run
(`--run-id`/`--phase`/`--question`) carries that run's own
`mode`/`phase` into its own `MULTI_LLM_RESULT`,
`--wait` or not** (`cli_common.record_mode_phase`, N5, 2026-09-16; the carry-over sat inside
the `--wait` branch until F3, 2026-09-17, so a single pinned poll still emitted
`status`/`null`), read off `.status.json.run.mode` falling back to the
phase dir — the dispatcher stamps `mode: "status", phase: null` before `run_status.main` ever runs,
and the opencode plugin's supervised poll had handed that pair to the model verbatim for a finished
`implement` run; a plain multi-run listing (no selector, or more than one view) keeps
`status`/`null`.

And a live run under a phase already in `phases_completed` (an `implement --resume` reopening
one) renders `workflow phase: completed (re-running since <run.started_at>)` rather than the bare `completed` a reader is likely
to take at face value beside `run.state: running` (N7, 2026-09-16) — "live" there excludes a
`stale` run the identity probe says is **gone**, so a `died (pid N gone, …)` line is
never followed two lines down by `(re-running since …)` (F4, 2026-09-17).

## Printed follow-up commands

**Printed follow-up commands are fully resolved, never placeholders** (`code_review_orchestrator.py`'s two
sites, `utils/validation.py`'s `reaggregate_command`): they compute absolute paths from
`__file__` and are built via `cli_common.reaggregate_command*`. Two rules bind them: the
**allow-rule prefix is part of the contract** — the command must still start with
`uv run …` / `PYTHONUNBUFFERED=1 uv …` or it stops matching Claude Code's
`Bash(uv:*)` glob and every follow-up becomes a permission prompt — and printed commands
**carry their own assertions**, because the golden regression bundle compares phase artifacts and
never stdout.

**`--help` is that same stdout surface**: every argparse `epilog`
interpolates `cli_common.dispatcher_prefix()` and names `multi_llm.py <mode>`, never `${CLAUDE_SKILL_DIR}`
(no shell but Claude Code's expands it) and never a `*_orchestrator.py` path;
`tests/test_instruction_portability.TestHelpSurfaceIsResolved` runs `--help` for every mode word, every `util`
subcommand and every orchestrator's own parser under a fixture skill dir containing a space. Module
docstrings are not printed and may keep a token, but only the `{SKILL_DIR}` template
spelling. An absolute skill-dir path is safe to *print* (session-scoped) and never safe to
*persist*: the Claude plugin cache is keyed by commit hash.

**A next-phase command (`user_next_step.command`) is the other kind, and none of these rules applies
to it**: it is the *host's* invocation (`/multi-llm:multi-llm …`, `$multi-llm:multi-llm …`,
`/multi-llm …`, or codex's file-shim sentence), not `uv run …` — both tool
servers redact every dispatcher command, and a host spelling survives that — and it carries the plan
path as its source supplied it, never resolved or rebased (relative only where the caller's was;
since 2026-10-02 that holds on the tool routes too: `planSpelling` / `plan_spelling`
hand the dispatcher the plan as typed, sigil off and `~` expanded, the child runs
in the anchor so a relative plan names the same file, and a continuation of the same plan keeps the
spelling; only the polls and the abort keep the absolute path, and only inside the server: every
tool call a result hands back (`next`, `abort_command`,
`options`) names the plan as the call typed it (`hand_back_typed` /
`handBackTyped`, 2026-10-05: a re-attach `next` carrying the absolute path
made the next call's `resume_command` absolute).

A needs-human exit 3 on a tool route also carries `resume_command`, the same-phase
`--resume` command in the host's spelling, asked of `util next-command --blocked <mode>`
(`dispatcher.resume` in the surface, additive within v5; since fix round 14 its
`user_resume` also carries the resume's `flags` and the
`--human-task` answers the implement stop recorded (`stop.answers`,
`write_batch_scheduler._stop_answers`), which the servers make the first options — the resume first, so it and
`resume_command` are one command — `resume_answers` and the surface's
`needs_human_message.answers` clause), which answers only for a phase the workflow view shows
`in_progress` — a prerequisite refusal, a stop before the first dispatch or a missing plan
gets none (2026-10-02); since 2026-10-05 it also answers a `generate-tasks` whose dispatch
failed with the plain re-invoke `--generate-tasks <plan>` (`next_phase.RERUN_RESUME_PHASES`), unless a failed
status is one no re-run changes (`retry_ladder.RERUN_FUTILE_STATUSES`) or its prerequisite gate would refuse the
re-run (the gate writes no status, so the file still shows the attempt before it) — and never
followed).

For a composite (`--blocked improve-plan|improve-code`, plans/improve-plan-and-improve-code.md §2.6) it delegates:
it answers for the step a bare re-run would run first (`next_phase._blocked_step`, the first
`sequence` word `improve_workflow.skip_reason` runs on the `read_records` records), with
that phase's own resume, answers, why and section when it has one, else the composite's own
continuation in the composite's word (`_composite_continuation`: `improve_workflow.recorded_continuation` of its run
record's `replay_flags`, `--resume` alone with none); every step on record
prints the `[ALREADY_COMPLETE]` line and exits 2 with no `user_resume`; and
`user_resume.mode` names the mode its command runs, which both tool routes build their option
calls from (`attach_answers` / `attachAnswers`: a call built from the composite's word
would hand `--human-task` to a parser that rejects it).

Every failed or needs-human tool result past argument validation carries `user_next_step: null` and,
without a `resume_command`, the surface's `no_command_reason` (except a result naming a run
with no `exit` — still alive — and a `status`/`abort`
call whose output shows a re-spelled phase command; such a call asks `--blocked` about
the run's phase, read off its result line or its one phase option, 2026-10-05); a needs-human
result's redacted dispatcher commands are re-spelled through `dispatcher.user_command` (plan as typed,
matched by realpath; a command a `"` cut mid-word keeps the pointer and comes
back unparseable), its `options` are one per distinct call, and a restated composed
flag with the same meaning (`reserved_flags.redundant`) is dropped instead of refused (2026-10-05),
quietly: no `extra_flags_note` since run 6, whose hosts relayed it as "`--yes`
was dropped from your flags".

Its quoting is for the host's reader, a model, not for a shell: bare when the path has no whitespace
or quote, double quotes otherwise (single quotes when the path holds a `"`),
nothing escaped, never `shlex.quote`/`_quote`.

## ETA outcomes and denominators

**An `[ETA]` line says what happened to its task, and why its total moved** (fix
round 24, live run 10 #5/#7, Claude Code racing #6).

A task that failed reads `task 2/11 failed (timeout) after 20m 5s` (plus `· half-applied: its edits are kept` when it changed files
first), never `done` and with no estimate, on both legs: the executor passes the
outcome (`_eta_failure`), the native `util metrics record` reads the record
`util batch-done` just wrote (`metrics.native_failure`: `metrics_progress.{phase}.failed` and the recorded
reason, for `--task-id`, which the four runbooks pass, or the id the implement label
starts with; with neither, the one failed task of this run that has no record yet, and among two or
more none — `[METRICS_NO_TASK_ID]` asks for `--task-id` and the entry, marked
`no_task_id`, counts as no progress, fix round 25); its entry is marked
`failed`, which no remaining-time estimate averages in or counts as progress
(`_compute_eta`).

The line whose total is lower than the last line's adds `total now 4 (T007 not run: check T002 reported problems)` in
`left_out_tail`'s words (executor `_eta_dropped` against `_eta_line_ids`;
native `metrics_progress.{phase}.named_out` over `withdrawn`).

A complete implement run whose human-task plan set tasks aside ends `ALL DISPATCHED TASKS COMPLETE · T011 left for you (manual); T012 skipped with a manual task`
(`set_aside_by_plan`; the native emit records it as `state.json`'s
`metrics_set_aside`, dropped by `finish` or an executor start), and
`phase complete` is unchanged. summary.md counts the tasks a runner was started for and the
batches they ran in, naming a task refused before dispatch apart (`_dispatched_sentence`; it read "5
batch task(s) dispatched across 4 batch(es)" for 4 tasks in 3 batches), and `status`'s
implement line counts tasks alone, with no `items applied`.

## Status parity across execution paths

**`status --view run` reads the same on every leg** (fix round 18, live run 7 #8/#11): its
`resume with:` and `re-run it with:` lines are the host's spelling
(`run_status.user_speller`: the `next_command` template, the plan as typed, only the flags the
user typed — `cli_common.drop_host_composed_flags` drops what the host's own skill or shim composes,
`file_route_composed_flags`: `--host` alone on Claude Code, `--host --yes --exec-subagents --detach` on the
codex and opencode shims, fix round 19; a token the one-line spelling cannot carry, such as an
`ask` question with a newline, keeps the dispatcher command), except on a tool
route, whose server re-spells the dispatcher command and hands it back as a tool call.

A native write phase (no run record) is read back from `state.json`: implement
(`state_implement_view`) and the apply phases whose folder holds the native emit's
`write_tasks.json` (`state_apply_views`, `--json`'s `apply_phases`) —
over an earlier executor run's records in that folder too once the native emit wrote
`write_tasks.json` last (its `leg: native` stamp, which an executor run's rewrite
drops; `run_status.set_aside_superseded_runs`, never a detached run still going; the block names that run on an
`earlier run:` line, fix round 26) — each headed `done on the host` /
`in progress on the host (started …; state.json last changed …)` (a `util metrics start` with no `finish`, which a loop that
ended without its finish reads the same as, so the block names the `--resume` for that
case, fix round 19) / `not finished on the host` (which, while a `--resume` continues it,
ends with `resume with:` and, for implement, each answer its stop waits for in
`util next-command`'s words, `run_status._native_stop_lines`, fix round 20) / implement's
`stopped and needs you (not marked completed)`, the executor routes' head, once its loop is closed, no task is in progress,
a task failed or a check's findings hold tasks, and no task is left the host could run without an
answer — or, since fix round 25, such tasks are left and `util metrics finish` closed the emit's
loop (no `metrics_left_out`), the block naming them `T003 can still run: the resume runs it`
(`native_still_runnable`, `--json`'s `still_runnable`) — (`run_status.native_run_stopped`,
`--json`'s `stopped`; the resume and answers below it unchanged; status's
exit code stays the no-run-record 2, fix round 24), and the native implement block prints the
executor routes' `all tasks:` tally.

The unapplied-task-groups notice, which is the optional `--apply-task-suggestions` entry's
`why`, names the valid groups waiting and what skipping them means, worded as the
`--resume` its command is once a run of the phase started (`check_workflow_prerequisites.unapplied_task_groups_notice`, fix
round 19), and every route relays the `command` first, then "Optional, before that:"
with each `why` (plans/next-phase-command.md §2.1, "Primary first").

## Clipping model text

**A model's words in a bounded line go through `utils/text_clip.clip`, never a bare
`text[:n]`** (fix round 22, live run 9 #4: "… would reverse that dec)", "(… 60 m/s), )",
"… but scripts/c)" reached `status`, the notice and a host's reply). It cuts at a word
boundary, ends in "…", drops a dangling separator and closes any bracket or quote the kept part
opened; the budgets stay each surface's own, the OpenCode stop title mirrors it in
`capTitleLine` (a parity test holds them together), and `plain_line` lives beside
it.
