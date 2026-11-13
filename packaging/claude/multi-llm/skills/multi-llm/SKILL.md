---
name: multi-llm
description: "Orchestrate code plan reviews, task generation, implementation, code reviews and commit reviews using multiple LLM providers in parallel, or ask each model a question about a plan. Use this skill whenever the user wants to review a plan with multiple models, generate implementation tasks, implement tasks with subagents, review code changes against a plan, review or fix the commits they made, ask the models a question about a plan, or run any multi-model workflow. Triggers on: 'review my plan', 'multi-llm', 'run code review', 'generate tasks from plan', 'implement this plan', 'use multiple models to review', 'ask the models about my plan', 'ask a question about this plan', '--ask', 'improve my plan', 'review and apply the suggestions', 'improve the code', 'review and fix the code', 'review my last commit', 'review the last N commits', 'review this branch's commits', 'fix the issues in my last commit', 'review the tasks', 'apply the suggestions', 'apply the task suggestions', 'apply the code fixes'."
allowed-tools:
  - Bash(uv:*)
  - Bash(PYTHONUNBUFFERED=1 uv:*)
  - Bash(command -v uv)
  - Bash(ls:*)
  - Bash(grep:*)
  - Bash(diff:*)
  - Bash(tail:*)
  - Bash(dirname:*)
  - Bash(test -x:*)
  - Bash(git rev-parse:*)
  - Read
  - Edit
  - Write
  - Task
  - AskUserQuestion
  - Glob
  - Grep
argument-hint: [--review-plan|--apply-suggestions|--improve-plan|--generate-tasks|--review-tasks|--apply-task-suggestions|--implement|--review-code|--apply-code-fixes|--improve-code|--review-commits|--apply-commit-fixes|--improve-commits|--full|--ask|--status|--init] <plan_path>|[<range>] (<range>: --*-commits only) ["<question>" (required for --ask)] [--models provider:model ...] [--interactive] [--quick] [--yes] [--force] [--scope-policy expand|preserve] [--focus "<text>"|--context-plan PATH (--*-commits only)] [--dir PATH|--gitignore|--template-only (--init only)]
---

# Multi-LLM — Claude Code entry shim

This file carries the facts true of Claude Code and nothing else; the workflow
lives in the shared host-neutral core, loaded in step 4 — never re-derived here.

## 1. Skill directory

Claude Code expands `${CLAUDE_SKILL_DIR}` to this skill's absolute path inside *this* file before you read it, so every command below is already resolved: run
them as written, and **keep the double quotes** (the path may contain spaces, e.g. `C:\Users\John Smith\...`, and an unquoted expansion word-splits).

## 2. Check `uv`, before the first `uv run`

Run `command -v uv` exactly as written — standalone, never chained (chaining loses the permission allow rule). A path means proceed; otherwise check the locations a
fresh shell may not have on `PATH` yet, reading **stdout, not the exit status** (this `ls` exits non-zero whenever any candidate is missing, the normal case):

```bash
ls "$HOME/.local/bin/uv" "$HOME/.local/bin/uv.exe" "$HOME/.cargo/bin/uv" "$HOME/.cargo/bin/uv.exe" /opt/homebrew/bin/uv /usr/local/bin/uv /opt/local/bin/uv 2>/dev/null
```

If uv is genuinely absent, print `${CLAUDE_SKILL_DIR}/hosts/uv-missing.txt`
verbatim and **stop with exit 3** — never install it, never ask, never fall back
to bare `python`.

## 3. Map the user's flag to a mode word

Flags are position-independent, and the mode flag is the mode word without its
`--` (`--review-code` → `review-code`, `--full` → `full`, and so on for every flag
in `argument-hint`). **No mode flag means `review-plan`.** Everything else the user
typed passes through verbatim — never add a flag they did not type.

**A request in plain words** has no flag, so its mode word is selected here, before step 4, and goes to `util instructions` and to the dispatch: 'review my plan' is `review-plan`; 'apply the suggestions' is `apply-suggestions`; 'improve my plan' and 'review and apply the suggestions' are `improve-plan`; 'generate tasks from plan' and 'generate tasks from the plan' are `generate-tasks`; 'review the tasks' is `review-tasks`; 'apply the task suggestions' is `apply-task-suggestions`; 'implement this plan' is `implement`; 'run code review' and 'review the code against the plan' are `review-code`; 'apply the code fixes' is `apply-code-fixes`; 'improve the code' and 'review and fix the code' are `improve-code`; 'review my last commit', 'review the last N commits' and 'review this branch's commits' are `review-commits`; 'fix the issues in my last commit' is `apply-commit-fixes`; 'review and fix this branch's commits' is `improve-commits`; 'run the full multi-llm workflow' is `full`; 'ask the models about my plan', 'ask a question about this plan' and 'ask the models a question about my plan' are `ask`; 'check the multi-llm run status' is `status`; 'set up multi-llm providers for this repo' is `init`. The mode word is the one exception to adding what they did not type: add no other flag, and a commit mode's range is the one the request names, in a form the dispatcher reads (`last 2 commits`, `-2`, `this branch` for 'this branch's commits'; none means `HEAD`). The plan path and the question come from the request and never from the conversation, so ask for a missing one before dispatching; `init` needs neither, and a commit mode takes its range instead of a plan path. For `ask`, pass the question as the one single-quoted argument after the plan path. Ask too when the intent is unclear: bare `multi-llm`, `/multi-llm` or 'use multiple models to review' names no workflow. A general request to review a plan does not override a user's requirement that they agree to multi-llm explicitly. A literal `/multi-llm:multi-llm` command keeps the rules around this paragraph, and its mode flag decides the mode: when the words around it ask for something else, ask which they mean instead of dispatching.

The first positional is the **plan path**; for `--ask` the next single quoted argument is the **question**. Never re-join an unquoted multi-word question, never
re-type a path you could copy, never infer a plan path from the conversation. A missing plan path, or an empty or unquoted question, is an error: show the
usage hint from `argument-hint` and stop. `--init` takes no plan path.

**The three commit modes** (`--review-commits`, `--apply-commit-fixes`, `--improve-commits`) take an optional **range**, not a plan path: the bare words right after the mode flag, none meaning `HEAD`. Bare words are joined into one range (`--review-commits last 2 commits` needs no quotes, unlike `--ask`'s question), and a `-N` there is a range (`-2`, the last 2 commits).
Pass them right after the mode word, before any flag, as typed: range words after a flag are refused with `[RANGE_AFTER_FLAGS]`, and a run takes one range. A plan goes in `--context-plan <path>`, never in the range; after the launch, every command you spell takes the brief `[COMMIT_BRIEF]` names as its plan path, as the mode's core says, and a command the run prints keeps the pinned range it prints.

## 4. Load the mode's core instructions, then execute them step by step

```bash
uv run --frozen --no-dev --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/multi_llm.py" util instructions <mode> --to-file
```

It writes the core, every path resolved, to a file and prints `[CORE_FILE] <path> lines=<n> bytes=<n>`: `Read` that file in full, through line `<n>` (paging if one read
returns fewer), before you run anything else. No such line means it printed the core instead, and its first lines say how to check you have all of it.
Load every core this way, including a phase's core that a core tells you to load with `util instructions <phase>`: add `--to-file` to that command and read the file it names by the same rule.
`status` has no core: `util instructions status` answers with its flags and the dispatcher drives that mode, so use the same dispatcher command prefix with `status --help`. If a `util instructions` call fails, report its last error line and stop, because reading a raw file in `instructions/` recovers nothing: a missing or unreadable core means a corrupt or partial install, so tell the user to reinstall the plugin, and a uv or Python failure or a denied call blocks every later dispatcher command too. Bundled resource references the core names as `references/<file>.md` are relative to the skill directory: Read `"${CLAUDE_SKILL_DIR}/references/<file>.md"`, never a repo-relative path; preserve dispatcher-supplied artifact paths and user-repository paths.

## 5. Capability verbs — how this host answers each one

The core names three capability verbs and never a tool. These are their one concrete resolution here, so nothing is left to decide mid-run:

- **Spawn subagents** — call your host's own subagent tool `Agent`: when the task carries `task.prompt_path`, tell the subagent to read that file in full — through line `task.prompt_lines`, paging if one read returns fewer — and follow it verbatim, printing only what it asks; if the file is absent, or the task has no `prompt_path`, pass `task.prompt` verbatim. Take the role from `task.subagent_type` when your tool needs one, and write the JSON it returns to `task.output_path`. Never pass `--exec-subagents` on this host. If `Agent` is unavailable and `Task` is exposed, use `Task` instead.
- **Ask the user** — call your host's own structured question tool `AskUserQuestion`, then copy the selected option's command verbatim. Never compose a flag from the user's words.
- **Run detached** — use your host's own background mechanism `run_in_background`, redirecting stdout and stderr to the phase dir's `orchestrator-run.log` (`> … 2>&1`, the one dispatcher command that takes a redirect: every foreground call runs bare), and give that launch call a timeout of its own, the longest the parameter takes: 7200000 here, in milliseconds (2 hours; a session whose maximum is lower clamps it rather than refusing it). Without one the mechanism stops the command at its default deadline, as short as ten minutes, and a fan-out still running then dies mid-run (exit `143`, `next` `null`). Never pass `--detach` on this host. Then make one **non-waiting** `status <plan> <selector> --host <this host's id> --view run --exit-zero` call — no `--wait`, no `--timeout` — and relay its `progress:` line and `models:` table once: the table's rows are the only thing that says which CLI and model each of them runs. The run writes that roster a few seconds after launch, so repeat that one call once if no table printed yet, adding `--wait --timeout 10` as the wait (never `sleep` before a call: a call that does not start with `uv` loses its permission rule) — and a table that *does* print can still be the previous run's, because the phase dir is re-used: a table under a `progress:` step that is not `fanout` is the last run's, so make that call again rather than relaying it. That step is the whole test, since a re-used dir can read `finished in foreground (…)` rather than `running` all run long. Then poll while it runs, one foreground `status <plan> <selector> --host <this host's id> --view run --wait --timeout <interval> --exit-zero` call per turn at a 120-300 s interval, relaying verbatim the rendered `progress:` line and, whenever one prints, the `just finished:` line that names the model that turned terminal during that wait. Its absence claims nothing, because a model can finish between two polls: every poll compares its `models:` table with the previous poll's and names every newly terminal row that line did not name, copying its spec and state word out of the row — a copy, never a recount. Nothing else from the view is relayed mid-run. Spell `--host` out yourself, since without it the wait is clamped under the tightest ceiling on record, and give the call itself a timeout of at least the interval plus 30 s, in that parameter's own unit — milliseconds here, so 330000 for a 300 s interval, not 330, which is a third of a second — because this host's default per-call cap is shorter than the interval. No exit code from that poll ends the loop except `1`, the unreadable one: without `--exit-zero`, `2` is the steady state and a `0` or a `3` belongs to an earlier run in the same dir and names no verdict for this one, so relay the rendered line exactly as on a `2` and keep polling. Both calls carry `--exit-zero` so every answer that was read exits `0` and none shows as an error. The loop ends when the background mechanism reports the process exited; read the markers and output paths from `orchestrator-run.log` then, and report a non-zero exit status with that path rather than retrying. A stop at the mechanism's time limit (exit `143`, `next` `null`, `status` reading `ended in foreground without a final state`) names no way forward of its own, so offer the user each of these and run none yourself: the launch, made again the same way with `--force` added, which resumes, keeping each model whose result file is in the phase dir (`[SKIP]`) and running the rest (without it, review-code and the commit modes refuse a dir holding results, exit `3`; never `--rerun-all`); and, on a review mode with at least one such file, its `--reaggregate`, which reports with those and lists the rest as failed. Say which models have a file: a `done` row can have none, its result held in memory for the fan-out's end. Where the mode instruction names a continuation for a stopped run (a composite's `[IMPROVE_RERUN]`), offer that instead. The mode instruction names the selector and the rest of the loop.

Pass `run_in_background: false` on each subagent call, putting a wave's calls in one message, so the replies come back in this turn: this skill's permission rules last only for the turn that loaded it, and with the subagents in the foreground and a detached launch waited on by its poll loop, a phase normally runs inside that turn. While a phase runs, wait in this session for what it started: never call `ScheduleWakeup`, `CronCreate` or `/loop` to come back later, and never schedule a phase command to run again: a scheduled prompt keeps a headless session open after its last reply and, when it fires, runs the phase a second time (the plugin's hook refuses both while a phase runs). When a batch's edits need more work, send the batch back to the subagent that ran it with `SendMessage`, loading that tool through `ToolSearch` first if it is deferred, and verify it again. A resumed subagent runs in the background, as does any subagent the host moves there anyway, so once nothing else is left to do, end your turn: its completion notification arrives on its own and starts the next one, and you carry on with the phase from there. If a dispatcher command or an edit then needs approval, that is the host asking the user, not a failure to work around: never re-spell, chain or wrap a command to avoid it, and if it is refused, report what was refused and stop. If `SendMessage` stays unavailable, tell the user this host cannot resume the batch, which takes `SendMessage` (Claude Code 2.1.200 and newer have it), and leave the batch pending: never claim it complete or create a replacement subagent. Never edit a batch's files yourself, since its diff, its record and its summary all count every change as its subagent's.

## 6. The one dispatcher command

Every mode is this single command, with `--host claude-code` baked in, and every `util` command a core names starts the same way. Run each one from the directory this session started in (normally the repository root), and never `cd` in a Bash call: this host's Bash keeps the new directory for every later call, so a relative plan path then names a file that is not there. Run each one alone in its Bash call and spelled exactly as written here, the skill directory as printed: nothing chained before or after it (no probe, no `cd`, no other command) and no shell variable standing in for it, since a call that does not start with `uv` loses this skill's allow rule:

```bash
uv run --frozen --no-dev --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/multi_llm.py" <mode> "<plan-path>" --host claude-code [flags…]
```

Exit codes are uniform (`0` done, `1` error, `2` nothing to do / already complete, `3` needs a human) and the
last stdout line is always `MULTI_LLM_RESULT: {…}`: read `state`, `artifacts` and `next` from it rather than
mining the transcript, and report what its `notice` says, when it has one, every part of it, as plain text. Put it above the closing section, and repeat nothing that section says. Run `next` verbatim when it is not `null` on `0` and `2`. On `3` stop: report to the
user which phase stopped, on which plan or commit range, the last error line it printed, whole (on a stop it says what stopped, what it
reported, what did not run and whether anything ran the repository's tests), and the options its output names (`next` included);
do not run a different mode. Do not perform the work this run was going to review, plan, or implement: doing
it yourself is never a substitute for a failed run. Run an option only after the user has picked it. Never edit `driver.write_paths`, or any other multi-llm setting in `.multi-llm/providers.yaml`, yourself. At a stop, offer only the ways forward the stop itself names, as it words them: when it names a change to `driver.write_paths` that lets a refused file be written (adding it to `allow`, or removing or narrowing a deny rule that covers it), that is one of them and the user's call; add no advice about the write policy of your own. Run `next` as printed, never re-typed: its plan path as printed, no flag added or dropped (no re-spelled plan flag, no lost `--quick --yes`). Run every foreground dispatcher call bare: no pipe (not even `| tail`), no `2>&1`, no `sleep` before it — a pipe reports `tail`'s exit code (a `3` reads as `0`), a trim drops the marker lines above the result line, and a command that does not start with `uv` loses this skill's allow rule. A long output is saved to a file: read it. Run the repository's own test or check command (implement's regression step) the same way: bare, its whole output read. The detached launch (*Run detached*, section 5) is the one exception: it redirects stdout and stderr into the phase dir's `orchestrator-run.log` (`> … 2>&1`).
A stop at the background mechanism's time limit is the one stop whose ways forward are not its own to name: it names none, so offer the ones *Run detached* (section 5) lists for it, as it words them.
`user_next_step` is the phase after this one, for the user to type: never run it yourself; end your report
with the section the lookup prints between its `[NEXT_STEP_MARKDOWN]` and `[NEXT_STEP_MARKDOWN_END]` lines (its `markdown`), copied as it is (its words and order, each command alone in its fenced block, each why and note plain text, the manual tasks it lists included, no caption of your own). Only that section says "Workflow complete.": never write those words yourself. Copy it word for word, never wrapped in a fenced block of your own (its commands are fenced already), and say none of it twice: its whys, notes, manual tasks and "Workflow complete." appear only in it, never above it. After a workflow phase exits `0` or `2` with `next` `null` (never after `status`, `ask`, `init`), get it as the report's first step, after every command the core names: the command above
with `util next-command "<plan-path>" --after <mode>` replacing `<mode> "<plan-path>"`, no `[flags…]`, the plan path as the user typed it — even when a `[NEXT_COMMAND]` line is already on screen (a `next` run can name it absolutely). After
`1` or `3` look nothing up: a failed run has no next command. A native `implement` that stopped part-way exited `0`, so look it up: its section is the resume and each answer the stop offers, each with its why: copy it as given, whys included, in place of any list, recap or summary of the ways forward of your own; when the plugin ran no tests, the lookup also prints a `[TESTS_NOT_RUN]` line: give the text after the tag in the report, verbatim, as plain text. When the command it names would be refused for the runner the config resolves to, the lookup prints a `[RUNNER_WARNING]` line above its block, which the section does not carry: give the text after the tag in the report, verbatim, as plain text, above the section. A lookup that prints no `[NEXT_COMMAND]` or `[WORKFLOW_COMPLETE]` line means there is none: write no command and never fill the gap (not from `suggested next:`, a plan path or a default).

## 7. Examples

```bash
/multi-llm:multi-llm plans/my-feature.md                 # review-plan (the default)
/multi-llm:multi-llm --full plans/my-feature.md --yes    # fully unattended
/multi-llm:multi-llm --ask "plans/my plan.md" "Is the rollback strategy sufficient?"
/multi-llm:multi-llm --review-commits                    # review the last commit (HEAD)
/multi-llm:multi-llm --review-commits last 2 commits     # a range: bare words, no quotes
/multi-llm:multi-llm --improve-commits this branch       # review, then fix, this branch's commits
```

Providers and models live in `${CLAUDE_SKILL_DIR}/providers.yaml` (per-repository override via `--init`); [the provider documentation](https://github.com/beastlabai/multi-llm-plugin/blob/master/docs/providers.md) has the rest.
