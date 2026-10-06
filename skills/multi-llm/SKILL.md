---
name: multi-llm
description: "Orchestrate code plan reviews, task generation, implementation, and code reviews using multiple LLM providers in parallel, or ask each model a free-text question about a plan. Use this skill whenever the user wants to review a plan with multiple models, generate implementation tasks, implement tasks with subagent delegation, review code changes against a plan, ask the models a question about a plan, or run any multi-model workflow. Triggers on: 'review my plan', 'multi-llm', 'run code review', 'generate tasks from plan', 'implement this plan', 'use multiple models to review', 'ask the models about my plan', 'ask a question about this plan', '--ask'."
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
argument-hint: [--review-plan|--apply-suggestions|--generate-tasks|--review-tasks|--apply-task-suggestions|--implement|--review-code|--apply-code-fixes|--full|--ask|--status|--init] <plan_path> ["<question>" (required for --ask)] [--models provider:model ...] [--interactive] [--quick] [--yes] [--force] [--dir PATH|--gitignore|--template-only (--init only)]
---

# Multi-LLM — Claude Code entry shim

This file carries the facts true of Claude Code and nothing else; the workflow
lives in the shared host-neutral core, loaded in step 4 — never re-derived here.

## 1. Skill directory

Claude Code expands `${CLAUDE_SKILL_DIR}` to this skill's absolute path inside
*this* file before you read it, so every command below is already resolved: run
them as written, and **keep the double quotes** (the path may contain spaces, e.g.
`C:\Users\John Smith\...`, and an unquoted expansion word-splits).

## 2. Check `uv`, before the first `uv run`

Run `command -v uv` exactly as written — standalone, never chained (chaining loses
the permission allow rule). A path means proceed; otherwise check the locations a
fresh shell may not have on `PATH` yet, reading **stdout, not the exit status**
(this `ls` exits non-zero whenever any candidate is missing, the normal case):

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

The first positional is the **plan path**; for `--ask` the next single quoted
argument is the **question**. Never re-join an unquoted multi-word question, never
re-type a path you could copy, never infer a plan path from the conversation. A
missing plan path, or an empty or unquoted question, is an error: show the
usage hint from `argument-hint` and stop. `--init` takes no plan path.

## 4. Load the mode's core instructions, then execute them step by step

```bash
uv run --frozen --no-dev --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/multi_llm.py" util instructions <mode>
```

This prints the core with every skill-dir placeholder already substituted — but most
cores exceed this host's Bash-result cap, and what comes back is then a
`<persisted-output>` stub: a file path plus the output's first 2 KB. **That stub is not
the core**: it keeps what the mode is and how it is invoked, which reads like a finished
overview, and drops the end, where the steps that do the work are. So first find, in the
result, the line its opening lines say the core ends with; if it is missing, `Read` the
path the stub names, in full, before your next tool call, and treat that file as the core.
Never run a mode command from a core you have not seen the end of — the usage line in the
surviving 2 KB runs cleanly, reports success, and applies nothing. The documented fallback,
`${CLAUDE_SKILL_DIR}/instructions/<mode>.md`, leaves its path placeholder for you to fill.

## 5. Capability verbs — how this host answers each one

The core names three capability verbs and never a tool. These are their one
concrete resolution here, so nothing is left to decide mid-run:

- **Spawn subagents** — call your host's own subagent tool `Task`: when the task carries `task.prompt_path`, tell the subagent to read that file in full — through line `task.prompt_lines`, paging if one read returns fewer — and follow it verbatim, printing only what it asks; if the file is absent, or the task has no `prompt_path`, pass `task.prompt` verbatim. Take the role from `task.subagent_type` when your tool needs one, and write the JSON it returns to `task.output_path`. Never pass `--exec-subagents` on this host.
- **Ask the user** — call your host's own structured question tool `AskUserQuestion`, then copy the selected option's command verbatim. Never compose a flag from the user's words.
- **Run detached** — use your host's own background mechanism `run_in_background`, redirecting stdout and stderr to the phase dir's `orchestrator-run.log` (`> … 2>&1`, the one dispatcher command that takes a redirect: every foreground call runs bare). Never pass `--detach` on this host. Then make one **non-waiting** `status <plan> <selector> --host <this host's id> --view run --exit-zero` call — no `--wait`, no `--timeout`, so it answers in well under a second — and relay its `progress:` line and `models:` table once: the count is on the one, and the other's rows are the only thing that says which CLI and model each of them runs. The run writes that roster a few seconds after launch, so repeat that one call once if no table printed yet, adding `--wait --timeout 10` as the wait (never `sleep` before a call: a call that does not start with `uv` loses its permission rule) — and a table that *does* print can still be the previous run's, because the phase dir is re-used and the write that claims it for this fan-out drops the predecessor's `progress` only about 0.2 s in: a table under a `progress:` step that is not `fanout` is the last run's, so make that call again rather than relaying it. That step is the whole test — a head line reading `finished in foreground (…)` rather than `running` is what a re-used dir prints all run long once an earlier `--detach` run left its own terminal `run` record in it, so the tense names no run. Then poll while it runs, one foreground `status <plan> <selector> --host <this host's id> --view run --wait --timeout <interval> --exit-zero` call per turn at a 120-300 s interval, relaying the rendered `progress:` line and, whenever one prints, the `just finished:` line that names the model that turned terminal during that wait — verbatim. Its absence claims nothing: that line is a difference between two readings taken inside one wait, so a model that finished in the gap between polls is already terminal when the next poll takes its baseline — so every poll compares its `models:` table with the previous poll's and names every newly terminal row that line did not name, copying its spec and state word out of the row — a copy, never a recount. Nothing else from the view is relayed mid-run. Spell `--host` out yourself — no launcher printed a command to copy it from, and without it the wait is clamped under the tightest ceiling on record — and give the call itself a timeout of at least the interval plus 30 s, in that parameter's own unit — milliseconds here, so 330000 for a 300 s interval, not 330, which is a third of a second — because this host's default per-call cap is shorter than the interval and would otherwise kill every poll from the first. No exit code from that poll ends the loop except `1`, the unreadable one: nothing on this host names one invocation, so without `--exit-zero` `2` is the steady state and a `0` or a `3` belongs to an earlier run in the same dir and names no verdict for this one: read the exit code as nothing, relay the rendered line exactly as on a `2`, and keep polling. Both calls carry `--exit-zero` so every answer that was read exits `0` and none shows as an error. The loop ends when the background mechanism reports the process exited; read the markers and output paths from `orchestrator-run.log` then, and report a non-zero exit status with that path rather than retrying. The mode instruction names the selector and the rest of the loop.

## 6. The one dispatcher command

Every mode is this single command, with `--host claude-code` baked in:

```bash
uv run --frozen --no-dev --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/multi_llm.py" <mode> "<plan-path>" --host claude-code [flags…]
```

Exit codes are uniform (`0` done, `1` error, `2` nothing to do / already complete, `3` needs a human) and the
last stdout line is always `MULTI_LLM_RESULT: {…}`: read `state`, `artifacts` and `next` from it rather than
mining the transcript. Run `next` verbatim when it is not `null` on `0` and `2`. On `3` stop: report to the
user what was being run, the last error line it printed, and the options its output names (`next` included);
do not run a different mode. Do not perform the work this run was going to review, plan, or implement: doing
it yourself is never a substitute for a failed run. Run an option only after the user has picked it. Run `next` as printed, never re-typed: its plan path as printed, no flag added or dropped (no re-spelled plan flag, no lost `--quick --yes`). Run every foreground dispatcher call bare: no pipe (not even `| tail`), no `2>&1`, no `sleep` before it — a pipe reports `tail`'s exit code (a `3` reads as `0`), a trim drops the marker lines above the result line, and a command that does not start with `uv` loses this skill's allow rule. A long output is saved to a file: read it. Run the repository's own test or check command (implement's regression step) the same way: bare, its whole output read. The detached launch (*Run detached*, section 5) is the one exception: it redirects stdout and stderr into the phase dir's `orchestrator-run.log` (`> … 2>&1`).
`user_next_step` is the phase after this one, for the user to type: never run it yourself; end your report
with its `command`, and each `optional_before` command, every one in its own fenced block, verbatim. After a workflow phase exits `0` or `2` with `next` `null` (never after `status`, `ask`, `init`), get it as the report's first step, after every command the core names: the command above
with `util next-command "<plan-path>" --after <mode>` replacing `<mode> "<plan-path>"`, no `[flags…]`, the plan path as the user typed it — even when a `[NEXT_COMMAND]` line is already on screen (a `next` run can name it absolutely). After
`1` or `3` look nothing up: a failed run has no next command. A native `implement` that stopped part-way exited `0`, so look it up: it prints the `--resume` command and, when tasks wait for an answer, `[NEXT_COMMAND_ANSWERS]`, one command carrying every answer — end with both, fenced, verbatim. A lookup that prints `[WORKFLOW_COMPLETE]` ends the report with "Workflow complete.", each `[MANUAL_TASK]` line's task (id and title) as the user's to do, then the `[MANUAL_TASK_NOTE]` line's text verbatim: it says how to record them done. After implement those lines can also follow a `[NEXT_COMMAND]` block: list those tasks and give that note the same way, after the commands. A lookup that prints no `[NEXT_COMMAND]` or `[WORKFLOW_COMPLETE]` line means there is none: write no command and never fill the gap (not from `suggested next:`, a plan path or a default).

## 7. Examples

```bash
/multi-llm:multi-llm plans/my-feature.md                 # review-plan (the default)
/multi-llm:multi-llm --full plans/my-feature.md --yes    # fully unattended
/multi-llm:multi-llm --ask "plans/my plan.md" "Is the rollback strategy sufficient?"
```

Providers and models live in `${CLAUDE_SKILL_DIR}/providers.yaml` (per-repository
override via `--init`); `docs/providers.md` in the repository has the rest.
