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

This prints the core with every skill-dir placeholder already substituted. Reading
`${CLAUDE_SKILL_DIR}/instructions/<mode>.md` off disk is the documented fallback:
that file uses a brace-delimited template placeholder no shell expands, so you
would have to substitute the path above yourself.

## 5. Capability verbs — how this host answers each one

The core names three capability verbs and never a tool. These are their one
concrete resolution here, so nothing is left to decide mid-run:

- **Spawn subagents** — call your host's own subagent tool `Task`: pass `task.prompt` verbatim, take the role from `task.subagent_type` when your tool needs one, and write the JSON it returns to `task.output_path`. Never pass `--exec-subagents` on this host.
- **Ask the user** — call your host's own structured question tool `AskUserQuestion`, then copy the selected option's command verbatim. Never compose a flag from the user's words.
- **Run detached** — use your host's own background mechanism `run_in_background`, redirecting stdout and stderr to the phase dir's `orchestrator-run.log`. Never pass `--detach` on this host.

## 6. The one dispatcher command

Every mode is this single command, with `--host claude-code` baked in:

```bash
uv run --frozen --no-dev --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/multi_llm.py" <mode> "<plan-path>" --host claude-code [flags…]
```

Exit codes are uniform (`0` done, `1` error, `2` nothing to do / already complete, `3` needs
a human) and the last stdout line is always `MULTI_LLM_RESULT: {…}`: read `state`, `artifacts`
and `next` from it rather than mining the transcript. Run `next` verbatim when it is not
`null` on `0` and `2`. On `3` stop: report to the user what was being run, the last error
line it printed, and the options its output names (`next` included); do not run a different
mode. Do not perform the work this run was going to review, plan, or implement: doing it
yourself is never a substitute for a failed run. Run an option only after the user has picked it.

## 7. Examples

```bash
/multi-llm:multi-llm plans/my-feature.md                 # review-plan (the default)
/multi-llm:multi-llm --full plans/my-feature.md --yes    # fully unattended
/multi-llm:multi-llm --ask "plans/my plan.md" "Is the rollback strategy sufficient?"
```

Providers and models live in `${CLAUDE_SKILL_DIR}/providers.yaml` (per-repository
override via `--init`); `docs/providers.md` in the repository has the rest.
