---
name: multi-llm
description: "Orchestrate code plan reviews, task generation, implementation, and code reviews using multiple LLM providers in parallel, or ask each model a free-text question about a plan. Use this skill whenever the user wants to review a plan with multiple models, generate implementation tasks, implement tasks with subagent delegation, review code changes against a plan, ask the models a question about a plan, or run any multi-model workflow. Triggers on: 'review my plan', 'multi-llm', 'run code review', 'generate tasks from plan', 'implement this plan', 'use multiple models to review', 'ask the models about my plan', 'ask a question about this plan', '--ask'."
allowed-tools:
  - Bash(uv:*)
  - Bash(grep:.*)
  - Read
  - Edit
  - Write
  - Task
  - AskUserQuestion
  - Glob
  - Grep
argument-hint: [--review-plan|--apply-suggestions|--generate-tasks|--review-tasks|--apply-task-suggestions|--implement|--review-code|--apply-code-fixes|--full|--ask|--status|--init] <plan_path> ["<question>" (required for --ask)] [--models provider:model ...] [--interactive] [--quick] [--yes] [--force] [--dir PATH|--gitignore|--template-only (--init only)]
---

# Multi-LLM Skill

> **Skill directory & path resolution — read this first.**
>
> Every orchestrator script, instruction file, prompt, schema, and template this
> skill uses is bundled inside the skill's own directory, whose absolute path is:
>
> ```
> ${CLAUDE_SKILL_DIR}
> ```
>
> Claude Code expands `${CLAUDE_SKILL_DIR}` to that absolute path inside *this*
> SKILL.md before you read it, so every command and path shown below is already
> fully resolved — run them as written.
>
> The mode instruction files you open with the Read tool, and any "run this
> next" commands the orchestrators print to stdout, are **not** pre-expanded:
> they still contain the literal placeholder `CLAUDE_SKILL_DIR` (written as the
> shell-style variable `${...}`). Whenever you encounter that placeholder in a
> file you read or in script output, substitute the absolute skill-directory
> path shown above before running the command or reading the file. The shell
> does **not** export this variable, so never run a command that still contains
> an unexpanded `CLAUDE_SKILL_DIR`.

A unified skill for multi-LLM plan automation. Supports eleven workflow modes plus a status command:

1. **Review Plan** (`--review-plan`): Review an implementation plan with multiple LLMs (default)
2. **Apply Suggestions** (`--apply-suggestions`): Apply validated suggestions from review to the plan
3. **Generate Tasks** (`--generate-tasks`): Generate detailed implementation tasks from a high-level plan
4. **Review Tasks** (`--review-tasks`): Review generated tasks with multiple LLMs
5. **Apply Task Suggestions** (`--apply-task-suggestions`): Apply validated task review suggestions to tasks.md
6. **Implement** (`--implement`): Execute implementation tasks from a plan
7. **Review Code** (`--review-code`): Review code changes against the plan
8. **Apply Code Fixes** (`--apply-code-fixes`): Apply validated fixes from code review
9. **Full Workflow** (`--full`): Run all modes in sequence
10. **Status** (`--status`): Show current workflow state and suggested next action
11. **Ask** (`--ask`): Ask each model a free-text question about a plan; aggregate answers into one markdown file
12. **Init Config** (`--init`): Set up a per-project provider config override at `<git-root>/.multi-llm/providers.yaml` (no plan path; routed via `instructions/init-config.md`). Fully automatic and zero-prompt: it auto-detects which provider CLIs are installed on `PATH` and writes a preconfigured override (uncommenting the detected providers' blocks and `default_provider`); `--template-only` skips detection and writes the inert commented stub. Flags: `--dir PATH`, `--force`, `--gitignore`, `--template-only`.

## Quick Start

```bash
# Review a plan using YAML defaults (no prompting)
/multi-llm:multi-llm plans/my-feature.md

# Explicit review-plan mode with YAML defaults
/multi-llm:multi-llm --review-plan plans/my-feature.md

# Specify models from multiple providers
/multi-llm:multi-llm --review-plan plans/my-feature.md --models cursor-agent:auto gemini:gemini-2.5-flash

# Force interactive selection (overrides YAML defaults)
/multi-llm:multi-llm --review-plan plans/my-feature.md --interactive

# Quick review with 2 models (faster)
/multi-llm:multi-llm --review-plan plans/my-feature.md --quick

# Quick code review
/multi-llm:multi-llm --review-code plans/my-feature.md --quick

# Apply validated suggestions from review to the plan
/multi-llm:multi-llm --apply-suggestions plans/my-feature.md

# Generate detailed tasks from a high-level plan
/multi-llm:multi-llm --generate-tasks plans/my-feature.md

# Review generated tasks with multiple LLMs
/multi-llm:multi-llm --review-tasks plans/my-feature.md

# Quick task review (fewer models, faster)
/multi-llm:multi-llm --review-tasks plans/my-feature.md --quick

# Apply validated task review suggestions to tasks.md
/multi-llm:multi-llm --apply-task-suggestions plans/my-feature.md

# Implement tasks from a plan
/multi-llm:multi-llm --implement plans/my-feature.md

# Review code changes
/multi-llm:multi-llm --review-code plans/my-feature.md

# Apply code fixes from a previous review
/multi-llm:multi-llm --apply-code-fixes plans/my-feature.md

# Run full workflow
/multi-llm:multi-llm --full plans/my-feature.md

# Run full workflow fully unattended (zero prompts): non-interactive model
# selection, Claude decides all human-decision items, auto-runs review-tasks,
# and auto-applies code-review fixes
/multi-llm:multi-llm --full plans/my-feature.md --yes
/multi-llm:multi-llm --full plans/my-feature.md --yes --quick   # also pin to the 2 quick models

# Check workflow status
/multi-llm:multi-llm --status plans/my-feature.md

# Ask each model a free-text question about a plan (read-only Q&A)
/multi-llm:multi-llm --ask plans/my-feature.md "Is the rollback strategy sufficient?"
```

## Target Repository

The repo where multi-llm runs should have one or more `AGENTS.md` files
throughout (root and/or subdirectories) so every code harness gets consistent
project context. `AGENTS.md` is supported by most harnesses that multi-llm
invokes (Cursor Agent, Codex, OpenCode, Gemini CLI, Grok Build, Cline, goose, Aider, Antigravity CLI, etc.).

For a single source of truth with Claude Code, add a matching `CLAUDE.md`
alongside each `AGENTS.md` containing only:

```text
@AGENTS.md
```

Example in this skill: [AGENTS.md](AGENTS.md), [CLAUDE.md](CLAUDE.md).

## Pre-execution Validation

Flags are **position-independent**: `--models` (variadic), `--quick`,
`--interactive`, `--yes` / `--non-interactive`, `--force`, and the mode flags
(`--ask`, `--review-plan`, …) may appear anywhere. These blocks scan argv for `--ask` to detect ask mode, then
collect the positional tokens (skipping every `--flag` and the values consumed
by variadic `--models`): the first positional is the **plan path** and, for
`--ask`, the second is the **question**.

```bash
! ASK=0; PLAN_PATH=""; Q=""; pos=0; skip_vals=0; for a in "$@"; do if [ "$skip_vals" = 1 ]; then case "$a" in --*) skip_vals=0;; *) continue;; esac; fi; case "$a" in --ask) ASK=1; continue;; --models) skip_vals=1; continue;; --*) continue;; esac; pos=$((pos+1)); if [ "$pos" = 1 ]; then PLAN_PATH="$a"; elif [ "$pos" = 2 ]; then Q="$a"; fi; done; if [ -z "$PLAN_PATH" ]; then echo "ERROR: Plan path required. Usage: /multi-llm:multi-llm [--review-plan|--implement|--review-code|--full|--ask <plan_path> \"<question>\"] <plan_path> (mode flag optional, defaults to --review-plan)"; exit 1; fi
```

```bash
! ASK=0; PLAN_PATH=""; pos=0; skip_vals=0; for a in "$@"; do if [ "$skip_vals" = 1 ]; then case "$a" in --*) skip_vals=0;; *) continue;; esac; fi; case "$a" in --ask) ASK=1; continue;; --models) skip_vals=1; continue;; --*) continue;; esac; pos=$((pos+1)); [ "$pos" = 1 ] && PLAN_PATH="$a"; done; if [ ! -f "$PLAN_PATH" ]; then echo "ERROR: Plan file not found: $PLAN_PATH. Usage: /multi-llm:multi-llm [--review-plan|--implement|--review-code|--full|--ask <plan_path> \"<question>\"] <plan_path> (mode flag optional, defaults to --review-plan)"; exit 1; fi
```

```bash
! ASK=0; Q=""; pos=0; skip_vals=0; for a in "$@"; do if [ "$skip_vals" = 1 ]; then case "$a" in --*) skip_vals=0;; *) continue;; esac; fi; case "$a" in --ask) ASK=1; continue;; --models) skip_vals=1; continue;; --*) continue;; esac; pos=$((pos+1)); [ "$pos" = 2 ] && Q="$a"; done; if [ "$ASK" = 1 ] && [ -z "${Q//[[:space:]]/}" ]; then echo "ERROR: --ask requires both a plan path and a non-empty question. Usage: /multi-llm:multi-llm --ask <plan_path> \"<question>\" (quote the question as a single argument)"; exit 1; fi
```

## Provider Configuration

Providers and models are configured in `${CLAUDE_SKILL_DIR}/providers.yaml` (the
**base** layer).

### Per-project override (optional)

A repository can override its provider/model defaults without editing the
installed plugin by adding `<git-root>/.multi-llm/providers.yaml`. It is optional
and auto-discovered; absent → base behavior, unchanged. Config is layered
base → project-local → `MULTI_LLM_PROVIDERS_CONFIG` env override, deep-merged with
**lists replacing wholesale**. Set one up with the **`--init`** flag (routed via
`instructions/init-config.md`) — fully automatic and zero-prompt, it auto-detects
the installed provider CLIs and writes a preconfigured override:

```bash
/multi-llm:multi-llm --init                 # auto-detect installed CLIs; writes <git-root>/.multi-llm/providers.yaml
/multi-llm:multi-llm --init --template-only # skip detection; write the inert commented stub to hand-edit
/multi-llm:multi-llm --init --gitignore     # keep it developer-local (untracked)
```

The auto-discovered file deep-merges its full contents (including a `providers:`
block) over base, identical to the env layer; `command:` is documentation-only and
is never executed. See the README "Per-project configuration" section for full
semantics.

### providers.yaml Format

```yaml
providers:
  cursor-agent:
    command: cursor-agent           # CLI binary name
    default_timeout: 1200           # Timeout in seconds
    supports_json_output: true      # Native JSON support
    models:
      - auto
      - gpt-5.2-high
      - gemini-3-pro

  gemini:
    command: gemini
    default_timeout: 900
    supports_json_output: true
    models:
      - gemini-2.5-flash
      - gemini-2.5-pro

  opencode:
    command: opencode
    default_timeout: 1200
    supports_json_output: true
    models:
      - opencode/big-pickle
      - opencode/sonnet

# Default provider when model name doesn't include provider prefix
default_provider: cursor-agent

# Default models when none specified on CLI (skips interactive selection)
defaults:
  models:
    - cursor-agent:auto
    - gemini:gemini-2.5-flash
```

### Model Specification Syntax

Models can be specified in two formats:

1. **With provider prefix**: `provider:model`
   - Examples: `gemini:gemini-2.5-flash`, `opencode:opencode/big-pickle`, `cursor-agent:auto`

2. **Without prefix** (uses `default_provider`):
   - Examples: `auto`, `gpt-5.2-high` (resolved as `cursor-agent:auto`, `cursor-agent:gpt-5.2-high`)

### Model Selection Priority

When running a mode, model selection follows this priority:

1. **CLI `--models` flag**: Use exactly these models
2. **CLI `--interactive` flag**: Force two-step interactive selection (provider, then models)
3. **CLI `--quick` flag**: Use `quick_models` from providers.yaml (lightweight, 2 models)
4. **YAML `defaults.models`**: Use configured defaults (no prompting)
5. **Interactive selection**: Two-step selection (fallback when no defaults configured)

### Example Commands

```bash
# Use YAML defaults (no prompting required)
/multi-llm:multi-llm --review-plan plans/my-feature.md

# Explicit models from multiple providers
/multi-llm:multi-llm --review-plan plans/my-feature.md --models cursor-agent:auto gemini:gemini-2.5-flash opencode:opencode/big-pickle

# Models without prefix (uses default_provider: cursor-agent)
/multi-llm:multi-llm --review-plan plans/my-feature.md --models auto gpt-5.2-high

# Force interactive selection (overrides YAML defaults)
/multi-llm:multi-llm --review-plan plans/my-feature.md --interactive
```

## Critical Rules

These rules exist for specific technical reasons. Violating them causes failures that are hard to debug.

1. **Orchestrator execution mode** — The Claude Code Bash tool hard-caps `timeout` at **600000 ms (10 min)**; any larger value is silently clamped to 600000. Choose the execution mode by orchestrator category:

   - **Fast JSON-only orchestrators** (`--implement`, `--apply-suggestions`, `--apply-task-suggestions`, `--apply-code-fixes`) — run in the **FOREGROUND** with `timeout: 600000` (10 min). They only generate batch/task JSON and finish well under the cap, so foreground output (markers, paths) is returned directly.

   - **Fan-out review/ask modes** (`--review-plan`, `--review-tasks`, `--review-code`, `--ask`) — runtime scales with the number of models and ROUTINELY exceeds 10 min, so they MUST run **DETACHED** (`run_in_background: true`). Redirect both stdout and stderr to a log file inside the phase output dir (`> "<phase-dir>/orchestrator-run.log" 2>&1`) and set `PYTHONUNBUFFERED=1` so Python streams output to the log instead of block-buffering it (Python block-buffers when stdout is a non-TTY pipe). Detached runs are NOT subject to the 10-min Bash cap, so a slow model's own timeout governs. When the background task completes, read all markers (`[VALIDATION_PENDING]`, `[VALIDATION_BATCHES_PENDING]`, `[SALVAGE_NEEDED]`) and final output paths (e.g. `answers.md`) **from the log file**, not from terminal stdout.

   The orchestrators manage their own internal parallelism. Backgrounding WITHOUT redirection loses stdout (markers/paths) — which is exactly why redirection to a log file is mandatory for the detached modes. Resume semantics: re-invoking a review/ask orchestrator with `--force` RESUMES (keeps already-completed per-model result files, runs only the missing models, and bypasses the completed-phase/partial-completion guards); `--rerun-all` forces a full re-run that discards existing per-model results. A fresh re-run of an already-completed phase is `--force --rerun-all`.

2. **Subagent delegation** — When implementing tasks or applying fixes, use the Task tool with `subagent_type: "general-purpose"`. The orchestrators produce JSON instructions; Claude Code subagents execute them. Implementing manually bypasses the batching, routing, and tracking the orchestrators provide.

3. **Output tracking** — After completing work, update the appropriate output file so the next phase can pick up where this one left off:
   - `--implement`: Generate `{plan}/implement/summary.md` and update the plan with a reference
   - `--review-code --apply-fixes`: Update `{plan}/code-review/report.md` to mark issues as fixed

4. **Plan-local state** — State is stored in `{plan}/state.json`, not in `${CLAUDE_SKILL_DIR}/state/`. This keeps state portable and scoped to each plan.

## Execution Instructions

**CRITICAL**: Before executing any mode, Claude MUST:

1. **Parse the mode** from arguments:
   - `--review-plan` or no flag: review-plan mode (default)
   - `--generate-tasks`: generate-tasks mode
   - `--review-tasks`: review-tasks mode
   - `--apply-task-suggestions`: apply-task-suggestions mode
   - `--implement`: implement mode
   - `--review-code`: review-code mode
   - `--ask`: ask mode
   - `--full`: full workflow mode
   - `--init`: init-config mode (no plan path)

2. **Load the mode-specific instructions** using the Read tool:
   ```
   ${CLAUDE_SKILL_DIR}/instructions/{mode}.md
   ```

   Where `{mode}` is one of:
   - `review-plan`
   - `apply-suggestions`
   - `generate-tasks`
   - `review-tasks`
   - `apply-task-suggestions`
   - `implement`
   - `review-code`
   - `apply-code-fixes`
   - `ask`
   - `full-workflow`
   - `init-config`

3. **Execute the loaded instructions** step-by-step

### Mode Detection Logic

```
Arguments: /multi-llm:multi-llm [flags] <plan_path> [options]

If first arg starts with "--":
  - "--review-plan" -> mode = review-plan
  - "--apply-suggestions" -> mode = apply-suggestions
  - "--generate-tasks" -> mode = generate-tasks
  - "--review-tasks" -> mode = review-tasks
  - "--apply-task-suggestions" -> mode = apply-task-suggestions
  - "--implement" -> mode = implement
  - "--review-code" -> mode = review-code
  - "--apply-code-fixes" -> mode = apply-code-fixes
  - "--ask" -> mode = ask
  - "--full" -> mode = full-workflow
  - "--init" -> mode = init-config (no plan path; see instructions/init-config.md)
  - Plan path is second arg
Else:
  - mode = review-plan (default)
  - Plan path is first arg
```

#### Ask Mode Argument Parsing (must match `instructions/ask.md` exactly)

After `--ask`, the first positional token is the **plan path** and the next
**single (quoted) argument** is the question. The `--models` / `--quick` /
`--interactive` / `--force` flags are **position-independent** (they may appear
before the plan path, between plan path and question, or after the question).
Flag-like text inside the quoted question (e.g. `"keep --quick mode?"`) is
**question text, not a flag**. An **unquoted multi-word question** or an
**empty/whitespace-only question** is an **error** — emit the usage hint
(`/multi-llm:multi-llm --ask <plan_path> "<question>"`); do **not** silently join tokens
or no-op. The question must never be string-interpolated into the shell command
line — pass it to the orchestrator via `--question-file` (temp file, preferred),
`--question-env`, or a fully-escaped `--question` (see `instructions/ask.md`).

### Common Mistakes

- Backgrounding an orchestrator WITHOUT redirecting stdout/stderr to a log file — markers/paths are lost (see Rule 1 above); detached review/ask runs MUST redirect to `<phase-dir>/orchestrator-run.log`
- Setting a Bash `timeout` above 600000 ms — silently clamped to 600000 (10 min); long fan-out runs must be detached, not given a bigger timeout
- Writing implementation code directly instead of spawning Task subagents — bypasses routing, batching, and tracking
- Using model names not listed in `providers.yaml` — the orchestrator rejects unknown models
- Forgetting to report output file paths after execution — the user needs them to review results
- Manually searching for state files — the orchestrator loads them automatically from `{plan}/state.json`
- Skipping the instruction file read for the active mode — each mode has specific steps that vary significantly

---

## Summary Format

After any mode completes, provide a summary:

```
## [Mode] Complete

**Plan**: /path/to/plan.md

### Results
- [Mode-specific statistics]

### Output Files
- [List of generated files with full paths]

### State
- State file: [path if applicable]

### Next Steps
- `/clear` (recommended before starting the next phase, for best performance)
- [Recommended actions if any]
```
