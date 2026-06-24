# Full Workflow Mode Instructions

Runs all modes in sequence for complete automation.

## Usage

```bash
/multi-llm:multi-llm --full <plan_path> [model1 model2 ...] [--quick] [--yes]
```

## Process

1. **Phase 1: Review Plan** (with specified models)
2. **Phase 2: Apply Suggestions** (apply validated suggestions to the plan)
3. **Phase 3: Generate Tasks** (automatic task generation)
3b. **Phase 3b: Review Tasks** (optional — review generated tasks with multiple LLMs)
3c. **Phase 3c: Apply Task Suggestions** (auto-apply validated task review findings to tasks)
4. **Phase 4: Implement** (automatic execution)
5. **Phase 5: Review Code** (automatic review)
6. **Phase 6: Apply Code Fixes** (auto-apply validated code-review findings to the code)

## How Full Mode Works

Full mode orchestrates the entire workflow by loading and executing each phase's instructions in sequence. This provides end-to-end automation from plan review through implementation and code review.

## Non-Interactive Mode (`--yes` / `--non-interactive`)

By default, full mode still stops to ask you a few questions (model selection, `needs-human-decision` items, whether to review tasks). Passing **`--yes`** (alias **`--non-interactive`**) makes the entire workflow run **fully unattended — zero AskUserQuestion prompts** — by applying these effects to every phase:

| Phase | Default (interactive) | With `--yes` |
|-------|----------------------|--------------|
| Model selection (Phase 1 & 5) | Prompts if no models given | Never prompts — uses `--models` if given, else `--quick` if given, else `providers.yaml` defaults |
| Phase 2 Apply Suggestions | Prompts on `needs-human-decision` + no-selection confirmation | Runs with `--claude-decide --no-confirm` (Claude decides every item; no confirmation) |
| Phase 3b Review Tasks | Prompts "review the tasks?" | Runs review-tasks **automatically** (no prompt) |
| Phase 3c Apply Task Suggestions | `--no-confirm` already | Adds `--claude-decide` (Claude decides every item) |
| Phase 6 Apply Code Fixes | Prompts on `needs-human-decision` + no-selection confirmation | Runs with `--claude-decide --no-confirm` |

`--yes` is the umbrella switch; the narrower flags (`--quick`, `--claude-decide`, `--no-confirm`) keep their individual meanings and can still be passed on their own. Throughout the steps below, **"if `--yes` was specified"** means apply the row above for that phase. If `--yes` is NOT set, follow the interactive behavior as written.

---

## Step-by-Step Execution

### 1. Handle Model Selection (same as review-plan mode)

- If models provided in args: use those directly
- If `--yes` (or `--non-interactive`) was specified: **never prompt.** Resolve models without asking — use `--models` if given, else pass `--quick` if given, else pass no model flag at all (the orchestrator falls back to `providers.yaml` defaults). Skip the AskUserQuestion below entirely.
- If NO models provided (and NOT `--yes`): Claude MUST use AskUserQuestion to prompt user for model selection
  - Present available models from `providers.yaml` (multi-select)

### 2. Execute Phase 1: Review Plan

Load instructions from `${CLAUDE_SKILL_DIR}/instructions/review-plan.md` and execute.

```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/review_plan_orchestrator.py --plan-file "$(realpath "$PLAN_PATH")" --models <selected>
```

If `--quick` was specified, pass `--quick` to the orchestrator instead of `--models`:
```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/review_plan_orchestrator.py --plan-file "$(realpath "$PLAN_PATH")" --quick
```

Wait for completion, then proceed to Phase 2.

### 2b. Conditional Consolidation (Between Phase 1 and Phase 2)

After Phase 1 completes (including any validation and reaggregation), check the orchestrator output for the `[CONSOLIDATION_RECOMMENDED]` marker.

**When present** — auto-run consolidation (NO user prompt in full workflow mode):

1. Run the consolidation orchestrator:
   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/review_plan_orchestrator.py --plan-file "$(realpath "$PLAN_PATH")" --consolidate
   ```

2. Handle subagent batches using the same Strategy A/B logic from review-plan instructions:
   - `[CONSOLIDATION_PENDING]`: single batch — spawn one Task subagent
   - `[CONSOLIDATION_BATCHES_PENDING]`: multiple batches — Strategy A (≤ 4: parallel direct) or Strategy B (> 4: wave-based spawning in waves of 4, last wave absorbs remainder)
   - Load `consolidation_tasks.json` from `{plan}/review-plan/`
   - Read prompt template from `${CLAUDE_SKILL_DIR}/prompts/consolidate_suggestions.txt`
   - Each subagent instruction: "Respond with ONLY the JSON output. No explanation."

3. Run reaggregation:
   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/review_plan_orchestrator.py --plan-file "$(realpath "$PLAN_PATH")" --reaggregate-consolidation
   ```

4. Present consolidated report paths to user:
   - `{plan}/review-plan/consolidated.json`
   - `{plan}/review-plan/consolidated-report.md`
   - `{plan}/review-plan/consolidated-report.html`

**When absent** — skip consolidation silently and proceed directly to Phase 2.

### 3. Execute Phase 2: Apply Suggestions

Load instructions from `${CLAUDE_SKILL_DIR}/instructions/apply-suggestions.md` and execute.

```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/apply_suggestions_orchestrator.py --plan-file "$(realpath "$PLAN_PATH")"
```

If `--yes` (or `--non-interactive`) was specified, append `--claude-decide --no-confirm` so the phase runs fully unattended (Claude judges every `needs-human-decision` item, salvaging partially-valid ones, and the no-selection confirmation is skipped):
```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/apply_suggestions_orchestrator.py --plan-file "$(realpath "$PLAN_PATH")" --claude-decide --no-confirm
```

This phase applies validated suggestions sequentially:
- Load validation results from Phase 1
- Filter to valid suggestions (HIGH/MEDIUM importance)
- Apply each suggestion one at a time using Task subagents
- Handle needs-human-decision items via AskUserQuestion (or, under `--yes`, via Claude's autonomous "Let Claude Decide" judgment — no prompts)
- Generate applied suggestions report

Wait for completion, then proceed to Phase 3.

### 4. Execute Phase 3: Generate Tasks

**Context isolation**: This phase does extensive codebase exploration that would bloat the main context. Delegate it to a Task subagent so the main workflow context stays lean for Phase 4 (Implement).

**Note**: In full workflow mode, Phase 2 already handled suggestions, so the prerequisite check (Step 0 of generate-tasks) is skipped. The subagent also cannot use AskUserQuestion, so no interactive prompts are included.

Spawn a single Task subagent:

```
Task tool call:
  subagent_type: general-purpose
  description: "Generate implementation tasks from plan"
  prompt: |
    You are generating implementation tasks for a plan.

    **Plan file**: {PLAN_PATH_ABSOLUTE}

    ## Steps

    1. Read the plan file using the Read tool
    2. Explore the codebase to understand context:
       - Use Glob to find files matching patterns mentioned in the plan
       - Use Grep to search for related code, patterns, or implementations
       - Use Read to examine key files that will need modification
    3. Generate 5-15 discrete implementation tasks:
       - First task (T001) should be setup/foundation with no dependencies
       - Each task should modify 1-5 files maximum
       - Identify dependencies between tasks (use explicit depends_on arrays)
       - Define clear acceptance criteria for each task
       - Set appropriate subagent_type:
         - general-purpose: All implementation tasks (default)
         - human: Manual steps requiring user action
    4. Write the tasks JSON to a temporary file:
       cat > /tmp/generated_tasks_{plan_stem}.json << 'EOF'
       {
         "plan_preamble": "3-5 sentences: overall goal, architecture, tech choices, conventions",
         "tasks": [
           {
             "id": "T001",
             "title": "Short title (5-10 words)",
             "description": "Detailed description (2-5 sentences)",
             "depends_on": [],
             "files_to_modify": [],
             "files_to_create": [],
             "acceptance_criteria": [],
             "estimated_complexity": "low|medium|high",
             "subagent_type": "general-purpose"
           }
         ]
       }
       EOF
    5. Run the update script:
       uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/update_plan_tasks.py \
         --plan-file "{PLAN_PATH_ABSOLUTE}" \
         --tasks-file /tmp/generated_tasks_{plan_stem}.json
    6. Report: number of tasks generated, task IDs with titles, and the tasks file path.

    ## Constraints
    - Do NOT use AskUserQuestion — you are running as a subagent
    - Do NOT skip codebase exploration — thorough analysis produces better tasks
    - Do NOT run any prerequisite checks (Step 0) — already handled
```

Where `{PLAN_PATH_ABSOLUTE}` is `$(realpath "$PLAN_PATH")` and `{plan_stem}` is the plan filename without extension.

**Error handling**: After the subagent completes:

1. **Verify success**: Read the plan file and check for the `<!-- TASKS_FILE: ... -->` marker
2. **If the subagent failed**: Report the error and ask the user whether to:
   - Retry (spawn another subagent)
   - Run generate-tasks manually in main context (fallback: load `${CLAUDE_SKILL_DIR}/instructions/generate-tasks.md` and execute)
   - Cancel the full workflow

Wait for completion, then proceed to Phase 3b.

### 4b. Optional Phase 3b: Review Tasks

After task generation completes successfully, determine whether to run the review-tasks phase.

**`--yes` / `--non-interactive` mode**: Run `review-tasks` **automatically without prompting**, using the same models resolved in Phase 1 (or `--quick` / `providers.yaml` defaults). Load instructions from `${CLAUDE_SKILL_DIR}/instructions/review-tasks.md` and execute. Wait for completion, then proceed to Phase 3c. (Do NOT skip and do NOT prompt.)

**No-TTY automation without `--yes`**: When running in a non-interactive context (no TTY) and `--yes` was NOT passed, skip `review-tasks` by default without prompting. To opt in during such runs, the user must pass `--review-tasks` (or `--yes`) explicitly on the CLI. If skipping without prompt, run:

```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/check_workflow_prerequisites.py --plan-file "$(realpath "$PLAN_PATH")" --mode review-tasks --skip --reason "Non-interactive mode: skipped by default"
```

Then proceed directly to Phase 3c.

**Interactive mode (no `--yes`)**: Use AskUserQuestion to prompt:

> "Would you like to review the generated tasks with multiple LLMs before implementation?"

- **Options**: "Yes, review tasks" / "No, proceed to implement"

**If yes**: Load instructions from `${CLAUDE_SKILL_DIR}/instructions/review-tasks.md` and execute. Use the same models selected in Phase 1. Wait for completion, then proceed to Phase 3c.

**If no**: Mark the phase as skipped for workflow tracking so `--status` output and rerun logic never show review-tasks as perpetually pending:

```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/check_workflow_prerequisites.py --plan-file "$(realpath "$PLAN_PATH")" --mode review-tasks --skip --reason "User declined"
```

Then proceed to Phase 3c.

### 4c. Execute Phase 3c: Apply Task Suggestions

After Phase 3b completes (or is skipped), determine whether to run the apply-task-suggestions phase. This phase applies validated findings from the task review to update the tasks file automatically.

**Three-case conditional logic:**

**(a) Review-tasks was skipped**: If Phase 3b was skipped (user declined or non-interactive mode), skip apply-task-suggestions as well. Record the skip:

```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/check_workflow_prerequisites.py --plan-file "$(realpath "$PLAN_PATH")" --mode apply-task-suggestions --skip --reason "Skipped: review-tasks was not executed"
```

Then proceed directly to Phase 4.

**(b) Review-tasks ran but produced zero findings**: Check `{plan}/review-tasks/grouped.json`. If the file exists but contains zero groups (empty `groups` array or all groups have zero valid suggestions), skip apply-task-suggestions:

```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/check_workflow_prerequisites.py --plan-file "$(realpath "$PLAN_PATH")" --mode apply-task-suggestions --skip --reason "Skipped: review-tasks produced zero findings"
```

Then proceed directly to Phase 4.

**(c) Review-tasks produced findings**: If `{plan}/review-tasks/grouped.json` contains valid findings, auto-run apply-task-suggestions:

```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/apply_task_suggestions_orchestrator.py --plan-file "$(realpath "$PLAN_PATH")" --no-confirm
```

If `--yes` (or `--non-interactive`) was specified, also append `--claude-decide` so any `needs-human-decision` task suggestions are judged by Claude rather than prompted:
```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/apply_task_suggestions_orchestrator.py --plan-file "$(realpath "$PLAN_PATH")" --no-confirm --claude-decide
```

This phase:
- Loads validated task suggestions from the review-tasks phase
- Filters to valid suggestions (HIGH/MEDIUM importance)
- Applies each suggestion to the tasks file using Task subagents
- Generates an applied task suggestions report (`{prefix}_applied_task_suggestions.md`)
- Produces `orchestrator_output.json` in `{plan}/apply-task-suggestions/`

Follow the full apply-task-suggestions instruction file (`${CLAUDE_SKILL_DIR}/instructions/apply-task-suggestions.md`) to process batches, generate the summary report, and mark the phase completed. Then proceed to Phase 4.

### 5. Execute Phase 4: Implement

Load instructions from `${CLAUDE_SKILL_DIR}/instructions/implement.md` and execute.

```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/implement_orchestrator.py --plan-file "$(realpath "$PLAN_PATH")"
```

Wait for completion, then proceed to Phase 5.

### 6. Execute Phase 5: Review Code

Load instructions from `${CLAUDE_SKILL_DIR}/instructions/review-code.md` and execute.

```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/code_review_orchestrator.py --plan-file "$(realpath "$PLAN_PATH")" --models <selected>
```

If `--quick` was specified, pass `--quick` to the code review orchestrator:
```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/code_review_orchestrator.py --plan-file "$(realpath "$PLAN_PATH")" --quick
```

Use the **same models resolved in Phase 1** (Critical Rule 4). Under `--yes`, resolve models the same non-interactive way as Phase 1 (never prompt). This phase only *reviews* the code and produces `{plan}/code-review/` outputs — the fixes are applied in Phase 6.

Wait for completion, then proceed to Phase 6.

### 6b. Execute Phase 6: Apply Code Fixes

Load instructions from `${CLAUDE_SKILL_DIR}/instructions/apply-code-fixes.md` and execute. This phase mirrors Phase 2 (Apply Suggestions), but applies the validated findings from the code review (Phase 5) to the **code** rather than the plan.

**Conditional — skip if there is nothing to apply.** Check `{plan}/code-review/grouped.json`. If it does not exist, or contains zero groups / zero valid (HIGH/MEDIUM) findings, skip this phase:

```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/check_workflow_prerequisites.py --plan-file "$(realpath "$PLAN_PATH")" --mode apply-code-fixes --skip --reason "Skipped: code review produced zero actionable findings"
```

Then proceed to the report.

**Otherwise, run the apply-code-fixes orchestrator:**

```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/apply_code_fixes_orchestrator.py --plan-file "$(realpath "$PLAN_PATH")"
```

If `--yes` (or `--non-interactive`) was specified, append `--claude-decide --no-confirm` so the phase runs fully unattended:
```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/apply_code_fixes_orchestrator.py --plan-file "$(realpath "$PLAN_PATH")" --claude-decide --no-confirm
```

This phase:
- Loads validated code-review findings from Phase 5
- Filters to valid findings (HIGH/MEDIUM importance)
- Applies each fix to the code one at a time using Task subagents
- Handles `needs-human-decision` items via AskUserQuestion (or, under `--yes`, via Claude's autonomous "Let Claude Decide" judgment — no prompts)
- Generates an applied code fixes report and overlays decisions onto `{plan}/code-review/report.html`

Follow the full apply-code-fixes instruction file to process batches, generate the summary report, and mark the phase completed. Wait for completion, then proceed to the report.

### 7. Report Combined Results

After all phases complete, provide a comprehensive summary:

```markdown
## Full Workflow Complete

**Plan**: /path/to/plan.md

### Phase 1: Review Plan
- Models used: [list]
- Suggestions found: N
- Valid suggestions: N
- Invalid (filtered): N

### Phase 2: Apply Suggestions
- Suggestions applied: N
- Human decisions made: N
- Suggestions skipped: N
- Applied suggestions report: {plan}_applied_suggestions.md

### Phase 3: Generate Tasks
- Tasks generated: N
- Tasks file: {plan}_tasks.md

### Phase 3b: Review Tasks *(only if executed, omit if skipped)*
- Issues found: N
- Importance breakdown: HIGH: N, MEDIUM: N, LOW: N
- Report: {plan}/review-tasks/report.html

### Phase 3c: Apply Task Suggestions *(only if executed, omit if skipped)*
- Suggestions applied: N
- Suggestions skipped: N
- Applied task suggestions report: {prefix}_applied_task_suggestions.md
- Orchestrator output: {plan}/apply-task-suggestions/orchestrator_output.json

### Phase 4: Implement
- Tasks completed: N/N
- Files modified: [list]
- Implementation summary: {plan}_implementation_summary.md

### Phase 5: Review Code
- Issues found: N
- Importance breakdown: HIGH: N, MEDIUM: N, LOW: N
- Review report: {plan}/code-review/report.html

### Phase 6: Apply Code Fixes *(only if executed, omit if skipped)*
- Fixes applied: N
- Decided by Claude: N (A approved, V salvaged, S skipped) *(under `--yes`)*
- Fixes skipped: N
- Applied code fixes report: {prefix}_applied_code_fixes.md

### All Output Files
- [List all generated files with full paths]
- Include `{prefix}_applied_task_suggestions.md` and `{plan}/apply-task-suggestions/orchestrator_output.json` (if Phase 3c ran)

   Generate combined resource usage:
   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/metrics.py report \
     --state-file "$STATE_FILE" --all-phases
   ```
   Include the output (if non-empty) after "### All Output Files".

### Next Steps
- [Any recommended follow-up actions]
```

---

## Critical Rules for Full Mode

1. **Sequential Execution**: Each phase MUST complete before the next begins
2. **Error Handling**: If a phase fails, report the error and ask user whether to continue. Under `--yes`, "ask whether to continue" still applies for hard failures — `--yes` suppresses *workflow decision* prompts (model choice, human-decision items, review-tasks opt-in), not crash recovery.
3. **State Preservation**: Each phase's state is preserved for the next phase
4. **Same Models**: Use the same model selection for both review-plan and review-code phases
5. **Apply Sequential**: In Phases 2, 3c, and 6, process items ONE AT A TIME to avoid conflicts
6. **`--yes` is all-or-nothing for prompts**: When `--yes` / `--non-interactive` is set, NO AskUserQuestion may be issued for normal workflow decisions in any phase — resolve models non-interactively, pass `--claude-decide --no-confirm` to every apply phase, and run review-tasks automatically. See the Non-Interactive Mode table above.

## Planned Features (Not Yet Implemented)

The following CLI options are planned but not yet available:

- `--task-timeout <seconds>`: Per-task timeout (default: 1200)
- `--continue-on-failure`: Continue with independent tasks after failure
- `--ignore-dependency <task-id>`: Skip dependency check for a task
- `--no-lsp`: Disable LSP-based file discovery
- `--cleanup-state`: Delete state files after completion

These features are documented in the implementation plan and will be added in future releases.
