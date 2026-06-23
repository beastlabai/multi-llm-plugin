# Implement Mode Instructions

Executes implementation tasks from a plan using Claude Code's native Task tool.

## Usage

```bash
/multi-llm:multi-llm --implement <plan_path> [--resume] [--task T001] [--dry-run] [--yes] [--no-confirm] [--force-strategy <strategy>]
```

## Options

- `--resume`: Resume from previous session state (skip completed tasks)
- `--task T001`: Execute only a specific task
- `--dry-run`: Show task summary without executing
- `--yes` / `--no-confirm`: Skip all advisory prompts (e.g., unapplied task suggestions) and proceed silently
- `--force-strategy <strategy>`: Skip the human task strategy prompt and use the given strategy directly. Valid values: `pause-and-ask`, `skip-continue`, `skip-dependents`. Useful for scripted/non-interactive runs.

## Process

1. **Generate Task JSON**: The orchestrator parses the plan and outputs task definitions to JSON
2. **Claude Code Reads JSON**: Claude Code reads the task definitions
3. **Execute Batches**: Claude Code executes each batch sequentially using its Task tool
4. **Parallel Execution**: Tasks within a batch can run in parallel (no dependencies between them)
5. **State Updates**: Claude Code updates the state file after each task completes
6. **Generate Implementation Summary**: After all tasks complete, Claude Code creates `{plan}_implementation_summary.md`
7. **Update Plan**: Add reference to the implementation summary in the original plan
8. **Report Results**: Summary of completed/failed tasks

## Output Files

After implementation, files are organized into phase-based subdirectories:

```
plans/my-feature.md                                     # Original plan (updated with summary reference)
plans/my-feature/                                       # Output folder
├── state.json                                          # Session state (plan-local)
├── tasks/
│   └── tasks.md                                        # Task breakdown (from --generate-tasks)
└── implement/
    └── summary.md                                      # Implementation summary
```

The implementation summary (`{plan}/implement/summary.md`) contains:
- Summary of what was implemented
- List of all files created/modified
- Any deviations from the planned tasks
- Task completion status
- Useful context for the code review phase

## Subagent Routing

Each task includes a `subagent_type` field:

| subagent_type | Usage |
|---------------|-------|
| `general-purpose` | All implementation tasks (default) |
| `human` | Manual steps (API keys, config, manual testing) |

These are the only subagent types available in Claude Code for implementation work.

## State Management

State is persisted in `{plan}/state.json` (plan-local):

- Tracks task completion status (pending, in_progress, completed, failed)
- Records files modified per task
- Stores git HEAD at session start for diff comparison
- Enables resume after interruption
- Automatically migrates from old hash-based location on first access

---

## Critical Subagent Delegation Rules

**MANDATORY**: These rules MUST be followed:

1. **ALWAYS use the Task tool**: When implementing tasks, ALWAYS use the Task tool with `subagent_type: "general-purpose"`. NEVER implement code manually.

2. **OUTPUT TRACKING**: After completing tasks:
   - Update state file with task status (completed/failed)
   - Generate `{plan}_implementation_summary.md` with results
   - Update the original plan file to reference the implementation summary

---

## Step-by-Step Execution

### 0. Check Prerequisites

The implement orchestrator checks prerequisites and may output special markers. Handle them as follows:

#### If output contains `[PREREQUISITE_CHECK]`:

There are unapplied suggestions from plan review. Parse the JSON that follows the marker.

1. Use **AskUserQuestion** to prompt the user:
   - "Review found N suggestions. Apply them before implementation?"
   - Options: "Apply suggestions first", "Skip suggestions, proceed", "Cancel"

2. Handle the response:
   - **"Apply suggestions first"**: Stop and instruct the user to run this in a separate context:
     ```
     ## Apply Suggestions First

     To keep the implementation context clean, please run in order:

     1. Run: `/multi-llm:multi-llm --apply-suggestions $PLAN_PATH`
     2. Run: `/clear`
     3. Run: `/multi-llm:multi-llm --implement $PLAN_PATH`
     ```
     **Do NOT** proceed further. Stop execution after displaying this message.

   - **"Skip suggestions, proceed"**: Run `uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/apply_suggestions_orchestrator.py --plan-file "$PLAN_PATH" --skip`, then re-run implement orchestrator (this is lightweight — no context bloat)
   - **"Cancel"**: Stop and inform user

#### If output contains `[TASKS_MISSING]`:

No implementation tasks exist. To avoid context bloat from codebase exploration, do NOT auto-run generate-tasks in this context. Instead, stop and instruct the user to run it separately.

**Output the following to the user:**

```
## Tasks Missing

No implementation tasks found for this plan. To keep the implementation context clean,
please run task generation in a separate context:

1. Run: `/multi-llm:multi-llm --generate-tasks $PLAN_PATH`
2. Run: `/clear`
3. Run: `/multi-llm:multi-llm --implement $PLAN_PATH`

The `/clear` step ensures the implementation phase starts with a fresh context,
free from the codebase exploration data generated during task creation.
```

**Do NOT** proceed further. Stop execution after displaying this message.

#### If output contains `[TASK_SUGGESTIONS_ADVISORY]`:

There are unapplied task suggestions from `review-tasks`. This is an **advisory only** — it does NOT block implementation. Parse the JSON that follows the marker to get the `actionable_count`.

**If `--yes` or `--no-confirm` was passed**: Skip the prompt silently and proceed with implementation. Do NOT prompt the user.

**Otherwise**, use **AskUserQuestion** to prompt the user:
- "There are N unapplied task suggestions from review-tasks. Apply them first? [y/N]"
- Options: "Apply task suggestions first", "Skip, proceed to implement" (default), "Cancel"

Handle the response:
- **"Apply task suggestions first"**: Stop and instruct the user to run this in a separate context:
  ```
  ## Apply Task Suggestions First

  To keep the implementation context clean, please run in order:

  1. Run: `/multi-llm:multi-llm --apply-task-suggestions $PLAN_PATH`
  2. Run: `/clear`
  3. Run: `/multi-llm:multi-llm --implement $PLAN_PATH`
  ```
  **Do NOT** proceed further. Stop execution after displaying this message.

- **"Skip, proceed to implement"** (default): Run `uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/apply_task_suggestions_orchestrator.py --plan-file "$(realpath "$PLAN_PATH")" --skip` to mark the phase as skipped, then continue with implementation.

- **"Cancel"**: Stop and inform user.

**Important**: The `[TASK_SUGGESTIONS_ADVISORY]` marker does NOT cause the orchestrator to exit early. The orchestrator continues and produces the full task JSON output. Check for this marker in the complete output alongside the other markers.

#### If output contains no blocking markers (`[PREREQUISITE_CHECK]` or `[TASKS_MISSING]`):

Prerequisites are met. If `[TASK_SUGGESTIONS_ADVISORY]` was present, handle it as described above, then proceed with normal implementation flow.

---

1. **Validate plan file exists**

2. **Run the orchestrator to generate task JSON:**
   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/implement_orchestrator.py --plan-file "$(realpath "$PLAN_PATH")" --output /tmp/implementation_tasks.json [--resume]
   ```

   **IMPORTANT**: Always use `$(realpath "$PLAN_PATH")` to convert to absolute path.

   **IMPORTANT**: Run this command in the FOREGROUND (do NOT use `run_in_background`). Use `timeout: 600000` (10 minutes — the Bash tool caps `timeout` at 600000 ms; larger values are silently clamped). The orchestrator runs quickly — it only generates task JSON, it does not execute tasks — so 10 minutes is far more than enough.

3. **Check for prerequisite markers in orchestrator output** (see Step 0 above)

4. **Read the task JSON file:**
   ```
   Use Read tool to load /tmp/implementation_tasks.json
   ```

#### 4a. Human Task Strategy (if applicable)

After reading the task JSON, scan all batches for tasks with `is_human: true`. If any are found, determine the strategy:

**Strategy resolution order:**

1. **`--force-strategy` flag**: If provided, use that strategy directly (no prompt).
2. **`--resume` with saved strategy**: On resume, load the saved strategy from state.json (`human_task_strategy` field). If present, use it without re-prompting.
3. **Prompt the user**: Use `AskUserQuestion` to ask the user how to handle them:

- Question: "This plan contains N human task(s) requiring manual action (e.g., {list first 2-3 titles}). How should these be handled?"
- Options:
  1. **"Pause and ask me" (Recommended)** — Stop at each human task, show what needs to be done, and wait for confirmation before continuing. Dependent tasks block until confirmed.
  2. **"Skip all, continue anyway"** — Skip all human tasks but continue executing all dependent code tasks as normal. List human tasks in the implementation summary for the user to complete later.
  3. **"Skip all, skip dependents too"** — Skip all human tasks AND skip any code tasks that depend on them. List everything skipped in the implementation summary.
  4. **"Cancel"** — Stop implementation.

**Persist the strategy**: After resolving (whether from flag, saved state, or user prompt), save it to state.json:
```python
state = json.load(state_file)
state["human_task_strategy"] = "pause-and-ask"  # or "skip-continue", "skip-dependents"
state["updated_at"] = datetime.now().isoformat()
json.dump(state, state_file)
```
This ensures `--resume` can recover the strategy without re-prompting.

5. **Execute batches sequentially (batches have dependencies):**

   Before the loop, count the total number of tasks across all batches as `{total_tasks}` and mark the phase start:
   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/metrics.py start \
     --state-file "$STATE_FILE" --phase "implement" --total-batches {total_tasks}
   ```

   For each batch in `batches` array:
   - Execute all tasks in the batch (can be parallel if `can_parallelize` is true)
   - For each task, spawn a Task subagent:
     ```
     Task tool call:
     - subagent_type: task["subagent_type"]  (e.g., "general-purpose")
     - description: task["title"]
     - prompt: task["prompt"]
     ```

   - **Human task handling** (tasks with `is_human: true`):

     **"Pause and ask me" strategy:**
     - Do NOT spawn a Task subagent for human tasks
     - Use `AskUserQuestion` to present the task:
       - Question: "Task {id}: {title} requires manual action. Please complete the following and confirm when done:\n\n{description}\n\nAcceptance criteria:\n{acceptance_criteria}"
       - Options: "Done — I've completed this task", "Skip this task", "Cancel implementation"
     - Handle each response:
       - **Done**: Mark task as "completed" in state, continue
       - **Skip**: Mark task as "skipped" in state with reason "skipped by user during pause-and-ask". Run the **transitive skip propagation** algorithm (see below) to skip dependent tasks.
       - **Cancel**: Execute **cancellation semantics** (see below)
     - Human tasks within a parallelizable batch: execute all non-human tasks in parallel first, then present human tasks sequentially

     **"Skip all, continue anyway" strategy:**
     - Mark all human tasks as "skipped" in state with reason "deferred — user chose to complete manually after implementation"
     - Treat human task dependencies as satisfied — dependent code tasks proceed as normal
     - For each code task that has a skipped human task in its dependency chain, record a dependency override in state:
       ```python
       state["dependency_overrides"] = state.get("dependency_overrides", {})
       state["dependency_overrides"][code_task_id] = {
           "overridden_deps": [human_task_id_1, human_task_id_2],
           "timestamp": datetime.now().isoformat()
       }
       ```
     - After all tasks complete, include a "## Manual Tasks (Deferred)" section in the implementation summary
     - Also include a "## Dependency Override Warnings" section listing each code task that ran with overridden human dependencies:
       ```markdown
       ## Dependency Override Warnings

       The following tasks ran with skipped human-task dependencies (skip-continue strategy):

       | Task | Overridden Dependencies | Risk |
       |------|------------------------|------|
       | T003: Build API client | T002: Create API key | Task may fail at runtime without API key |
       | T005: Deploy service | T004: Configure DNS | Service may not be reachable |
       ```

     **"Skip all, skip dependents too" strategy:**
     - Mark all human tasks as "skipped" in state with reason "deferred — user chose to complete manually after implementation"
     - Run the **transitive skip propagation** algorithm to find and skip all dependent tasks
     - After remaining tasks complete, include a "## Manual Tasks (Deferred)" section listing each human task and each skipped dependent

     #### Transitive Skip Propagation Algorithm

     When a human task is skipped (either via "skip dependents too" strategy or individual skip during "pause-and-ask"), propagate the skip forward through the dependency graph:

     1. Start with the set of skipped human task IDs as the **seed set**
     2. Use BFS forward walk: for each task in the seed set, find all tasks that have it in their `depends_on`
     3. Mark each discovered dependent as "skipped" with reason: "skipped — depends on skipped task {parent_id} ({parent_title})"
     4. Add the newly skipped task to the queue and continue until no more dependents are found
     5. Track the full skip chain for reporting: `T001 (human, skipped) → T003 (skipped, depends on T001) → T005 (skipped, depends on T003)`

     #### Cancellation Semantics

     When the user chooses "Cancel" (either at the strategy prompt or during pause-and-ask):

     1. Mark all already-completed tasks as "completed" (preserve their status)
     2. Mark all not-yet-attempted tasks as "pending" (NOT "skipped" — they were never reached)
     3. Mark the current task (if any) as "pending"
     4. Do NOT mark the implement phase as completed (`mark_phase_completed("implement")` is NOT called)
     5. Write a partial implementation summary with a "## Cancelled" header:
        ```markdown
        ## Cancelled

        Implementation was cancelled by user after completing {N} of {total} tasks.

        ### Completed Tasks
        | Task | Status |
        |------|--------|
        | T001: Setup DB | Completed |

        ### Remaining Tasks (not attempted)
        | Task | Status |
        |------|--------|
        | T002: Create API key | Pending |
        | T003: Build feature | Pending |
        ```
     6. The partial summary allows `--resume` to pick up where it left off

   - After each task completes, update the state file:
     ```python
     # Load state file (path from JSON's "state_file" field)
     state = json.load(state_file)
     state["task_status"][task_id] = "completed"  # or "failed"
     state["updated_at"] = datetime.now().isoformat()
     json.dump(state, state_file)
     ```
   - **Record metrics**: Run the metrics utility to store Task tool metrics:
     ```bash
     uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/metrics.py record \
       --state-file "$STATE_FILE" --phase "implement" \
       --label "{task_id}: {task_title}" --subagent-type "{subagent_type}" \
       --tokens {token_count} --tool-uses {tool_uses} --duration-ms {duration_ms} \
       --total-batches {total_tasks} --batch-index {N}
     ```
     Where `token_count`, `tool_uses`, `duration_ms` come from the Task tool result, and `{N}` is the cumulative 1-based task index across all batches. Omit any flags the Task tool didn't return. The script prints an `[ETA]` line to stderr after each call.

6. **Handle errors:**
   - If a task fails, mark it as "failed" in state
   - Continue with independent tasks (skip tasks that depend on failed ones)
   - Log the error for reporting

7. **Finalize file tracking:**
   After all tasks complete (before generating summary), mark the phase finish and run finalization:
   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/metrics.py finish \
     --state-file "$STATE_FILE" --phase "implement"
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/finalize_tracking.py --state-file "$STATE_FILE"
   ```
   Where `$STATE_FILE` is the `state_file` path from the JSON output.

   This deterministically computes which files were changed during implementation
   by comparing current git changes against pre-existing changes recorded at startup.
   The tracked files are stored in state and used by the code review phase to focus
   only on implementation changes (ignoring unrelated uncommitted changes).

8. **Run post-implementation regression tests:**
   After all tasks complete, run the project's test suite **once** to catch any integration issues:

   - If plan modifies `${CLAUDE_SKILL_DIR}/` files:
     ```bash
     uv run --project ${CLAUDE_SKILL_DIR} -- pytest ${CLAUDE_SKILL_DIR}/tests/ -v --tb=short 2>&1 | tail -50
     ```
   - Otherwise, detect common test runners:
     - If `package.json` exists with a `test` script: `npm test 2>&1 | tail -50`
     - If `pytest.ini`, `pyproject.toml` with `[tool.pytest]`, or `conftest.py` exists: `pytest -v --tb=short 2>&1 | tail -50`
   - Record pass/fail results for inclusion in the implementation summary
   - Do NOT block on failure — record the results and continue to summary

9. **Generate implementation summary:**
   After all tasks complete, create the implementation summary file using the `summary_file` path from the JSON output.
   Use `summary_file_relative` for links within the file (relative to plan's parent directory).

   Write the summary to the path specified in `summary_file` from the JSON:
   ```markdown
   # Implementation Summary: {plan_name}

   *Generated: {timestamp}*

   **Plan**: [{plan_name}](../{plan_name})
   **Tasks**: [{tasks_file_relative}](./{tasks_file_name})

   ## Summary

   Brief description of what was implemented...

   ## Task Results

   | Task | Status | Notes |
   |------|--------|-------|
   | T001: Task title | Completed | ... |
   | T002: Task title | Completed | ... |

   ## Files Modified

   - `path/to/file1.ts` - Description of changes
   - `path/to/file2.ts` - Description of changes

   ## Files Created

   - `path/to/new-file.ts` - Description

   ## Test Results

   {Pass/fail results from regression test run, or "No test runner detected"}

   ## Deviations from Plan

   List any changes that deviated from the original task plan...
   (Or "None - implementation followed the plan exactly.")

   Before writing the summary, generate the resource usage section:
   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/metrics.py report \
     --state-file "$STATE_FILE" --phase "implement"
   ```
   Include the output (if non-empty) in the summary before "## Notes for Code Review".

   ## Notes for Code Review

   Any important context for the code review phase...
   ```

10. **Update the original plan:**
   Add a reference to the implementation summary in the original plan using the absolute `plan_file` path and the `summary_file_relative` path from the JSON:
   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python -c "
   from pathlib import Path
   import sys
   sys.path.insert(0, '${CLAUDE_SKILL_DIR}')
   from utils import insert_implementation_summary_reference
   plan_path = Path('$PLAN_FILE')  # Use absolute plan_file from orchestrator JSON
   content = plan_path.read_text()
   summary_file = '$SUMMARY_FILE_RELATIVE'  # Use summary_file_relative from JSON
   updated = insert_implementation_summary_reference(content, summary_file)
   plan_path.write_text(updated)
   print(f'Updated {plan_path} with reference to {summary_file}')
   "
   ```

11. **Report results** including:
   - Number of tasks completed/failed
   - Implementation summary file location
   - State file location
   - Any errors encountered
   - Files modified
   - Resource usage summary (if metrics were captured)

**IMPORTANT:** Claude Code orchestrates the execution by spawning Task subagents. Do NOT manually implement the code yourself - delegate to subagents via the Task tool.

---

## Example Execution

```
User: /multi-llm:multi-llm --implement plans/my-feature.md

Claude:
1. Runs: uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/implement_orchestrator.py --plan-file "$(realpath plans/my-feature.md)" --output /tmp/tasks.json
2. Orchestrator outputs:
   - /tmp/tasks.json (5 tasks in 3 batches)
   - State file: ${CLAUDE_SKILL_DIR}/state/abc123.json
   - Summary file path: plans/my-feature/my-feature_implementation_summary.md
   - Tasks file path: plans/my-feature/my-feature_tasks.md
   - Summary file relative: my-feature/my-feature_implementation_summary.md
3. Claude reads /tmp/tasks.json
4. For each batch, Claude spawns Task subagents:
   - Batch 0 (1 task): Task(subagent_type="general-purpose", prompt="Create database schema...")
   - Batch 1 (2 tasks, parallel):
     - Task(subagent_type="general-purpose", prompt="Implement service layer...")
     - Task(subagent_type="general-purpose", prompt="Create form component...")
   - Batch 2 (2 tasks, parallel):
     - Task(subagent_type="general-purpose", prompt="Add API routes...")
     - Task(subagent_type="general-purpose", prompt="Write E2E tests...")
5. After each task, Claude updates state file with status
6. After all tasks complete, Claude generates:
   - plans/my-feature/my-feature_implementation_summary.md (summary of what was done)
7. Claude updates the original plan to link to the summary using relative path
8. Reports: 5 tasks completed, summary file, state file, files modified
```
