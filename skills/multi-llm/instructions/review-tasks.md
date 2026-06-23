# Review Tasks Mode Instructions

Reviews generated implementation tasks using multiple LLMs, with plan coverage analysis as the primary concern.

## Usage

```bash
/multi-llm:multi-llm --review-tasks <plan_path> [model1 model2 ...] [--quick] [--interactive]
```

## Process

1. **Prerequisite Check**: Verify that `generate-tasks` phase is complete
2. **Model Selection**: Models provided as args, `--quick` for quick_models, or interactive selection via AskUserQuestion
3. **Run Reviews**: Execute the orchestrator with selected models
4. **Post-processing**: Group findings, validate (valid/invalid/needs-human-decision), handle human decisions
5. **Generate Report**: Create `report.html` with all findings

## Output Files

```
plans/todo/my-feature/                           # Output folder
├── state.json                                   # Session state (plan-local)
├── tasks/
│   └── tasks.md                                 # Generated tasks (input for this phase)
└── review-tasks/                                # Review tasks phase outputs
    ├── report.html                              # Interactive HTML report
    ├── report.md                                # Markdown review report
    ├── grouped.json                             # Grouped findings
    ├── validation.json                          # Validation results
    ├── {model}.json                             # Per-model results
    ├── log_{model}.txt                          # Agent output logs
    ├── error_{model}.log                        # Error logs (if failed)
    └── salvage_{model}.json                     # Salvage requests (if JSON parse failed)
```

## Options

- `--skip-validation`: Skip the LLM validation step (faster but no false-positive filtering)
- `--validation-model <model>`: Model to use for validation (default: "auto")
- `--quick` / `-q`: Use quick_models from providers.yaml for lightweight reviews (2 models)
- `--interactive` / `-i`: Force interactive model selection even when models are provided

## Validation Statuses

- **valid** (checkmark): Finding is real and should be addressed
- **invalid** (x): False positive, can be ignored
- **needs-human-decision** (?): Requires human judgment

When items have `needs-human-decision` status, Claude should use `AskUserQuestion` to get user input on how to proceed.

---

## Step-by-Step Execution

### 0. Check Prerequisites

Before reviewing tasks, verify that `generate-tasks` has been completed:

```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/check_workflow_prerequisites.py --plan-file "$(realpath "$PLAN_PATH")" --mode review-tasks
```

Parse the JSON output. If `prerequisites_met` is `false`:

- Use **AskUserQuestion** to inform the user that `generate-tasks` must run first.
- Options:
  - **"Run generate-tasks first"**: Stop and instruct the user to run:
    1. `/multi-llm:multi-llm --generate-tasks $PLAN_PATH`
    2. `/clear`
    3. `/multi-llm:multi-llm --review-tasks $PLAN_PATH`

    **Do NOT** proceed further. Stop execution after displaying this message.
  - **"Cancel"**: Stop and inform the user that review-tasks was cancelled.

If `prerequisites_met` is `true`, proceed to step 1.

Also verify the tasks file exists: check for `<!-- TASKS_FILE: ... -->` marker in the plan file, then confirm the referenced tasks file is present. If not found, stop and instruct the user to run `--generate-tasks` first.

---

### 1. Model Selection

- If models provided in args (e.g., `/multi-llm:multi-llm --review-tasks plans/foo.md gpt-5.2 gemini-3-pro`): use those directly
- If `--quick` flag is set: use `quick_models` from `providers.yaml`
- If `--interactive` flag is set OR no models provided: Claude MUST use AskUserQuestion to prompt user for model selection
  - Present available models from `providers.yaml` (multi-select)
  - Then run orchestrator with `--models <selected>`

### 2. Resume Detection (check BEFORE running orchestrator)

Check for existing output to avoid expensive duplicate runs. Follow the detection cases in `references/resume-detection.md`, using `{plan}/review-tasks/` as the phase directory.

**Note**: The orchestrator guards against re-runs (exits with code 2 if phase is already marked complete in state.json). Use `--force` to override. This instruction-level resume detection is the FIRST line of defense; the orchestrator guard is backup protection.

### 3. Run the Orchestrator

```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/review_tasks_orchestrator.py --plan-file "$(realpath "$PLAN_PATH")" --models <selected>
```

**IMPORTANT**: Always use `$(realpath "$PLAN_PATH")` to convert to absolute path.

**IMPORTANT**: Run this command in the FOREGROUND (do NOT use `run_in_background`). Use `timeout: 1200000` (20 minutes) to allow enough time for all models to complete. The orchestrator manages parallelism internally.

**TIMEOUT RECOVERY**: If this command times out or returns "(No output)", do NOT re-run. Go to step 4 (Timeout Recovery).

### 4. Timeout Recovery (conditional -- only if Bash returns "(No output)" or timeout)

If the orchestrator times out or returns "(No output)":

1. Read `{plan}/review-tasks/.status.json`
2. Based on `state` field: `models_running` -> check which `{model}.json` files exist; `models_complete`/`grouping_complete` -> check for `grouped.json`, `validation_tasks.json`; `validation_pending` -> proceed to validation; `complete` -> report results
3. If no `.status.json` -> fall back to file-based resume detection (step 2)
4. **Use AskUserQuestion** to inform the user which models completed/failed, offering: (a) Proceed with partial results, (b) Re-run missing models, (c) Re-run everything with longer timeout

### 5. Salvage Handling (Post-Orchestrator)

After the orchestrator completes, check for `[SALVAGE_NEEDED]` markers in the output. Follow the salvage process in `references/salvage-handling.md`.

The reaggregation command for this mode is:
```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/review_tasks_orchestrator.py \
  --plan-file "$(realpath "$PLAN_PATH")" --reaggregate
```

### 6. Validation Handling (Post-Orchestrator)

After the orchestrator completes, check for validation markers in the output.

#### Reference-Based Validation

The `validation_tasks.json` file uses a reference-based format (small metadata file):
```json
{
  "batches": [
    {
      "batch_index": 0,
      "group_indices": [0, 1, 2],
      "output_path": "{plan}/review-tasks/validation_batch_0.json"
    }
  ],
  "grouped_file": "{plan}/review-tasks/grouped.json",
  "plan_file": "/absolute/path/to/plan.md",
  "tasks_file": "/absolute/path/to/plan/tasks/tasks.md",
  "reaggregate_command": "uv run ... --reaggregate"
}
```

#### Single Batch: `[VALIDATION_PENDING]`

If `[VALIDATION_PENDING]` marker is present:
1. Read `validation_tasks.json` from `{plan}/review-tasks/` (small metadata file)
2. Spawn one validation subagent that reads the source files directly:

   ```
   Task tool call:
     subagent_type: general-purpose
     description: "Validate task review findings"
     prompt: |
       Validate finding groups from a task review.

       Read these files:
       - Grouped findings: {grouped_file from validation_tasks.json}
       - Plan context: {plan_file from validation_tasks.json}
       - Tasks file: {tasks_file from validation_tasks.json}

       Validate ONLY these groups (use group_hash from grouped.json to identify each):
       {for each i: "  index {group_indices[i]} (group_hash: {group_ids[i]})" from batches[N]}
       You MUST output EXACTLY {len(group_indices)} results -- one per requested group.
       Do NOT validate or include results for any other groups.

       For each group, determine:
       - "valid": Finding is real, should be addressed
       - "invalid": False positive, not applicable
       - "needs-human-decision": Requires human judgment

       Use the Write tool (not Bash) to save results to: {output_path from batches[N]}

       Output format:
       {
         "groups": [
           {"group_index": 0, "group_hash": "<copy from group's group_hash field>", "status": "valid|invalid|needs-human-decision", "reason": "...", "confidence": 0.0-1.0}
         ],
         "metadata": {"model": "claude", "timestamp": "ISO", "schema_version": "2.1"}
       }

       IMPORTANT: Copy the exact group_hash value from each group in grouped.json into your output.

       CRITICAL: Your entire response MUST be a single brief line, e.g.:
       "Done: wrote 3 validation results to /path/to/output.json"
       Do NOT include reasoning, analysis, or explanation in your response.
   ```

3. Wait for the subagent to complete
4. Run reaggregation using the command from `reaggregate_command` in validation_tasks.json

#### Multiple Batches: `[VALIDATION_BATCHES_PENDING]`

If `[VALIDATION_BATCHES_PENDING]` marker is present (for 5+ finding groups):
1. Read `validation_tasks.json` once from `{plan}/review-tasks/`
2. Count the number of batches as `{total_batches}` and mark the phase start:
   ```bash
   uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/metrics.py start \
     --state-file "$STATE_FILE" --phase "review-tasks" --total-batches {total_batches}
   ```
3. Choose a strategy:

**Strategy A: Direct parallel spawning (at most 4 batches)**

For each batch in the `batches` array, spawn a Task agent using the same prompt template as single-batch validation above, but with that batch's `group_indices` and `output_path`.

**Run all batches in parallel** (single message with multiple Task tool calls).

**Strategy B: Wave-based parallel spawning (> 4 batches)**

Follow the wave-based spawning strategy in `references/wave-batching.md`.

3. After ALL batches complete, run reaggregation using `reaggregate_command`

#### Reaggregation (After Validation)

After validation completes, run the `reaggregate_command` from validation_tasks.json:
```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/review_tasks_orchestrator.py \
  --plan-file "$(realpath "$PLAN_PATH")" --reaggregate
```

**CRITICAL**: Do NOT report results to the user until reaggregation is complete.

### 7. Report Results

Report all generated files in `{plan}/review-tasks/`:
- `report.html` - Interactive HTML report
- `report.md` - Markdown review report
- `grouped.json` - Grouped findings
- `validation.json` - Validation results

**Metrics (optional):** After validation/salvage subagents complete, record their metrics. When the validation phase used multiple batches, include the cumulative `--batch-index` (1..N across waves) and `--total-batches` so the script prints an `[ETA]` line:
```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/metrics.py record \
  --state-file "$STATE_FILE" --phase "review-tasks" \
  --label "Validation batch {N}" --subagent-type "general-purpose" \
  --tokens {token_count} --tool-uses {tool_uses} --duration-ms {duration_ms} \
  --total-batches {total_batches} --batch-index {N}
```
Where `token_count`, `tool_uses`, `duration_ms` come from the Task tool result. Omit any flags the Task tool didn't return. Single-batch validation runs may omit `--total-batches`/`--batch-index`.

After all batches and reaggregation finish, mark the phase finish:
```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/metrics.py finish \
  --state-file "$STATE_FILE" --phase "review-tasks"
```

Optionally generate and include resource usage report:
```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/utils/metrics.py report --state-file "$STATE_FILE" --phase "review-tasks"
```

#### High-Volume Results (>25 groups)

If the review produces more than 25 finding groups, note to the user: *Consider using importance filtering (`--approve-importance LOW`) to auto-approve low-importance findings and focus review on HIGH and MEDIUM items.* This guidance is informational only -- do not prompt interactively for filtering options.

#### Applying Findings

Findings from task review can be applied automatically or manually:

> **Next step:** `/multi-llm:multi-llm --apply-task-suggestions $PLAN_PATH`

The `--apply-task-suggestions` phase reads the validated findings and user selections, then uses subagents to update `tasks.md` automatically (additions, modifications, deletions, and clarifications).

**Manual application (alternative):** If you prefer to apply findings by hand, review the report and update `tasks.md` directly:
- **Add missing tasks**: If plan coverage gaps are identified, add new tasks to `tasks.md`
- **Fix dependencies**: Update `depends_on` arrays if dependency issues are found
- **Refine descriptions**: Improve task descriptions where ambiguity or incompleteness was flagged
- **Split/merge tasks**: Restructure tasks if granularity issues were identified

#### HIGH-Importance Plan Coverage Findings

For findings with **HIGH importance** and type `plan_coverage` (missing plan sections not covered by any task), provide actionable next steps:

1. **Re-run `--generate-tasks`**: If multiple plan sections are uncovered, regenerate tasks:
   ```
   /multi-llm:multi-llm --generate-tasks $PLAN_PATH --force
   ```
   Review the new task list to confirm coverage.

2. **Manual patching**: For 1-2 missing items, manually add tasks to `tasks.md` following the existing `### Task T0XX:` format with dependencies, files, complexity, description, and acceptance criteria.

3. **Proceed anyway**: If the uncovered sections are intentionally deferred or out of scope, document the decision in the plan file as a "Deferred Items" section and proceed to `--implement`.

**Note**: The `--review-tasks` phase is optional. Users can proceed directly to `--implement` after `--generate-tasks` if they prefer to skip task review.

---

## Example Execution

```
User: /multi-llm:multi-llm --review-tasks plans/my-feature.md

Claude:
1. Checks prerequisites: generate-tasks is complete
2. No models in args -> AskUserQuestion for model selection
3. User selects: gpt-5.2, gemini-3-pro
4. Runs orchestrator with --models gpt-5.2 gemini-3-pro
5. Orchestrator outputs [VALIDATION_BATCHES_PENDING] (8 groups -> 2 batches)
6. Reads validation_tasks.json, Strategy A (2 batches): spawns 2 parallel subagents
7. Subagents write validation_batch_0.json, validation_batch_1.json
8. Runs reaggregation
9. Reports: 8 findings, 5 valid, 2 invalid, 1 needs-human-decision
   "Next step: /multi-llm:multi-llm --apply-task-suggestions plans/my-feature.md"
```
