# Generate Tasks Mode Instructions

Generates detailed implementation tasks from a high-level plan. **Claude Code itself** analyzes the plan and codebase to create structured tasks.

## Usage

```bash
/multi-llm:multi-llm --generate-tasks <plan_path>
```

## Process (Claude Code executes these steps)

1. **Read the Plan**
   - Use Read tool to load the plan file
   - Identify: problem statement, proposed approach, any existing structure

2. **Explore the Codebase**
   - Use Glob to find relevant files mentioned in the plan
   - Use Grep to search for related code patterns
   - Use Read to understand existing implementations to integrate with

3. **Generate Tasks**
   - Break down the plan into 5-15 discrete tasks
   - Each task should be completable in 15-60 minutes
   - Identify dependencies between tasks
   - Specify files to create/modify for each task

4. **Output Tasks as JSON**
   - Generate JSON matching the schema below
   - Write it with the **Write tool** (NOT Bash) to the workspace temp dir:
     `{PROJECT_ROOT}/.multi-llm/tmp/generated_tasks_{plan_stem}.json`
   - Use unique filename based on plan name to avoid conflicts (full details in Step-by-Step Execution step 5)

5. **Update the Plan**
   - Call the helper script to insert tasks into the plan:
   ```bash
   uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/update_plan_tasks.py" --plan-file "$PLAN_PATH" --tasks-file "$PROJECT_ROOT/.multi-llm/tmp/generated_tasks_{plan_stem}.json"
   ```

## Task JSON Schema

```json
{
  "plan_preamble": "3-5 sentences summarizing the overall goal, key architectural decisions, technology choices, and global conventions",
  "tasks": [
    {
      "id": "T001",
      "title": "Short title (5-10 words)",
      "description": "Detailed description covering: (1) What to implement, (2) How — approach, patterns/helpers to reuse, (3) Interface contracts — what this task produces/consumes for other tasks",
      "depends_on": [],
      "files_to_modify": ["path/to/file.ts"],
      "files_to_create": ["path/to/new-file.ts"],
      "acceptance_criteria": ["Criterion 1", "Criterion 2"],
      "estimated_complexity": "low|medium|high",
      "subagent_type": "general-purpose|human"
    }
  ]
}
```

### Plan Preamble

The `plan_preamble` field provides essential context for subagents implementing tasks. Include:
- **Overall goal/outcome** - What the plan aims to achieve
- **Key architectural decisions** - Major design choices affecting implementation
- **Technology/library choices** - Specific frameworks, tools, or dependencies to use
- **Global conventions** - Patterns or standards to follow across all tasks

Keep it concise: 3-5 sentences, approximately 200 tokens. This context is included in every task prompt.

**subagent_type values:**
- `general-purpose`: Default for all implementation tasks
- `human`: Manual steps the user must perform (e.g., API key creation, external service config, manual testing)

## Guidelines for Task Generation

**Task Granularity**
- First task (T001) should be setup/foundation with no dependencies
- Each task should modify 1-5 files maximum
- Complex features should be broken into multiple tasks
- Tasks should be independently testable when possible
- Generate a `human` task when:
  - The plan explicitly states something must be done manually
  - The task requires access to external services/dashboards the agent cannot reach
  - The task requires physical or manual verification
- Human task descriptions should include clear step-by-step instructions since the user (not an agent) will read them
- Human task acceptance criteria should describe how the user can verify completion

**Dependencies**
- Use explicit `depends_on` arrays referencing task IDs
- Independent tasks can share the same dependencies (enables parallel execution)
- Avoid circular dependencies
- Tasks with no dependencies can run in parallel

**Codebase Exploration**
- Look for existing patterns to follow (search similar features)
- Identify integration points with existing code
- Check for related tests that need updates
- Note any shared utilities or components to reuse

**Task Descriptions**

Each task description must cover three components (2-5 sentences total):

1. **What** to implement — the concrete deliverable
2. **How** — key implementation approach, patterns/functions to reuse, APIs to call
3. **Interface contracts** — if this task produces or consumes interfaces, types, files, or data structures used by other tasks, state them explicitly

*Example — weak vs strong description:*

> **Weak:** "Add validation to the form component."
>
> This tells the subagent *what* at a surface level, but nothing about *how* or what interfaces are involved. The subagent must guess the validation approach, which helpers exist, and whether other tasks depend on its output.

> **Strong:** "Add Zod schema validation to UserProfileForm using the existing `validateWithSchema()` helper from `lib/validation.ts`. The schema should enforce email format, required fields, and max-length constraints matching the DB column limits in `T001`'s migration. Export the `UserProfileSchema` type from this file — `T003` (API handler) imports it for request body validation."
>
> This covers *what* (Zod validation on UserProfileForm), *how* (reuse `validateWithSchema()`, match DB constraints from T001), and *interface contracts* (exports `UserProfileSchema` for T003).

**Acceptance Criteria**

Acceptance criteria must be **task-scoped, specific and verifiable** — each criterion should describe an observable outcome that can be checked with a concrete test, command, or inspection. Avoid vague criteria like "works correctly" or "handles errors properly."

**Test scope rules**: Acceptance criteria may include running a **specific, narrowly-scoped test** (a single test file or test class) directly related to this task's changes, but must NOT include broad regression tests like "full test suite passes", "all tests pass", or "existing tests still pass." Full regression testing runs automatically once post-implementation — individual tasks should only verify their own changes.

*Example — bad vs good criteria:*

> **Bad criteria:**
> - "Form validation works correctly"
> - "Errors are handled properly"
> - "Performance is acceptable"
> - "Full test suite passes: uv run -- pytest"
> - "All existing tests still pass"
>
> These are not verifiable — what does "correctly" mean? What errors? What performance threshold? Broad test suite runs waste time and tokens when run per-task.

> **Good criteria:**
> - "Submitting the form with an empty email field shows the error message 'Email is required' below the input"
> - "`validateWithSchema()` returns `{ valid: false, errors: [...] }` when any required field is missing"
> - "The exported `UserProfileSchema` type is importable from `lib/validation.ts` and matches the column types in the `user_profiles` table migration"
> - "`uv run -- pytest tests/test_validation.py::TestUserProfileSchema -v` passes"

**File References**
- Use relative paths from project root
- Clearly distinguish between files to modify vs create
- Only reference files that actually need changes

## Output Format

Tasks are stored in a **separate file in a phase subdirectory** to keep the plan lean. The plan gets a reference:

**Plan file** (`my-plan.md`):
```markdown
# My Plan

... plan content ...

## Implementation Tasks

See [my-plan/tasks/tasks.md](my-plan/tasks/tasks.md) for the detailed task breakdown.

<!-- TASKS_FILE: my-plan/tasks/tasks.md -->
```

**Tasks file** (`my-plan/tasks/tasks.md`):
```markdown
# Implementation Tasks for my-plan.md

*Generated: 2026-01-08 12:00:00*

**Parent Plan**: [my-plan.md](../my-plan.md)

---

## Implementation Tasks

### Task T001: Create base configuration
**Dependencies**: None
**Files to create**: src/config.ts
**Complexity**: low

Description of the task...

**Acceptance Criteria**:
- [ ] Criterion 1
- [ ] Criterion 2

---
```

This approach keeps the plan file small so LLMs don't bloat their context with task details when editing the plan. All generated files for a plan are centralized in the plan's subfolder.

## Auto-Detection in Implement Mode

When running `--implement`, the orchestrator checks if the plan has tasks:
- If task file reference exists (`<!-- TASKS_FILE: ... -->`): proceeds with implementation
- If embedded tasks exist (legacy `## Task N:` format): proceeds with implementation
- If no tasks found: prints error directing user to run `--generate-tasks` first

---

## Step-by-Step Execution

### 0. Check Prerequisites

Before generating tasks, check if there are unapplied suggestions from plan review:

```bash
uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/check_workflow_prerequisites.py" --plan-file "$PLAN_PATH" --mode generate-tasks
```

Parse the JSON output. If `prerequisites_met` is `false`:

1. Use **AskUserQuestion** with the provided `prompt` content:
   - Question: Use `prompt.question`
   - Options: Map `prompt.options` to AskUserQuestion format

2. Handle the user's response:
   - **"Apply suggestions first"**: Stop and instruct the user to run in order:
     1. `/multi-llm:multi-llm --apply-suggestions $PLAN_PATH`
     2. `/clear`
     3. `/multi-llm:multi-llm --generate-tasks $PLAN_PATH`

     **Do NOT** proceed further. Stop execution after displaying this message.
   - **"Skip suggestions, proceed"**: Run `uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/apply_suggestions_orchestrator.py" --plan-file "$PLAN_PATH" --skip` to mark as skipped, then continue
   - **"Cancel"**: Stop and inform the user that generate-tasks was cancelled

If `prerequisites_met` is `true`, proceed with task generation.

---

1. **Validate plan file exists**

2. **Read the plan file:**
   ```
   Use Read tool to load the plan content
   ```

3. **Explore the codebase to understand context:**
   - Use Glob to find files matching patterns mentioned in the plan
   - Use Grep to search for related code, patterns, or implementations
   - Use Read to examine key files that will need modification

4. **Generate tasks based on your analysis:**
   - Create 5-15 discrete implementation tasks
   - Ensure first task has no dependencies
   - Identify which files each task will modify/create
   - Write descriptions covering what, how, and interface contracts (see Task Descriptions guidance above)
   - Define specific, verifiable acceptance criteria for each task (see Acceptance Criteria guidance above)

5. **Write tasks JSON to a temporary file using the Write tool (NOT Bash):**
   - Resolve the project root first: `PROJECT_ROOT=$(git rev-parse --show-toplevel)`.
     If this fails or prints nothing, STOP with the error: "multi-llm requires
     running inside a git repository." Never fall back to a relative path or `$PWD`.
   - file_path: `{PROJECT_ROOT}/.multi-llm/tmp/generated_tasks_{plan_stem}.json`
     (absolute path; the Write tool creates parent directories automatically —
     no mkdir step. Use plan stem in filename to avoid conflicts if multiple
     instances run.)
   - content: the tasks JSON document
   - If `{PROJECT_ROOT}/.multi-llm/tmp/.gitignore` does not exist yet, also Write
     it with content `*` so the temp dir ignores itself (run artifacts must never
     appear in `git status`).
   - Cleanup semantics: filenames are deterministic per plan, so each run simply
     overwrites the previous run's file. No deletion step exists or is needed;
     the user may delete `.multi-llm/tmp/` at any time — the next run recreates it.

6. **Run the update script:**
   ```bash
   uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/update_plan_tasks.py" --plan-file "$PLAN_PATH" --tasks-file "$PROJECT_ROOT/.multi-llm/tmp/generated_tasks_{plan_stem}.json"
   ```
   Pass `$PLAN_PATH` as given — the script resolves it to an OS-native absolute
   path itself (do NOT wrap it in `$(realpath ...)`; on Git for Windows that
   emits a POSIX `/c/...` path a native process cannot use).

7. **Report results:**
   - Path to generated tasks file (`{plan_subfolder}/{plan_stem}_tasks.md`)
   - Number of tasks generated
   - Summary of task dependencies
   - Any warnings from validation
   - Next step hint: "Run `/multi-llm:multi-llm --review-tasks $PLAN_PATH` to review tasks with multiple LLMs before implementation (optional)"

---

## Example JSON Output

```json
{
  "plan_preamble": "Implement a new feature for user data management. Use the existing service layer pattern with TypeScript. All database queries should use Supabase client. Follow the existing error handling conventions in lib/server.",
  "tasks": [
    {
      "id": "T001",
      "title": "Create database schema for feature",
      "description": "Create the user_features table with columns for id (uuid), user_id (FK to auth.users), feature_data (jsonb), and timestamps. Use the existing migration pattern from supabase/schemas/profiles.sql as a template. This table is read by T002's service layer and must have an index on user_id for the list-by-user query.",
      "depends_on": [],
      "files_to_create": ["apps/web/supabase/schemas/feature.sql"],
      "files_to_modify": [],
      "acceptance_criteria": [
        "Schema validates with supabase db lint with zero errors",
        "All foreign keys reference existing tables with ON DELETE CASCADE",
        "Index exists on user_id column for the user_features table"
      ],
      "estimated_complexity": "medium",
      "subagent_type": "general-purpose"
    },
    {
      "id": "T002",
      "title": "Implement service layer for feature CRUD",
      "description": "Create getFeature, listFeatures, createFeature, updateFeature, and deleteFeature functions following the pattern in lib/server/profiles/service.ts. Use the Supabase client from lib/supabase.ts and the error wrapper from lib/server/errors.ts. Export the FeatureService type so T003 (API routes) can import and use it.",
      "depends_on": ["T001"],
      "files_to_create": ["apps/web/lib/server/feature/service.ts"],
      "files_to_modify": [],
      "acceptance_criteria": [
        "All five CRUD functions (get, list, create, update, delete) are exported",
        "Each function wraps Supabase calls with handleServiceError() from lib/server/errors.ts",
        "FeatureService type is exported and importable by T003"
      ],
      "estimated_complexity": "medium",
      "subagent_type": "general-purpose"
    }
  ]
}
```
