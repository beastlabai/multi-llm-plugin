---
name: implement
description: Implement a plan from a markdown file using subagents, then update the plan to reflect actual implementation
disable-model-invocation: true
argument-hint: <plan-path>
---

# Implement Plan

Implement the plan at: `$ARGUMENTS`

## Workflow

1. **Read the plan** at the given path. Understand all requirements, phases, success criteria, and technical details.

2. **Implement using subagents** — use the Agent tool to parallelize independent work:
   - Break the plan into logical tasks that can be worked on concurrently
   - Launch subagents for independent tasks (e.g., different files, different services, different features)
   - For dependent tasks, run them sequentially
   - Each subagent should be given clear, self-contained instructions with all necessary context from the plan

3. **Run tests** after implementation to verify everything works:
   ```
   source .venv/bin/activate && PYTHONPATH=src/app pytest tests/ -v
   ```

4. **Update the plan file** — compare what was actually implemented vs what the plan described:
   - Add an "Implementation Notes" section at the bottom of the plan
   - Document any deviations, simplifications, or improvements made during implementation
   - Mark completed items
   - Note any items that were skipped or deferred, and why
   - Keep the original plan content intact — only append/annotate

## Guidelines

- Read the entire plan before starting any implementation
- Prefer editing existing files over creating new ones
- Follow existing code patterns and conventions in the codebase
- If the plan is ambiguous, make a reasonable choice and document it in the update
- If a task fails or is blocked, document it and continue with other tasks
- Do not create git commits unless the plan specifically asks for it
