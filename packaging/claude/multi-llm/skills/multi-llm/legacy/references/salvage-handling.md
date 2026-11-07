# Salvage Handling

When a model's review output cannot be parsed as JSON, the orchestrator writes a
salvage *request* (`salvage_{model}.json`) and, beside it, a **materialized
salvage task** (`salvage_task_{model}.json`) — printing
`[SALVAGE_NEEDED] {request_path}` and `[SALVAGE_TASK] {task_path}`.

**For each task in the task file, run `task.prompt` as a subagent, then write the
JSON it returns to `task.output_path`.** That is the whole contract; the prompt
is assembled in Python (`utils/subagent_prompts.build_salvage_prompt`) and
carries the request's `raw_output`, `expected_type` and `expected_schema` inline,
so a runner with no workspace access can serve it.

Two steps that used to live in this prompt are the invoker's, not the subagent's:

1. **Back up before overwriting.** Immediately before writing `task.output_path`,
   run `utils/backup.py` on it; an existing file is copied to
   `{filename}-BEFORE-SALVAGE-{timestamp}.json`.
2. **Delete the salvage request** after reaggregation, not after extraction — the
   request is what the task's `inputs.salvage_request_sha256` binds to.

After ALL salvage tasks complete, run reaggregation. The orchestrator prints a
`[REAGGREGATE_AFTER_SALVAGE]` marker with the exact command — copy and run it.
This is mandatory: reaggregation regenerates grouped.json, validation.json and
the report files to include the salvaged data. Do NOT report results to the user
until reaggregation is complete.
