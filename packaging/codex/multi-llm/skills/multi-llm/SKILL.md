---
name: multi-llm
description: "Orchestrate plan reviews, task generation, implementation and code reviews across several LLM CLIs in parallel, or ask every model one free-text question about a plan. Use when the user says: review my plan, multi-llm, run a code review, generate tasks from a plan, implement this plan, ask the models about my plan."
---

# Multi-LLM

Call the `multi_llm` tool with the argument text, converted to its four properties exactly
as that tool's own argument descriptions specify. The tool is the only supported entry point
on this host: it appends the flags this host requires, refuses the ones it owns, and
supervises the run to a terminal result. Do not reach the dispatcher any other way. The tool
runs outside the exec sandbox, and on the modes that apply changes it writes to the workspace.

Read the result's `exit` and `next` the way the dispatcher's own contract reads them; on `state: partial`, name the failed models and their reasons from `progress.models`; on exit 0, report `validation_counts`, and if `needs_human_decision` or `validation_failed` is non-zero point at `report.md` for those groups; they are decided later, when the suggestions are applied.
On this host `next` arrives as arguments for this same tool, not as a command: "running" it means calling the tool again with exactly those arguments, never a shell.
Run `next` verbatim when it is not `null` on `0` and `2`. On `3` stop: report to the user what was being run, the last error line it printed, and the options its output names (`next` included); do not run a different mode. Do not perform the work this run was going to review, plan, or implement: doing it yourself is never a substitute for a failed run. Run an option only after the user has picked it.

If that tool is not listed, the plugin's server did not start: `codex mcp get multi_llm`
names the command it tried, and `python3 >= 3.9` must be on `PATH`.
