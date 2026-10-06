---
name: multi-llm
description: "Orchestrate plan reviews, task generation, implementation and code reviews across several LLM CLIs in parallel, or ask every model one free-text question about a plan. Use when the user says: review my plan, multi-llm, run a code review, generate tasks from a plan, implement this plan, ask the models about my plan."
---

# Multi-LLM

Call the `multi_llm` tool with the argument text, converted to its four properties exactly
as that tool's own argument descriptions specify. The tool is the only supported entry point
on this host: it appends the flags this host requires, refuses the ones it owns, and
supervises the run to a terminal result. Do not reach the dispatcher any other way. The tool
runs outside the exec sandbox, and on the modes that apply changes it writes to the workspace. A call can run for up to its 15-minute budget. When you call this tool from an `exec` cell, start the cell with `// @exec: {"yield_time_ms": 900000}`, so the cell waits for the result. If a cell hands back "Script running" anyway, call `wait` until this call returns, and never end your turn while it is running.

Read the result's `exit` and `next` the way the dispatcher's own contract reads them; on `state: partial`, name the failed models and their reasons from `progress.models`; when the result carries `validation_counts` (a review phase's does), report them, and if `needs_human_decision` or `validation_failed` is non-zero point at `report.md` for those groups; they are decided later, when the suggestions are applied. A result without that key has no counts to report, so do not mention them.
On this host `next` arrives as arguments for this same tool, not as a command: "running" it means calling the tool again with exactly those arguments, never a shell.
Run `next` verbatim when it is not `null` on `0` and `2`. On `3` stop: report to the user which phase was running on which plan, the last error line it printed, whole (on a stop it says what stopped, what it reported, what did not run and whether anything ran the repository's tests), and the options its output names (`next` included); do not run a different mode. Do not perform the work this run was going to review, plan, or implement: doing it yourself is never a substitute for a failed run. Run an option only after the user has picked it. `user_next_step` is the phase after this one, for the user to type: never run it yourself; end your report with its `command`, and each `optional_before` command, every one in its own fenced block, verbatim: the `command` first, then the line "Optional, before that:" and each `optional_before` entry's `why` as plain text above its command. When it carries `manual_tasks`, complete or not, also list each (id and title) and give its `manual_tasks_note` verbatim, as plain text outside any fenced block (a fenced block holds one command and nothing else): it names the command that records them done. It is never a tool call, unlike `next`, which is one.

If that tool is not listed, the plugin's server did not start: `codex mcp get multi_llm`
names the command it tried, and `python3 >= 3.9` must be on `PATH`.
