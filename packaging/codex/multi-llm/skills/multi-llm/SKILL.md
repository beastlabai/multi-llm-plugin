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

Read the result's `exit` and `next` the way the dispatcher's own contract reads them; on `state: partial`, name the failed models and their reasons from `progress.models`; when the result carries `validation_counts` (a review phase's does), report them, and if `needs_human_decision` or `validation_failed` is non-zero point at `report.md` for those groups; they are decided later, when the suggestions are applied. A result without that key has no counts to report, so do not mention them. When the result carries a `notice` (`result.notice`, a finished run's: what it left out or could not check, such as items it did not apply or that nothing ran this repository's tests), report what it says, every part of it, as plain text.
On this host `next` arrives as arguments for this same tool, not as a command: "running" it means calling the tool again with exactly those arguments, never a shell.
Run `next` verbatim when it is not `null` on `0` and `2`. On `3` stop: report to the user which phase stopped, on which plan, the last error line it printed, whole (on a stop it says what stopped, what it reported, what did not run and whether anything ran the repository's tests), and the options its output names (`next` included); do not run a different mode. Do not perform the work this run was going to review, plan, or implement: doing it yourself is never a substitute for a failed run. Run an option only after the user has picked it. Never edit `driver.write_paths`, or any other multi-llm setting in `.multi-llm/providers.yaml`, yourself. At a stop, offer only the ways forward the stop itself names, as it words them: when it names a change to `driver.write_paths` that lets a refused file be written (adding it to `allow`, or removing or narrowing a deny rule that covers it), that is one of them and the user's call; add no advice about the write policy of your own. `user_next_step` is the phase after this one, for the user to type: never run it yourself; end your report with its `markdown`, copied as it is (each command stays alone in its fenced block, each why and note stays plain text, the manual tasks it lists included). Say "Workflow complete." only when it is `complete`, never while it names a `command`: the workflow goes on. It is never a tool call, unlike `next`, which is one.

If that tool is not listed, the plugin's server did not start: `codex mcp get multi_llm`
names the command it tried, and `python3 >= 3.9` must be on `PATH`.
