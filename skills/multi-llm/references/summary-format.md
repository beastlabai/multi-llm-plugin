# Standard Summary Format

The shape of the report a mode ends with. It moved out of the Claude Code shim in
the D5c cutover: what a run reports is *workflow* prose, identical on every host,
so it belongs in the core rather than in one host's entry file.

After any mode completes, report:

```
## [Mode] Complete

**Plan**: /path/to/plan.md

### Results
- [Mode-specific statistics]

### Output Files
- [Every generated file, with its full path]

### State
- State file: [path, if the mode has one]

### Next Steps
- `/clear` (recommended before the next phase, for best performance)
- [Recommended actions, if any]
```

Two rules:

- **Paths come from the run, never from memory.** Take them from the orchestrator's
  own output — the `[OUTPUT_FILE]` lines, the paths a phase printed, and
  `MULTI_LLM_RESULT`'s `artifacts` array. Do not reconstruct a path from a pattern.
- **Report the one follow-up the run owes, verbatim.** `MULTI_LLM_RESULT.next` is
  the command to run next when it is not `null`; copy it exactly, and report nothing
  as a next step that the run did not name.
