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
- Start the next phase in a fresh context (recommended, for best performance) —
  use whatever context-reset command your host provides, and omit this line if it
  has none
- [Recommended actions, if any]
```

Every mode ends with this report. Three rules, and a fourth for `ask`:

- **Paths come from the run, never from memory.** Take them from the orchestrator's
  own output — the `[OUTPUT_FILE]` lines, the paths a phase printed, and
  `MULTI_LLM_RESULT`'s `artifacts` array. Do not reconstruct a path from a pattern.
- **Report the one follow-up the run owes, verbatim.** `MULTI_LLM_RESULT.next` is
  the command to run next when it is not `null`; copy it exactly, and report nothing
  as a next step that the run did not name.
- **Name every model that produced nothing, with its reason.** A review
  phase's `report.md` lists them on its `**Models failed:** N of M requested — …`
  line, directly under `**Models:**`, on the fan-out and after `--reaggregate`
  alike; a `--reaggregate` also prints one `[MODEL_FAILED] <model>: <reason>`
  line per requested model that still has no result. When
  `MULTI_LLM_RESULT.state` is `"partial"`, the same models and reasons are the
  `failed` rows of `progress.models` (a `status` poll carries them). List each
  one under Results with its reason; never report the run as if only the models
  that answered had been requested. **In `ask` mode** they are the
  `## Failed Models` section of `answers.md`. A failed model whose run left some
  text has it kept in a `partial_<model>.md` that its line names; say so, since
  a partial is not counted and a resume re-asks the model. You may offer
  `--accept-partial <model>` for a partial the user has read and judged whole —
  never run it unasked.
- **In `ask` mode, say how the answers were obtained.** Report the run's
  `answered: …` counts line verbatim, and the `[WARN]` it prints when every
  model asked to write an answer file answered on stdout instead. A host that
  reads only the result line takes both from `MULTI_LLM_RESULT.answer_origins`
  (its integers are the counts line's; `warn_all_fallback` is the warning).
