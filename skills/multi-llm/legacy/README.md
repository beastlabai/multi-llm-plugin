# `legacy/` — the pre-refactor instruction tree

This directory holds the **host-coupled instruction core exactly as it was before
the host-neutral rewrite**: the same 11 `instructions/*.md` and 5
`references/*.md` files, byte-for-byte, still naming `Task` /
`AskUserQuestion` / `run_in_background` / `${CLAUDE_SKILL_DIR}` and each
`*_orchestrator.py` directly.

It exists because the rewrite touched ~5k lines of battle-tested markdown, and the
golden regression guard (`tests/test_instruction_refactor_guard.py`) is necessary
but not sufficient: it proves the modes the guard fixture exercises, not the
interactive corners a real user hits on day two. So there is **a way back that does
not undo the Python work** — reverting the rewrite commit would also give back
every `${CLAUDE_SKILL_DIR}` site the dispatcher was built to delete.

## Using it

```bash
MULTI_LLM_LEGACY_INSTRUCTIONS=1 <your usual invocation>
```

With that variable set to `1`, `multi_llm.py util instructions <mode>` serves the
file from `legacy/instructions/` instead of `instructions/`, renders it against the
running skill dir exactly as it renders the current core (the renderer substitutes
every spelling of the placeholder, `${CLAUDE_SKILL_DIR}` included), and prints a
one-line notice to **stderr** so a transcript records which core was read.

Nothing else changes: same dispatcher, same modes, same exit codes, same
`MULTI_LLM_RESULT` line. The escape hatch is a *read-path* switch, not a second
implementation.

## Boundaries

- **Claude Code only.** The legacy text names Claude Code's tools, so it is a
  fallback for the reference host and for nothing else. A non-Claude host that sets
  the variable gets instructions naming tools it does not have.
- **Not maintained.** These files are frozen at the pre-refactor state. Fixes land
  in `instructions/`; nothing is back-ported here. One exception, forced by a
  Python change rather than a markdown one (2026-10-05): the write phases'
  orchestrators now record a phase *started* at emit and leave completion to the
  host's closing step, so `instructions/apply-suggestions.md` gained a closing
  `apply_suggestions_orchestrator.py --plan-file … --mark-completed` step (§8) and
  `instructions/implement.md`'s step 10 a closing
  `finalize_tracking.py --state-file … --mark-completed` call. Without them this
  core could no longer complete either phase; a write-path branch on the
  environment variable would have broken the "read path only" rule above
  instead. `tests/test_legacy_instructions.py` pins both.
- **Not linted, not counted.** The host-neutrality lint, the `${CLAUDE_SKILL_DIR}`
  manifest and the `{SKILL_DIR}` site inventory all scan `instructions/` and
  `references/` and deliberately skip this directory — a frozen copy that had to
  pass the new rules would not be a copy.
- **Expected lifetime: one release.** Once the reference host has run the rewritten
  core through the interactive corners the guard cannot reach, this directory and
  the environment variable go away together.
