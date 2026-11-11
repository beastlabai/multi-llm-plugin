# Repository map for agents

The source of truth is `skills/multi-llm/` (the skill itself) with `packaging/`
(builders and release tooling) and `docs/` beside it. Start with
[`skills/multi-llm/AGENTS.md`](skills/multi-llm/AGENTS.md); it routes each topic
to the page under `docs/development/`.

## Generated trees: build output, not source

`packaging/claude/multi-llm/` and `packaging/codex/multi-llm/` are committed
byte copies of the skill, built from `skills/multi-llm/`, `hooks/` and the
packaging templates. A symbol found there is a copy of a symbol found in the
source; edit the source, then regenerate with `make claude-regen codex-regen`
and commit the result with the source edit. Never edit the trees by hand.

The root `.ignore` hides both trees from `rg` and from Claude Code's Grep tool,
so a search returns source hits only. `git grep` does not read it, Claude Code's
Glob bypasses it by default (unless `CLAUDE_CODE_GLOB_NO_IGNORE=false`), and no
build step reads it. To search the trees on purpose, pass `rg --no-ignore`.
