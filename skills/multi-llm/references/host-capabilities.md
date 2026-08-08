<!-- GENERATED FILE — DO NOT EDIT.
     Rendered from hosts/capabilities.yaml by hosts/render_capabilities.py.
     Edit the YAML and re-render; CI fails if this file differs from a fresh
     render, so a hand-edit cannot ship. -->

# Host capabilities

Human-readable view of `hosts/capabilities.yaml`. **Nothing reads this file at run
time** — not an orchestrator, not a shim, and above all not a driving agent. The
YAML is consumed by `multi_llm.py`, by `install-host` and by the tests; this
document exists for people reading the repo.

Every value carries a **provenance**: `verified` means a recorded live smoke run
observed it, `documented` means host docs only, `unknown` is the shipping default.
A `documented`-only value never selects the non-conservative path — which is why
hosts whose native subagent dispatch is merely documented ship as
`subagents: executor`, and why a documented command-timeout ceiling is worked
around with `--detach` rather than trusted.

The three **capability verbs** are the spine of each entry. The host-neutral
instruction core says *spawn subagents* / *ask the user* / *run detached*; the
columns below say what each resolves to, and a shim bakes that resolution in as a
sentence when it is installed.


| Host | Shim (repo → installed) | `arguments` | *spawn subagents* | *ask the user* | *run detached* | Default runner | `nested_exec` (spawn/network/creds) | Max subagents | Timeout ceiling | Verified |
|---|---|---|---|---|---|---|---|---|---|---|
| `claude-code` | *not rendered* → `SKILL.md` | `placeholder` | `native` (`Task`) — verified | `structured` (`AskUserQuestion`) — verified | `native` (`run_in_background`) — verified | `claude-code:sonnet` | allowed/allowed/allowed | 10 | 600 s (raise via not raisable) | `arguments`, `ask`, `default_subagent_runner`, `detach`, `installable`, `multi_turn`, `native_tool`, `subagent_tiers`, `subagents` |
| `codex` | `hosts/codex/shim.md.tmpl` → `SKILL.md` | `placeholder` | `executor` (`agent`) — documented | `free-text` — documented | `python` — documented | `codex:gpt-5.2-codex` | unknown/unknown/unknown | 4 | none documented | *none yet* |
| `opencode` | `hosts/opencode/shim.md.tmpl` → `commands/multi-llm.md` | `placeholder` | `executor` (`task`) — documented | `free-text` — documented | `required` — documented | `opencode:opencode/sonnet` | unknown/unknown/unknown | 4 | 120 s (raise via env: `unknown`, config: `unknown`) | *none yet* |

## Per-host notes and stripped variables

### `claude-code`

- Installable: `false`; reload after install: `none`; multi-turn session: `true`
- Subagent tiers: full `sonnet`, cheap `haiku`
- Verb sentences a rendered shim carries:
  - **Spawn subagents** — call your host's own subagent tool `Task`: pass `task.prompt` verbatim, take the role from `task.subagent_type` when your tool needs one, and write the JSON it returns to `task.output_path`. Never pass `--exec-subagents` on this host.
  - **Ask the user** — call your host's own structured question tool `AskUserQuestion`, then copy the selected option's command verbatim. Never compose a flag from the user's words.
  - **Run detached** — use your host's own background mechanism `run_in_background`, redirecting stdout and stderr to the phase dir's `orchestrator-run.log`. Never pass `--detach` on this host.
- Install paths (project): `.claude/skills/multi-llm`
- Install paths (global): `$HOME/.claude/skills/multi-llm`
- Install paths (legacy): *none*
- Runners:
  - `claude-code`: bundled=`true`, needs_auth=`false`, profiles=[generate, agent-readonly, agent-write], usage=`full`
  - `cursor-agent`: bundled=`false`, needs_auth=`true` (`cursor-agent login`), profiles=[generate], usage=`none`
- Refuse-to-start guards stripped from a dispatched runner: `CLAUDECODE`, `CLAUDE_CODE_ENTRYPOINT`
- Session-scoping variables stripped: `CLAUDE_SESSION_ID`, `CLAUDE_SKILL_DIR`
- Sanctioned exceptions that survive stripping: `ANTHROPIC_API_KEY`
- Nested-exec notes: Native subagents run in-session; the executor is not on the default path for this host.

### `codex`

- Installable: `true`; reload after install: `new-session`; multi-turn session: `true`
- Subagent tiers: resolved from `providers.yaml` `driver.subagent_runner{,_cheap}` (this host does not dispatch subagents natively)
- Verb sentences a rendered shim carries:
  - **Spawn subagents** — run the dispatcher with `--exec-subagents`: it dispatches every materialized task itself and continues past each marker in the same process. Never spawn a subagent by hand on this host.
  - **Ask the user** — print the numbered option block exactly as the dispatcher rendered it and copy the selected option's command verbatim, and only under `--interactive`. Otherwise the documented `--yes` default applies and nothing is asked.
  - **Run detached** — pass `--detach` (which implies `--yes`) and poll `status <plan> --view run`; the foreground stays legal for short calls. Never combine `--detach` with a host background mechanism — pick one.
- Install paths (project): `.codex/skills/multi-llm`
- Install paths (global): `$HOME/.codex/skills/multi-llm`
- Install paths (legacy): `$HOME/.codex/prompts`
- Runners:
  - `claude-code`: bundled=`false`, needs_auth=`true` (`claude login`), profiles=[generate, agent-readonly, agent-write], usage=`full`
  - `codex`: bundled=`true`, needs_auth=`false`, profiles=[generate, agent-readonly, agent-write], usage=`full`
- Refuse-to-start guards stripped from a dispatched runner: *none recorded* — the env diff has not run yet
- Session-scoping variables stripped: *none recorded* — the env diff has not run yet
- Sanctioned exceptions that survive stripping: *none recorded* — the env diff has not run yet
- Nested-exec notes: Filled by the Phase 3 nested-exec smoke: --exec-subagents launched from inside `codex exec`'s own sandboxed shell, for both the same-host runner and a third-party one.

### `opencode`

- Installable: `true`; reload after install: `new-session`; multi-turn session: `false`
- Subagent tiers: resolved from `providers.yaml` `driver.subagent_runner{,_cheap}` (this host does not dispatch subagents natively)
- Verb sentences a rendered shim carries:
  - **Spawn subagents** — run the dispatcher with `--exec-subagents`: it dispatches every materialized task itself and continues past each marker in the same process. Never spawn a subagent by hand on this host.
  - **Ask the user** — print the numbered option block exactly as the dispatcher rendered it and copy the selected option's command verbatim, and only under `--interactive`. Otherwise the documented `--yes` default applies and nothing is asked.
  - **Run detached** — pass `--detach` (which implies `--yes`) on every fan-out mode and poll `status <plan> --view run`. This host caps a single shell command below a fan-out's runtime, so no foreground fan-out path is offered.
- Install paths (project): `.opencode/commands`
- Install paths (global): `$XDG_CONFIG_HOME/opencode/commands`, `$HOME/.config/opencode/commands`
- Install paths (legacy): `.opencode/command`
- Runners:
  - `claude-code`: bundled=`false`, needs_auth=`true` (`claude login`), profiles=[generate, agent-readonly, agent-write], usage=`full`
  - `opencode`: bundled=`true`, needs_auth=`false`, profiles=[generate, agent-readonly, agent-write], usage=`full`
- Refuse-to-start guards stripped from a dispatched runner: *none recorded* — the env diff has not run yet
- Session-scoping variables stripped: *none recorded* — the env diff has not run yet
- Sanctioned exceptions that survive stripping: *none recorded* — the env diff has not run yet
- Nested-exec notes: Filled by the Phase 3 nested-exec smoke from inside `opencode run`.
