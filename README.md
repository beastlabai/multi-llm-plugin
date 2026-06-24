# multi-llm - a Claude Code plugin

**Wisdom of crowds for your codebase.** Use this plugin to improve planning, implementation, and code review by running the same work through multiple AI coding tools and LLMs in parallel - then consolidating their feedback so you catch bugs, blind spots, and improvements a single model might miss.

Orchestrate **plan reviews, task generation, implementation, and code reviews across multiple LLM providers in parallel** - or ask every model the same free-text question about a plan and get one consolidated answer.

`multi-llm` fans a single workflow out to several CLI-based LLMs ([Cursor Agent](https://cursor.com/docs/cli/overview), [Gemini CLI](https://geminicli.com/docs/get-started/), [Codex](https://developers.openai.com/codex/cli), [OpenCode](https://opencode.ai/docs/cli/), [Kilocode](https://kilo.ai/docs/cli), and [Claude Code](https://code.claude.com/docs/en/) itself), validates and consolidates their suggestions, and hands Claude Code structured instructions to apply the results. The orchestrators never modify your code directly - they produce JSON that Claude Code executes through its own tools, so every change stays reviewable.

---

## Installation

In Claude Code:

```text
/plugin marketplace add beastlabai/multi-llm-plugin
/plugin install multi-llm@beastlabai
```

Then invoke the skill (plugin skills are namespaced `plugin:skill`):

```text
/multi-llm:multi-llm plans/my-feature.md
```

> Claude can also load the skill automatically when you ask it to "review my plan with multiple models", "run a multi-LLM code review", etc.

### Configure providers (required before first use)

Edit [`skills/multi-llm/providers.yaml`](skills/multi-llm/providers.yaml) manually to match the code harnesses and models you want to use. The shipped file is a starting point only - keep the providers whose CLIs you have installed, remove the rest, and set `defaults.models` (and optionally `quick_models`) to the `provider:model` pairs you actually run. Without this step, default invocations will call harnesses or models you may not have configured. See [Providers](#providers) for the format and available keys.

> **Where is the file after installing via `/plugin`?** If you installed the plugin (rather than cloning this repo), `providers.yaml` lives inside the installed plugin, under your Claude Code plugins directory - typically `~/.claude/plugins/<marketplace>/multi-llm/skills/multi-llm/providers.yaml`. The quickest way to open it is to ask Claude Code to "open the multi-llm `providers.yaml`": the skill resolves its own install location (via `${CLAUDE_SKILL_DIR}`), so Claude can read and edit the file in place.

### Prerequisites

| Requirement | Why |
| --- | --- |
| [`uv`](https://docs.astral.sh/uv/) on your `PATH` | The orchestrators run as `uv run` Python scripts. uv installs their dependencies (just `pyyaml`) on first use. |
| One or more provider CLIs | You only need the CLIs for the models you actually run. See [Providers](#providers). |

The plugin ships its own `pyproject.toml`/`uv.lock`, so no manual Python setup is required - `uv` resolves the environment the first time a mode runs.

---

## Plans and outputs

A **plan** is just a markdown file describing the change you want to make - a feature spec, refactor outline, or design doc. You write it (or have Claude draft it), then point multi-llm at it. There's no required template; the more concrete the plan, the more useful the reviews. Most workflows start from a plan, with one exception: `--ask` only needs a plan to give the models context for your question.

The path you pass (e.g. `plans/my-feature.md`) is always a **file**. From it, multi-llm derives a sibling **output directory** in the same folder, named after the plan file without its extension - so `plans/my-feature.md` produces `plans/my-feature/`. All workflow state and generated artifacts live there:

| Path | Written by |
| --- | --- |
| `plans/my-feature/state.json` | every mode (tracks workflow progress) |
| `plans/my-feature/review-plan/` | `--review-plan`, `--apply-suggestions` |
| `plans/my-feature/tasks/` | `--generate-tasks`, `--review-tasks` |
| `plans/my-feature/implement/summary.md` | `--implement` |
| `plans/my-feature/code-review/report.md` | `--review-code`, `--apply-code-fixes` |
| `plans/my-feature/ask/<question-slug>/answers.md` | `--ask` |

Throughout this README and the skill docs, `{plan}/...` refers to that derived directory. Because everything is stored next to the plan, a workflow is fully portable - move or commit the plan folder and its state travels with it.

---

## Modes

Invoke with `/multi-llm:multi-llm [mode-flag] <plan_path> [options]`. With no mode flag, `--review-plan` is the default.

| Flag | What it does |
| --- | --- |
| `--review-plan` | Review an implementation plan with multiple LLMs (default) |
| `--apply-suggestions` | Apply validated review suggestions back into the plan |
| `--generate-tasks` | Generate detailed implementation tasks from a high-level plan |
| `--review-tasks` | Review generated tasks with multiple LLMs |
| `--apply-task-suggestions` | Apply validated task-review suggestions to `tasks.md` |
| `--implement` | Execute implementation tasks via subagent delegation |
| `--review-code` | Review code changes against the plan |
| `--apply-code-fixes` | Apply validated fixes from a code review |
| `--full` | Run the whole pipeline in sequence |
| `--status` | Show workflow state and the suggested next action |
| `--ask` | Ask each model a free-text question about a plan; aggregate the answers |

Two notes on the table above:

- **`--full`** chains the phases end to end: review-plan -> apply-suggestions -> generate-tasks -> review-tasks -> apply-task-suggestions -> implement -> review-code -> apply-code-fixes. It pauses at each apply step to let you approve `needs-human-decision` findings; pass `--yes` (alias `--non-interactive`) to run the whole pipeline fully unattended — zero prompts: non-interactive model selection, Claude decides every `needs-human-decision` item, and review-tasks runs automatically. For finer control, add just `--no-confirm` and/or `--claude-decide` (see [below](#letting-claude-decide-ambiguous-findings)) instead.
- **Applying code fixes** can happen two ways. Run `--apply-code-fixes` as a standalone pass to apply a previous review's fixes - this is the phase that handles `needs-human-decision` items (prompts, salvage, HTML badges). Or, to apply only the clearly-valid fixes inline during the review itself, add `--apply-fixes` to a `--review-code` run.

### Examples

```text
# Review a plan using the configured default LLMs
/multi-llm:multi-llm plans/my-feature.md

# Pick models explicitly, from several providers
/multi-llm:multi-llm --review-plan plans/my-feature.md --models codex:gpt-5.5-extra-high cursor-agent:composer-2.5

# Quick review (fewer, preselected subset of models)
/multi-llm:multi-llm --review-plan plans/my-feature.md --quick

# Review code changes against the plan
/multi-llm:multi-llm --review-code plans/my-feature.md

# Ask every model the same question (read-only Q&A)
/multi-llm:multi-llm --ask plans/my-feature.md "Is the rollback strategy sufficient?"
```

### Options

These flags combine with any mode and are **position-independent** - they may appear before or after the plan path (and, for `--ask`, around the question). Not every flag applies to every mode.

| Flag | Effect |
| --- | --- |
| `--models <provider:model> ...` | Use exactly these models (variadic), overriding YAML defaults. Each must exist in `providers.yaml`. |
| `--quick` | Use the smaller `quick_models` subset for a faster  run. |
| `--interactive` | Force two-step model selection (provider, then models), ignoring YAML defaults. |
| `--no-confirm` | Skip confirmation prompts - for unattended/silent runs. |
| `--dry-run` | Show what would happen (tasks/batches) without making changes. Applies to `--implement` and the apply modes. |
| `--claude-decide` / `--let-claude-decide` | In the apply modes, let a Claude subagent judge every `needs-human-decision` finding instead of prompting you (see [below](#letting-claude-decide-ambiguous-findings)). |
| `--force` | **Meaning depends on mode.** In the fan-out review/ask modes (`--review-plan`, `--review-tasks`, `--review-code`, `--ask`), resume an interrupted run: keep already-completed per-model results, run only the missing models, and bypass the completed-phase / partial-completion guards. In the apply modes, it's an alias for `--yes` (confirm bulk approval). |
| `--rerun-all` | In the fan-out review/ask modes, re-run every model from scratch, discarding any existing per-model results. Combine with `--force` (`--force --rerun-all`) for a fresh full re-run of an already-completed phase. |

The apply and implement modes accept additional approval flags (`--yes`, `--approve-all`, `--resume`, `--task`, and more); see [`SKILL.md`](skills/multi-llm/SKILL.md) and the per-mode files in `skills/multi-llm/instructions/`.

### Letting Claude decide ambiguous findings

When the models disagree or a suggestion is borderline, validation marks it `needs-human-decision`. In the apply modes (`--apply-suggestions`, `--apply-code-fixes`, `--apply-task-suggestions`) you're normally prompted via `AskUserQuestion` to approve or skip each one - and every prompt also offers a **Let Claude decide** option, per item or for a whole batch, that hands just that judgment to a Claude subagent instead of you.

To skip the prompts entirely and have Claude judge **every** `needs-human-decision` finding up front, pass `--claude-decide` (alias `--let-claude-decide`):

```text
/multi-llm:multi-llm --apply-code-fixes plans/my-feature.md --claude-decide --no-confirm
```

Unlike the blanket approve/skip flags, "Let Claude decide" is a **per-item** judgment that **salvages** partially-valid findings - it trims each to its worthwhile core and applies that, skipping only findings with nothing worth keeping. The generated HTML report badges every finding **Approved / Salvaged / Skipped**. Add `--no-confirm` (as above) for a fully unattended run.

---

## Providers

**Edit [`skills/multi-llm/providers.yaml`](skills/multi-llm/providers.yaml) before using the plugin** - it defines which code harnesses and models multi-llm calls by default. Each provider maps to a CLI binary that must be installed and authenticated separately:

| Provider key | CLI binary | Install / docs |
| --- | --- | --- |
| `claude-code` | `claude` | [Claude Code](https://code.claude.com/docs/en/) |
| `cursor-agent` | `cursor-agent` | [Cursor Agent CLI](https://cursor.com/docs/cli/overview) |
| `gemini` | `gemini` | [Gemini CLI](https://geminicli.com/docs/get-started/) |
| `codex` | `codex` | [Codex CLI](https://developers.openai.com/codex/cli) |
| `opencode` | `opencode` | [OpenCode CLI](https://opencode.ai/docs/cli/) |
| `kilocode` | `kilocode` | [Kilo Code CLI](https://kilo.ai/docs/cli) |

A model is referenced as `provider:model` (e.g. `codex:gpt-5.5`). A bare model name uses `default_provider` from `providers.yaml`.

> **Heads up - defaults:** the shipped `defaults.models` list references a specific set of models across `cursor-agent`, `kilocode`, and `claude-code`. If you don't have those CLIs installed and authenticated, either pass `--models` explicitly, run with `--interactive` to choose, or edit `defaults.models` in `providers.yaml` to match the providers you have. The safest zero-setup choice is `claude-code` models, which every Claude Code user already has.

Model selection priority (highest first): `--models`, then `--interactive`, then `--quick`, then `defaults.models` (from YAML), then interactive fallback.

---

## Per-project configuration

`providers.yaml` (above) is the **base** layer, shared by every repo. To give a
single repository its own **selection** defaults without editing the installed
plugin, add an override file:

```
<git-root>/.multi-llm/providers.yaml
```

It is **optional and auto-discovered**. When absent, behavior is identical to
today (base defaults). It overrides only the *selection* keys — `default_provider`,
`defaults.models`, `defaults.quick_models`, `defaults.modes` — and inherits
everything else from the base, so it can be tiny:

```yaml
# .multi-llm/providers.yaml — only what this repo changes
default_provider: claude-code
defaults:
  models:
    - claude-code:opus
    - cursor-agent:composer-2.5
  quick_models:
    - claude-code:opus
```

### Layering and precedence

Config is assembled lowest → highest, each layer **deep-merged** over the one below:

1. **Base** — the plugin's `providers.yaml` (always present).
2. **Project-local** — `<git-root>/.multi-llm/providers.yaml` (the feature).
3. **Env override** — `MULTI_LLM_PROVIDERS_CONFIG=/path.yaml` (escape hatch; see below).

**Lists replace, they do not append.** If you set `defaults.models`, you get
*exactly* that list — the base list is discarded, not extended. Only nested maps
deep-merge; any list or scalar value replaces wholesale.

### Blank vs. clear vs. omit

| You write | Result |
| --- | --- |
| Omit the key | Inherit the base value |
| `models:` (blank / null) | **Also inherit the base** — a blank value is *not* "clear" |
| `models: []` (empty list) | **Wipe out** the inherited list (use no models here) |

A blank/omitted key inherits the base; an **explicit empty list `[]` is the only
way to deliberately empty a list**. `[]` is a footgun, not a no-op: an empty
`defaults.models` falls back to interactive selection (and fails in unattended
runs like `--full`); an empty `quick_models` errors under `--quick`. You **cannot
unset a scalar back to "absent"** — you can only overwrite it. Likewise inherited
`providers.<name>` definitions and `defaults.modes.<mode>` entries can be changed
but **not removed** (only overwritten, e.g. `defaults.modes.code-review: []`).

### Two caveats worth knowing

- **A mode-specific base list still wins over a project-wide `defaults.models`.**
  `defaults.modes[<mode>]` is consulted *before* `defaults.models`. Because the
  project file deep-merges over the base, setting **only** `defaults.models` does
  **not** change modes that have a base `defaults.modes` entry (e.g. `--review-plan`
  / `--review-code`) — those keep their mode-specific list. To change a specific
  mode's models, override `defaults.modes.<mode>` (not just `defaults.models`).
- **Auto-discovery requires a git repo.** Discovery resolves the git root via
  `git rev-parse --show-toplevel`. A directory that is not a git repository has no
  auto-discovered layer at all — `.multi-llm/providers.yaml` is ignored even if it
  sits in the current directory. Non-git projects must use
  `MULTI_LLM_PROVIDERS_CONFIG` instead.

### Where discovery is anchored: plan path → CWD

The git root that discovery resolves is **anchored at the plan file's directory
when a plan path is supplied**, and falls back to the **current working directory**
otherwise. In a real run the orchestrator threads the plan-file path through, so
config discovery resolves from the *same* git root the orchestrator derives from
the plan (`get_project_root(plan_path)`) — not from CWD. CWD is only a fallback,
used when no plan path is available.

This matters when CWD is not the plan's git root — centrally-stored plans, an
absolute plan path from another repo, multi-repo workflows, or a CI job invoked
from a subdirectory. In those cases discovery follows the **plan's** repo, so the
loaded `.multi-llm/providers.yaml` matches the repo the orchestrator actually
operates on rather than whatever happens to be in CWD. (The env override
`MULTI_LLM_PROVIDERS_CONFIG`, by contrast, is always CWD-anchored for relative
paths — see below.)

### One override per repository

Although discovery is *anchored* at the plan path (above), it still resolves up to
the git **root**, so a repository has exactly **one** `.multi-llm/providers.yaml`,
shared repo-wide. The anchor selects *which repo's* config is loaded, not a finer
scope within it: two plans in the same repo always resolve to the same file. A
monorepo with multiple apps/packages **cannot** set per-package or per-subdirectory
defaults via auto-discovery — every subdirectory resolves to the same git root. For
genuinely per-plan or per-subdirectory overrides, use `MULTI_LLM_PROVIDERS_CONFIG`
per invocation.

### Scaffold it with `--init`

```text
/multi-llm:multi-llm --init                 # writes a commented stub at the git root
/multi-llm:multi-llm --init --gitignore     # also ignore it (developer-local)
/multi-llm:multi-llm --init --force         # overwrite an existing file
```

Under the hood this runs the standalone scaffolder (no orchestrator):

```bash
uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/init_config.py [--dir PATH] [--force] [--gitignore]
```

It refuses to clobber an existing file without `--force` and prints the path it wrote.

### Commit it or ignore it?

The override is **trackable by default** — that's the common case:

- **Commit** it for a **team-wide, repo-standard** model/provider selection that
  everyone shares.
- **Ignore** it for **personal, per-developer** preferences: run
  `--init --gitignore` (which appends `.multi-llm/` to `.gitignore`), or point
  `MULTI_LLM_PROVIDERS_CONFIG` at a fully out-of-tree path. This keeps individual
  model choices out of the shared repo.

Decide deliberately so teams neither accidentally commit individual model choices
nor accidentally ignore a config they meant to share.

### Env override: `MULTI_LLM_PROVIDERS_CONFIG`

Set `MULTI_LLM_PROVIDERS_CONFIG=/path/to.yaml` to layer a config on top of the
project-local file (highest precedence) — useful for CI or one-offs, and the only
override mechanism for non-git directories. **Relative paths are resolved against
the current working directory** (which is the repo root in real runs), not the git
root. A value that is set but points at a missing file warns and is skipped.

### Trust model & safety

Running multi-llm inside a repo **auto-loads that repo's
`.multi-llm/providers.yaml`** — so a freshly cloned, untrusted repo can ship one
that changes which providers/models a run selects. The blast radius is limited to
**selection**, not code execution: provider binaries are hardcoded in the plugin,
and the config `command` field is documentation-only and is **never executed from
config**, in any layer. For that reason the auto-discovered layer is restricted to
selection keys — a `providers:` block there is **ignored** (dropped with a warning)
so cloned content cannot redefine provider capabilities.

A **present-but-malformed** explicit override (bad YAML, a non-mapping root, an
unreadable file) **fails fast** with a clear error naming the file, rather than
silently degrading to the base (which could run the wrong, more expensive models).
An **absent** override falls through silently to the base. Set
`MULTI_LLM_PROVIDERS_CONFIG_PERMISSIVE=1` to opt back into best-effort
warn-and-skip loading.

---

## How it works

- **Orchestrators emit instructions, not edits.** Each mode runs a Python orchestrator that queries the selected models in parallel, validates and consolidates their output, and prints structured instructions (and JSON) for Claude Code to act on.
- **State is plan-local.** Workflow state lives in `{plan}/state.json` and all outputs are written next to the plan, so everything is portable and scoped to the plan it belongs to.
- **Relocatable paths.** The skill references its own bundled files through `${CLAUDE_SKILL_DIR}`, so it works wherever Claude Code installs the plugin.

See [`skills/multi-llm/SKILL.md`](skills/multi-llm/SKILL.md) for the full operating contract and [`skills/multi-llm/AGENTS.md`](skills/multi-llm/AGENTS.md) for architecture notes.

### Your project repository

The repo where you run multi-llm workflows should include one or more `AGENTS.md` files - at the repo root and/or in subdirectories - with project-specific guidance for AI agents (conventions, architecture, pitfalls, etc.). `AGENTS.md` is supported by most code harnesses (Cursor, Codex, OpenCode, Gemini CLI, and others), so the external models multi-llm invokes pick up the same context.

To keep a single source of truth while still working with Claude Code, add a sibling `CLAUDE.md` that contains only:

```text
@AGENTS.md
```

Claude Code includes the referenced file automatically. This plugin follows that pattern in [`skills/multi-llm/AGENTS.md`](skills/multi-llm/AGENTS.md) and [`skills/multi-llm/CLAUDE.md`](skills/multi-llm/CLAUDE.md).

---

## Development

The skill's source and test suite live under `skills/multi-llm/`.

```bash
# Run the test suite (from the repo root)
uv run --project skills/multi-llm -- pytest

# Run a single test file
uv run --project skills/multi-llm -- pytest skills/multi-llm/tests/test_filtering.py -v

# Robustness (error-handling) tests
uv run --project skills/multi-llm -- pytest -m robustness
```

Tests marked `live` require provider CLIs and real API access and are skipped in CI.

### Validating the plugin

```bash
claude plugin validate .   # validates marketplace.json and the referenced plugin
```

---

## Roadmap

Planned improvements and open ideas are tracked in [`TODO.md`](TODO.md).

Contributions are welcome - pull requests, bug reports, feature suggestions, and documentation improvements. Open an issue to discuss a larger change before starting work, or send a PR directly for smaller fixes. See the [Development](#development) section for running tests locally.

---

## About BeastLab.ai

This plugin is maintained by [BeastLab.ai](https://beastlab.ai) - a frontier lab building multi-agent reasoning models. Beast models run multiple internal agents that deliberate at inference time for deeper reasoning on complex coding, agentic workflows, and research tasks. If you want to try one of the most capable LLM offerings available today, visit [beastlab.ai](https://beastlab.ai) to explore the lineup and integration guide.

**Disclaimer:** BeastLab.ai provides this plugin as-is, with no warranties or liabilities of any kind. BeastLab.ai is not affiliated with, endorsed by, or sponsored by any of the code harnesses (Cursor, Claude Code, Codex, OpenCode, Gemini CLI, Kilo Code, etc.) or third-party LLM providers referenced in this project. Trademarks and product names belong to their respective owners.

---

## License

[MIT](LICENSE) © beastlabai
