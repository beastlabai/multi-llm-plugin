# multi-llm - Harness the harnesses

![multi-llm plugin for Claude Code](https://beastlab.ai/images/multi-llm-plugin-for-claude-code-monster.jpg)

**Wisdom of crowds for your codebase.** Use this plugin to improve planning, implementation, and code review by running the same work through multiple AI coding tools and LLMs in parallel - then consolidating their feedback so you catch bugs, blind spots, and improvements a single model might miss.

The diversity works on two axes: not only do you get different **LLM models** (Opus, GPT, Gemini, Grok, Composer, ...), you also get different **code harnesses** (19 supported CLIs: [Cursor Agent](https://cursor.com/docs/cli/overview), [Gemini CLI](https://geminicli.com/docs/get-started/), [Grok Build](https://docs.x.ai/build), [Codex](https://developers.openai.com/codex/cli), [OpenCode](https://opencode.ai/docs/cli/), [Kilocode](https://kilo.ai/docs/cli), [Cline](https://docs.cline.bot/), [goose](https://block.github.io/goose/), [Aider](https://aider.chat/), [Antigravity CLI](https://antigravity.google/docs/cli/using), [Pi](https://pi.dev/docs/latest/usage), [Muse Code](https://dev.meta.ai/docs/muse-code/extending), [Kimi Code](https://code.kimi.com/docs), [Qwen Code](https://qwenlm.github.io/qwen-code-docs/), [GitHub Copilot CLI](https://docs.github.com/en/copilot/concepts/agents/about-copilot-cli), [Mistral Vibe](https://docs.mistral.ai/vibe/), [oh-my-pi](https://github.com/can1357/oh-my-pi), [Vercel fx](https://github.com/vercel-labs/fx), and [Claude Code](https://code.claude.com/docs/en/) itself) - each with its own prompting, tooling, and context-gathering behavior. Two harnesses running the *same* model can still surface different findings, so combining both axes widens the crowd further.

📺 **[Short video intro](https://www.youtube.com/watch?v=91lpnxAsyqs)** or **[Full video walkthrough](https://www.youtube.com/watch?v=_ySRt7rr0r8)** of the plugin in action.

`multi-llm` fans **plan reviews, task generation, implementation, code reviews and commit reviews** out to several CLI-based LLMs in parallel - or asks every model the same free-text question about a plan - then validates and consolidates their suggestions and hands the host structured instructions to apply the results. The orchestrators never modify your code directly - they produce JSON that the host executes through its own tools, so every change stays reviewable.

It installs as a **Claude Code plugin**, and the same workflow runs inside **Codex** and **OpenCode** too: on Codex as a Codex plugin, on OpenCode as the [`opencode-multi-llm`](https://www.npmjs.com/package/opencode-multi-llm) npm package. For either host, `install-host` is an alternative that writes a native entry point (a small command file) instead. See [Installation](#1-installation) and [docs/hosts.md](docs/hosts.md).

---

## What it looks like

`plans/my-feature.md` stands for any Markdown plan; the commit modes need none. These are Claude Code commands: on OpenCode type `/multi-llm` instead, and on Codex `$multi-llm:multi-llm`, followed by the same arguments.

```text
# Review a plan using the configured default LLMs
/multi-llm:multi-llm plans/my-feature.md

# Pick models explicitly, from several providers
/multi-llm:multi-llm --review-plan plans/my-feature.md --models codex:gpt-5.5-extra-high cursor-agent:composer-2.5

# Quick review (fewer, preselected subset of models)
/multi-llm:multi-llm --review-plan plans/my-feature.md --quick

# Review code changes against the plan
/multi-llm:multi-llm --review-code plans/my-feature.md

# Review the plan, then apply the validated suggestions to it, unattended
/multi-llm:multi-llm --improve-plan plans/my-feature.md --quick

# Review your last commit (no plan needed)
/multi-llm:multi-llm --review-commits

# Review the commits after main, with error handling as the priority
/multi-llm:multi-llm --review-commits main..HEAD --focus "error handling"

# Review this branch's commits, then apply the validated fixes, unattended
/multi-llm:multi-llm --improve-commits this branch

# Review/implement/code-review a plan using the configured default LLMs completely hands-off
/multi-llm:multi-llm --full plans/my-feature.md --agent-decide --non-interactive

# Ask every model the same question (read-only Q&A)
/multi-llm:multi-llm --ask plans/my-feature.md "Is the rollback strategy sufficient?"
```

Every mode and option is in [3. Modes](#3-modes).

---

## 1. Installation

Pick your host (1.1 to 1.3), then configure providers (1.4). The prerequisites in 1.5 apply to every host.

### 1.1 Claude Code

Installs as a Claude Code plugin. In Claude Code:

```text
/plugin marketplace add beastlabai/multi-llm-plugin
/plugin install multi-llm@beastlabai
/reload-plugins
/reload-skills
/multi-llm:multi-llm --init
```

Then invoke the skill (plugin skills are namespaced `plugin:skill`), or just ask Claude to "review my plan with multiple models":

```text
/multi-llm:multi-llm plans/my-feature.md
```

> **Added the marketplace before, in another project?** `/plugin marketplace add` reuses the cached copy and installs a stale version. Run the [update steps](docs/installation.md#updating-the-claude-code-plugin) instead.

### 1.2 Codex

Installs as a Codex plugin: a `multi_llm` tool plus a `multi-llm:multi-llm` skill. Needs codex >= 0.159.2, python3 >= 3.9 and `uv` on `PATH`. Linux is verified; on Windows use the file route.

```bash
codex plugin marketplace add beastlabai/multi-llm-plugin --ref release \
  --sparse .agents --sparse packaging/codex
codex plugin add multi-llm@beastlabai
# then start a NEW Codex session
```

Codex plugins cannot add slash commands, so `/multi-llm` is not available. Mention the skill with `$` instead, followed by the same arguments the Claude Code skill takes. Once per repository, run `--init` first: it writes `<git-root>/.multi-llm/providers.yaml`, which no upgrade touches.

```text
$multi-llm:multi-llm --init
$multi-llm:multi-llm plans/my-feature.md
```

If the model ever reworks your arguments, spell the request out: `Use the multi-llm:multi-llm skill with exactly these arguments: plans/my-feature.md`.

> **Before you install:** the plugin's tool runs outside Codex's exec sandbox, so in a read-only session it refuses the modes that edit your working tree unless `MULTI_LLM_CODEX_UNSANDBOXED_WRITES=1` is set. Despite its name, `MULTI_LLM_CODEX_UNSANDBOXED_WRITES=1` lifts only this tool's own refusal of write-capable modes under a read-only or unreadable session posture. It never lets the Codex processes the orchestrator starts write without a sandbox, and it does not change which sandbox they run under: a Codex editing job refused because Codex's own sandbox cannot start on this host stays refused with it set. Provider API keys must be exported from `~/.profile` or `~/.bash_profile`, because Codex starts the tool with a stripped environment.

Updating, uninstalling, platform support and troubleshooting are in [docs/hosts.md](docs/hosts.md#codex-plugin-route).

### 1.3 OpenCode

Installs from npm as [`opencode-multi-llm`](https://www.npmjs.com/package/opencode-multi-llm): a `/multi-llm` command plus a `multi_llm` tool. Needs opencode 1.19.5 or newer and `uv` on `PATH`. Linux is verified; macOS and Windows are untested on this route.

1. **Add the package** to the `plugin` list in `~/.config/opencode/opencode.json` (create the file if you have none), keeping the exact version (OpenCode never updates an unpinned entry):

   ```json
   { "plugin": ["opencode-multi-llm@0.1.1"] }
   ```

2. **Restart OpenCode** while online, then check that it loaded - on this route a failure is silent:

   ```bash
   opencode debug config | jq '.command["multi-llm"] != null'   # true = loaded
   ```

3. **Run `--init` once per repository,** then use it:

   ```text
   /multi-llm --init
   /multi-llm plans/my-feature.md
   ```

To update, change the pin and restart. Moving from the `install-host opencode` file route, uninstalling and troubleshooting are in [docs/hosts.md](docs/hosts.md#opencode-npm-route).

### 1.4 Configure providers

`--init` (the last install step on every host) writes `<git-root>/.multi-llm/providers.yaml`, switching on the provider CLIs it finds on your `PATH` and a model panel for them. Open it before your first run and trim `defaults.models` (and `quick_models`, used by `--quick`) to the `provider:model` pairs you actually want: a default run calls every model listed there (a mode with its own list under `defaults.modes` uses that instead). Run `--init` in each repository you use multi-llm in; commit the file to share the selection with your team.

- [docs/configuration.md](docs/configuration.md) - what `--init` writes, how the file layers over the plugin's own defaults, and why it can never change *what executes*.
- [docs/providers.md](docs/providers.md) - every supported CLI, its quirks, and pointing one at a different endpoint.

### 1.5 Prerequisites

- **[`uv`](https://docs.astral.sh/uv/) on your `PATH`.** It supplies Python 3.10+ and the one dependency (`pyyaml`) on first use. If it is missing, the skill prints install instructions and stops; it never installs uv itself.
- **[Git](https://git-scm.com/), and a Git repository to run in.**
- **On Windows, [Git for Windows](https://git-scm.com/download/win) (Git Bash)** - a hard prerequisite; PowerShell is not supported.
- **One or more provider CLIs,** installed and signed in - only the ones whose models you run.

Install paths per OS, and why each one is needed: [docs/installation.md](docs/installation.md#prerequisites-in-full).

---

## 2. Plans and outputs

A **plan** is just a markdown file describing the change you want to make - a feature spec, refactor outline, or design doc. You write it (or have Claude draft it), then point multi-llm at it. There's no required template; the more concrete the plan, the more useful the reviews. The commit modes need no plan, and `--ask` uses one only as context for your question.

From `plans/my-feature.md`, multi-llm derives a sibling folder, `plans/my-feature/`, that holds the workflow state (`state.json`) and every report. Because everything is stored next to the plan, a workflow is portable and resumable. Where each mode writes: [docs/workflows.md](docs/workflows.md#plans-and-outputs).

---

## 3. Modes

Invoke with `/multi-llm:multi-llm [mode-flag] <plan_path> [options]` - on OpenCode the same command is `/multi-llm`, without the plugin namespace; on Codex it is a skill (model-invoked, not a slash command), so you ask for `multi-llm:multi-llm` by name instead (see [docs/hosts.md](docs/hosts.md)). With no mode flag, `--review-plan` is the default. The three commit modes take an optional `[<range>]` in place of `<plan_path>`, and `--init` takes neither.

| Flag | What it does |
| --- | --- |
| `--review-plan` | Review an implementation plan with multiple LLMs (default) |
| `--apply-suggestions` | Apply validated review suggestions back into the plan |
| `--improve-plan` | `--review-plan`, then `--apply-suggestions`, in one unattended command ([below](#32-review-and-apply-in-one-command)) |
| `--generate-tasks` | Generate detailed implementation tasks from a high-level plan |
| `--review-tasks` | Review generated tasks with multiple LLMs |
| `--apply-task-suggestions` | Apply validated task-review suggestions to `tasks.md` |
| `--implement` | Execute implementation tasks via subagent delegation |
| `--review-code` | Review code changes against the plan |
| `--apply-code-fixes` | Apply validated fixes from a code review |
| `--improve-code` | `--review-code`, then `--apply-code-fixes`, in one unattended command that edits source files ([below](#32-review-and-apply-in-one-command)) |
| `--review-commits` | Review a commit or a range of commits (default: the last commit) ([below](#33-review-commits)) |
| `--apply-commit-fixes` | Apply validated fixes from a commit review ([below](#33-review-commits)) |
| `--improve-commits` | `--review-commits`, then `--apply-commit-fixes`, in one unattended command that edits source files ([below](#33-review-commits)) |
| `--full` | Run the whole pipeline in sequence |
| `--status` | Show workflow state and the suggested next action |
| `--ask` | Ask each model a free-text question about a plan; aggregate the answers |
| `--init` | Write a per-project `.multi-llm/providers.yaml` ([Configure providers](#14-configure-providers)) |

### 3.1 Pipeline flow

The mode flags are designed to run as an ordered pipeline (exactly what `--full` chains for you). Each **review** step fans the work out to multiple LLMs in parallel; each **apply** step is where the host writes the changes back and pauses for any `needs-human-decision` findings.

```mermaid
flowchart TD
    START(["plan.md"]) -.-> FULL["--full"]:::full
    FULL -. "runs every step in order" .-> DONE(["done"])

    subgraph PLAN ["Implementation Plan"]
        RP["--review-plan"]:::review --> AS["--apply-suggestions"]:::apply
    end

    subgraph TASKS ["Implementation Tasks"]
        GT["--generate-tasks"]:::gen --> RT["--review-tasks"]:::review --> ATS["--apply-task-suggestions"]:::apply
    end

    subgraph IMPLEMENT ["Implement"]
        IMP["--implement"]:::gen
    end

    subgraph CODE ["Code Review"]
        RC["--review-code"]:::review --> ACF["--apply-code-fixes"]:::apply
    end

    START --> RP
    AS --> GT
    ATS --> IMP
    IMP --> RC
    ACF --> DONE

    classDef review fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a;
    classDef gen fill:#ede9fe,stroke:#8b5cf6,color:#4c1d95;
    classDef apply fill:#ffedd5,stroke:#f97316,color:#7c2d12;
    classDef full fill:#dcfce7,stroke:#22c55e,color:#14532d;
```

- 🔵 **Review** steps query multiple models in parallel
- 🟠 **Apply** steps let the host write changes and pause for `needs-human-decision` items
- 🟣 **Generate & implement** steps produce tasks and write code
- 🟢 **`--full`** runs the entire flow with one command; add `--yes` to run it unattended ([details](docs/workflows.md#running-the-pipeline))

You don't have to run the whole chain: **every step is also a standalone entry point** that reads/writes the shared `{plan}/state.json`, so you can start, stop, and resume at any phase.

### 3.2 Review and apply in one command

`--improve-plan` runs the *Implementation Plan* pair above and `--improve-code` the *Code Review* pair, each as one unattended command: a judge agent decides every `needs-human-decision` item and nobody is asked. **They edit files without asking** - `--improve-plan` the plan in place, `--improve-code` source files anywhere in the repository - so commit or stash first; git is the way back. Re-runs, `--force`, undo and cost: [docs/workflows.md](docs/workflows.md#review-and-apply-in-one-command).

### 3.3 Review commits

`--review-commits` reviews what was committed, not what is on disk, and needs no plan: the commit messages are the stated intent, and every model reviews the range's net change. Leave the range out to review the last commit, or type one right after the mode word, before any flag - `HEAD~2`, `main..HEAD`, `last 3 commits`, `yesterday`, `this branch`. `--apply-commit-fixes` then applies the validated fixes as uncommitted edits, and `--improve-commits` does both, unattended. Nothing commits, amends or rebases. The range forms, the report, `--focus` and `--context-plan`: [docs/commit-reviews.md](docs/commit-reviews.md).

### 3.4 Common options

Options may appear before or after the plan path. Not every flag applies to every mode.

| Flag | Effect |
| --- | --- |
| `--models <provider:model> ...` | Use exactly these models, overriding the YAML defaults. |
| `--quick` | Use the smaller `quick_models` subset for a faster run. |
| `--interactive` | Pick the models yourself (provider, then models). |
| `--dry-run` | Show what `--implement` or an apply mode would do, without changing anything. |
| `--agent-decide` | In the apply modes, let a judge agent decide every `needs-human-decision` finding instead of asking you. |
| `--no-confirm` | Skip confirmation prompts, for unattended runs. |

When the models disagree or a suggestion is borderline, validation marks it `needs-human-decision`. On Claude Code the apply modes ask you about each one, each with a **Let the agent decide** option, per item or for a whole batch; on Codex and OpenCode, and with `--agent-decide`, a judge agent decides them all. The judge salvages partially-valid findings rather than dropping them, and the HTML report badges every finding **Approved / Salvaged / Skipped**.

Every flag, including `--force`, `--rerun-all`, `--focus` and `--context-plan`, and how the judge is chosen: [docs/workflows.md](docs/workflows.md#options).

---

## 4. How it works

- **Orchestrators emit instructions, not edits.** Each mode runs a Python orchestrator that queries the selected models in parallel, validates and consolidates their output, and prints structured instructions (and JSON) for the host to act on.
- **State is plan-local.** Workflow state lives in `{plan}/state.json` and all outputs are written next to the plan, so everything is portable and scoped to the plan it belongs to.

Where the instruction core, the host shims and the architecture notes live: [docs/development.md](docs/development.md#where-things-live).

---

## 5. Your project repository

The repo where you run multi-llm workflows should include one or more `AGENTS.md` files - at the repo root and/or in subdirectories - with project-specific guidance for AI agents (conventions, architecture, pitfalls, etc.). `AGENTS.md` is supported by most code harnesses (Cursor, Codex, OpenCode, Gemini CLI, and others), so the external models multi-llm invokes pick up the same context.

To keep a single source of truth while still working with Claude Code, add a sibling `CLAUDE.md` that contains only `@AGENTS.md`. Claude Code includes the referenced file automatically. This plugin follows that pattern in [`skills/multi-llm/AGENTS.md`](skills/multi-llm/AGENTS.md) and [`skills/multi-llm/CLAUDE.md`](skills/multi-llm/CLAUDE.md).

---

## 6. Documentation

| Document | What's in it |
| --- | --- |
| [docs/installation.md](docs/installation.md) | Updating the Claude Code plugin, the prerequisites in full, and the plugin's own `providers.yaml` |
| [docs/hosts.md](docs/hosts.md) | Codex and OpenCode in full: both plugin routes, the `install-host` file route, capability matrix, verification, uninstall |
| [docs/providers.md](docs/providers.md) | Every supported harness CLI, `provider:model` naming, and endpoint recipes (GLM / Z.ai) |
| [docs/configuration.md](docs/configuration.md) | The per-project `.multi-llm/providers.yaml` override, layer precedence, and its safety model |
| [docs/workflows.md](docs/workflows.md) | Where each mode writes, `--full`, the review-and-apply modes, every option, and how ambiguous findings get decided |
| [docs/commit-reviews.md](docs/commit-reviews.md) | `--review-commits` and its apply modes: range forms, output, reports, `--focus` / `--context-plan`, safety |
| [docs/development.md](docs/development.md) | Where things live, the test suite, the containerized E2E harness (Docker), and plugin validation |
| [docs/attribution.md](docs/attribution.md) | Trademark disclaimer, the one proprietary harness, and HTML-report brand marks |
| [`skills/multi-llm/AGENTS.md`](skills/multi-llm/AGENTS.md) | Architecture notes for anyone changing the skill itself |

---

## 7. Roadmap and contributing

Planned improvements and open ideas are tracked in [`TODO.md`](TODO.md).

Contributions are welcome - pull requests, bug reports, feature suggestions, and documentation improvements. Open an issue to discuss a larger change before starting work, or send a PR directly for smaller fixes. See [docs/development.md](docs/development.md) for running tests locally.

---

## 8. About BeastLab.ai

This plugin is maintained by [BeastLab.ai](https://beastlab.ai) - a frontier lab building multi-agent reasoning models.

**Disclaimer:** this plugin is provided as-is, with no warranties or liabilities of any kind, and BeastLab.ai is not affiliated with, endorsed by, or sponsored by any of the code harnesses or LLM providers it talks to. The full notice is in [docs/attribution.md](docs/attribution.md).

---

## 9. License

[MIT](LICENSE) © beastlabai
