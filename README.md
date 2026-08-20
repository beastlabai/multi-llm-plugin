# multi-llm - Harness the harnesses

![multi-llm plugin for Claude Code](https://beastlab.ai/images/multi-llm-plugin-for-claude-code-monster.jpg)

**Wisdom of crowds for your codebase.** Use this plugin to improve planning, implementation, and code review by running the same work through multiple AI coding tools and LLMs in parallel - then consolidating their feedback so you catch bugs, blind spots, and improvements a single model might miss.

The diversity works on two axes: not only do you get different **LLM models** (Opus, GPT, Gemini, Grok, Composer, ...), you also get different **code harnesses** (Codex, OpenCode, Cursor Agent, Gemini CLI, Grok Build, Cline, goose, Aider, Antigravity CLI, Pi, Muse Code, Kimi Code, Qwen Code, Copilot CLI, Mistral Vibe, ...) - each with its own prompting, tooling, and context-gathering behavior. Two harnesses running the *same* model can still surface different findings, so combining both axes widens the crowd further.

📺 **[Short video intro](https://www.youtube.com/watch?v=91lpnxAsyqs)** or **[Full video walkthrough](https://www.youtube.com/watch?v=_ySRt7rr0r8)** of the plugin in action.

Orchestrate **plan reviews, task generation, implementation, and code reviews across multiple LLM providers in parallel** - or ask every model the same free-text question about a plan and get one consolidated answer.

`multi-llm` fans a single workflow out to several CLI-based LLMs ([Cursor Agent](https://cursor.com/docs/cli/overview), [Gemini CLI](https://geminicli.com/docs/get-started/), [Grok Build](https://docs.x.ai/build), [Codex](https://developers.openai.com/codex/cli), [OpenCode](https://opencode.ai/docs/cli/), [Kilocode](https://kilo.ai/docs/cli), [Cline](https://docs.cline.bot/), [goose](https://block.github.io/goose/), [Aider](https://aider.chat/), [Antigravity CLI](https://antigravity.google/docs/cli/using), [Pi](https://pi.dev/docs/latest/usage), [Muse Code](https://dev.meta.ai/docs/muse-code/extending), [Kimi Code](https://code.kimi.com/docs), [Qwen Code](https://qwenlm.github.io/qwen-code-docs/), [GitHub Copilot CLI](https://docs.github.com/en/copilot/concepts/agents/about-copilot-cli), [Mistral Vibe](https://docs.mistral.ai/vibe/), and [Claude Code](https://code.claude.com/docs/en/) itself), validates and consolidates their suggestions, and hands Claude Code structured instructions to apply the results. The orchestrators never modify your code directly - they produce JSON that Claude Code executes through its own tools, so every change stays reviewable.

---

## Installation

In Claude Code:

```text
/plugin marketplace add beastlabai/multi-llm-plugin
/plugin install multi-llm@beastlabai
/reload-plugins
/reload-skills
/multi-llm:multi-llm --init
```

> **Already added the marketplace before (e.g. in another project)?** `/plugin marketplace add` does **not** refetch an existing marketplace - it silently reuses the locally cached copy, so `/plugin install` will give you a stale version. Follow [Updating](#updating) instead.

Then invoke the skill (plugin skills are namespaced `plugin:skill`):

```text
/multi-llm:multi-llm plans/my-feature.md
```

> Claude can also load the skill automatically when you ask it to "review my plan with multiple models", "run a multi-LLM code review", etc.

### Updating

The marketplace cache only refreshes when you explicitly ask it to - `/plugin marketplace update` is the step that actually pulls the latest commits from GitHub. Then update (or install) the plugin and reload:

```text
/plugin marketplace update beastlabai
# remove "--scope local" if you installed it as user scope
! claude plugin update multi-llm@beastlabai --scope local
/reload-plugins
/reload-skills
/multi-llm:multi-llm --init --force
```


### Configure providers (required before first use)

Edit [`skills/multi-llm/providers.yaml`](skills/multi-llm/providers.yaml) manually to match the code harnesses and models you want to use. The shipped file is a starting point only - keep the providers whose CLIs you have installed, remove the rest, and set `defaults.models` (and optionally `quick_models`) to the `provider:model` pairs you actually run. Without this step, default invocations will call harnesses or models you may not have configured. See [Providers](#providers) for the format and available keys.

> **Where is the file after installing via `/plugin`?** If you installed the plugin (rather than cloning this repo), `providers.yaml` lives inside the installed plugin, under your Claude Code plugins directory - typically `~/.claude/plugins/<marketplace>/multi-llm/skills/multi-llm/providers.yaml`. The quickest way to open it is to ask Claude Code to "open the multi-llm `providers.yaml`": the skill resolves its own install location (via `${CLAUDE_SKILL_DIR}`), so Claude can read and edit the file in place.

### Prerequisites

| Requirement | Why |
| --- | --- |
| [`uv`](https://docs.astral.sh/uv/) on your `PATH` | The orchestrators run as `uv run` Python scripts. uv installs their dependencies (just `pyyaml`) on first use. If `uv` is missing when a mode starts (`--init`, `--review-plan`, `--review-code`, ...), the skill detects that and offers to install it before running anything. Install location and installer differ per OS: on **Linux/macOS** the official installer (`curl -LsSf https://astral.sh/uv/install.sh \| sh`) puts `uv` in `~/.local/bin/uv`; on **macOS** Homebrew installs to `/opt/homebrew/bin/uv` (Apple Silicon) or `/usr/local/bin/uv` (Intel), and MacPorts to `/opt/local/bin/uv`; on **Windows** the official installer is PowerShell (`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"`) and creates `%USERPROFILE%\.local\bin\uv.exe` — which Git Bash sees as `~/.local/bin/uv.exe` (note the `.exe`); `winget install --id=astral-sh.uv -e` and `pipx install uv` also work. |
| **Windows only:** [Git for Windows](https://git-scm.com/download/win) (Git Bash) | **Hard prerequisite on Windows.** The skill's entire instruction set and its permission (`allowed-tools`) rules are Bash-only; Claude Code's PowerShell-tool fallback (used when Git for Windows is absent) is explicitly **unsupported** — the skill fails fast with an install pointer rather than attempting PowerShell equivalents. Git for Windows also supplies the bundled coreutils the command blocks depend on (`ls`, `grep`, `diff`, `tail`, `dirname`, ...), which are absent from PowerShell/cmd. Note that `realpath` is *not* one of them: plan paths are passed to the orchestrators as-is (never wrapped in `$(realpath ...)`), and the orchestrators resolve them to OS-native absolute paths themselves. |
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

### Pipeline flow

The mode flags are designed to run as an ordered pipeline (exactly what `--full` chains for you). Each **review** step fans the work out to multiple LLMs in parallel; each **apply** step is where Claude Code writes the changes back and pauses for any `needs-human-decision` findings.

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
- 🟠 **Apply** steps let Claude Code write changes and pause for `needs-human-decision` items
- 🟣 **Generate & implement** steps produce tasks and write code
- 🟢 **`--full`** runs the entire flow with one command


You don't have to run the whole chain: **every step is also a standalone entry point** that reads/writes the shared `{plan}/state.json`, so you can start, stop, and resume at any phase. A few modes sit outside the pipeline - **`--ask`** (read-only Q&A about a plan), **`--status`** (show progress and the suggested next step), and **`--init`** (write a per-project config) - while **`--full`** simply runs the entire flow above end to end.

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

# Review/implement/code-review a plan using the configured default LLMs completely hands-off
/multi-llm:multi-llm --full plans/my-feature.md --claude-decide --non-interactive

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

The apply and implement modes accept additional approval flags (`--yes`, `--approve-all`, `--resume`, `--task`, and more); run `multi_llm.py <mode> --help` for the authoritative list, or read the per-mode files in `skills/multi-llm/instructions/`.

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
| `grok` | `grok` | [Grok Build](https://docs.x.ai/build) |
| `codex` | `codex` | [Codex CLI](https://developers.openai.com/codex/cli) |
| `opencode` | `opencode` | [OpenCode CLI](https://opencode.ai/docs/cli/) |
| `kilocode` | `kilocode` | [Kilo Code CLI](https://kilo.ai/docs/cli) |
| `cline` | `cline` | [Cline](https://docs.cline.bot/) |
| `goose` | `goose` | [goose](https://block.github.io/goose/) |
| `aider` | `aider` | [Aider](https://aider.chat/) |
| `agy` | `agy` | [Antigravity CLI](https://antigravity.google/docs/cli/using) |
| `pi` | `pi` | [Pi](https://pi.dev/docs/latest/usage) |
| `muse` | `muse` | [Muse Code](https://dev.meta.ai/docs/muse-code/extending) |
| `kimi` | `kimi` | [Kimi Code](https://code.kimi.com/docs) - argv-only: a large prompt exceeds the Windows command-line cap and is refused before spawn, so this provider serves every fan-out mode on POSIX only. Also note: at 0.36.1 the CLI contacts `telemetry-logs.kimi.com` on every run even though multi-llm sets `KIMI_DISABLE_TELEMETRY=1` (measured; the opt-out does not gate the connection attempt) |
| `qwen-code` | `qwen` | [Qwen Code](https://qwenlm.github.io/qwen-code-docs/) - the provider key names the harness, the binary is `qwen`. Supports a stdin prompt transport (so a multi-KB prompt is never measured against the Windows command-line cap) and `--json-schema` structured output. Note: at 0.21.13 the CLI **exits 0 on every API error** (401/404/400/500 alike) with the diagnosis on stdout as `[API Error: ...]` and stderr empty - multi-llm detects that marker, so a failed run is reported as a failure rather than as the model's answer. It also runs with `--max-tool-calls 0`, so a tool call is refused before it executes rather than touching your checkout - measured at ~29% of live runs, which the retry ladder normally absorbs at the cost of a second call |
| `copilot` | `copilot` | [GitHub Copilot CLI](https://docs.github.com/en/copilot/concepts/agents/about-copilot-cli) - driven **BYOK**, so it needs no GitHub account, no PAT and no Copilot seat: pointing it at an OpenRouter key is the whole setup, and multi-llm strips any `GH_TOKEN`/`GITHUB_TOKEN` you already export so a stale one cannot re-enable the GitHub side underneath. Supports a stdin prompt transport (so a multi-KB prompt is never measured against the Windows command-line cap) and the `:effort` suffix. Two measured notes: the shipped model list is **non-reasoning slugs only**, because a stream carrying `reasoning_content` breaks the CLI's own completions parser; and an empty provider 200 makes the CLI **exit 0 with an empty answer and no error record**, which multi-llm detects structurally rather than by a marker |
| `vibe` | `vibe` | [Mistral Vibe](https://docs.mistral.ai/vibe/) - installed from PyPI as `mistral-vibe`, and it needs Python 3.12+. It has **no `--model` flag**: the entries above are aliases defined in `$VIBE_HOME/config.toml`, and multi-llm selects one through `VIBE_ACTIVE_MODEL`, so pointing vibe at a different model means editing that file as well as this list. Supports a stdin prompt transport (so a multi-KB prompt is never measured against the Windows command-line cap) and the `:effort` suffix, which rides the config's `thinking` key rather than a flag. One measured note worth knowing: vibe's own default agent is `accept-edits`, which auto-approves file writes, so multi-llm always passes `--agent plan` (read-only) and pins `VIBE_DEFAULT_AGENT` to match |

A model is referenced as `provider:model` (e.g. `codex:gpt-5.5`). A bare model name uses `default_provider` from `providers.yaml`.

> **Heads up - defaults:** the shipped `defaults.models` list references a specific set of models across `cursor-agent`, `kilocode`, and `claude-code`. If you don't have those CLIs installed and authenticated, either pass `--models` explicitly, run with `--interactive` to choose, or edit `defaults.models` in `providers.yaml` to match the providers you have. The safest zero-setup choice is `claude-code` models, which every Claude Code user already has.

Model selection priority (highest first): `--models`, then `--interactive`, then `--quick`, then `defaults.models` (from YAML), then interactive fallback.

### Pointing a provider CLI at a different endpoint (GLM / Z.ai, and the general case)

**No config layer sets environment for a provider** - `providers.yaml`, the per-project override and `MULTI_LLM_PROVIDERS_CONFIG` carry `default_timeout`, `max_concurrent` and `models` and nothing else, and a key the base does not declare merges and is then ignored, so there is no `env:` block to write. Endpoint and credential wiring is **shell configuration this plugin neither owns nor validates**: export the variables in your shell, your `.envrc`, or per invocation (`env VAR=... multi-llm ...`). They reach the CLI because each spawn hands the child the inherited environment plus the adapter's own delta, and no adapter touches the `ANTHROPIC_*` namespace below.

GLM (Z.ai) is a worked example, because GLM models are already first-class here - `z-ai/glm-5.2` is the default model for five providers. There are three recipes, at **two different evidence levels**, and the difference is stated rather than left to be assumed.

**R1 - GLM over the OpenRouter lane (recommended, live-verified).** Runs `claude-code` against a GLM slug on an OpenRouter key:

```bash
export ANTHROPIC_BASE_URL=https://openrouter.ai/api   # the bare /api origin - Claude Code appends /v1/messages itself
export ANTHROPIC_AUTH_TOKEN="$OPENROUTER_API_KEY"
export ANTHROPIC_API_KEY=                             # present AND empty - this is what keeps the run off your own Anthropic account
# then select a GLM slug, e.g.:  --models claude-code:z-ai/glm-5.2
```

Two caveats on this path, both **measured** rather than theoretical:

- **`total_cost_usd` is wrong on any non-first-party base URL.** Claude Code prices the tokens as if they were first-party Anthropic (`"provider": "firstParty"`), so a run reporting `total_cost_usd: 0.2061` really spent `$0.0204`. Only the key's own usage delta is a spend figure on this path.
- **A GLM slug is not an Anthropic model,** and OpenRouter's own cookbook warns that Claude Code "may not work correctly with other providers". A tool-calling failure on this recipe may be the model rather than the wiring.

**R2 - the direct Z.ai Anthropic endpoint (documented, not live-verified (vendor docs)).** Needs a Z.ai Coding Plan or pay-as-you-go key:

```bash
export ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic
export ANTHROPIC_AUTH_TOKEN=<your Z.ai key>
export ANTHROPIC_DEFAULT_OPUS_MODEL=glm-5.2
export ANTHROPIC_DEFAULT_SONNET_MODEL=glm-5.2
export ANTHROPIC_DEFAULT_HAIKU_MODEL=glm-5.2-air
```

**R3 - the Z.ai OpenAI-compatible coding endpoint (documented, not live-verified (vendor docs)).** For `opencode` / `goose` / `cline` / `kilocode`, point each CLI's own base-URL and key variables (see that CLI's documentation - this plugin does not set them) at `https://api.z.ai/api/coding/paas/v4`. Note that the **OpenRouter** half of R3 needs no endpoint recipe at all: the shipped `kilocode`, `cline`, `goose`, `aider` and `pi` blocks already list `openrouter/z-ai/glm-5.2` entries, so "GLM through those CLIs" is a model-selection line, not an endpoint recipe.

Two rules span all three recipes:

- **Pin models with `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL`, never `ANTHROPIC_MODEL` / `ANTHROPIC_SMALL_FAST_MODEL`.** The latter two are stripped from every detached run, so a recipe built on them works interactively and then silently falls back to the vendor default under `--detach`.
- **Z.ai restricts *Coding Plan subscription* keys to its officially supported tool list.** Pay-as-you-go keys, or the existing OpenRouter lanes, are the safe default - a second reason R1 is the recommended path, on top of being the one that is actually verified here.

---

## Per-project configuration

`providers.yaml` (above) is the **base** layer shared by every repo. To give one
repository its own provider/model defaults — without editing the installed
plugin — add an optional, auto-discovered override file at
`<git-root>/.multi-llm/providers.yaml`. When it's absent, behavior is unchanged.

It deep-merges its full contents over the base and inherits everything you omit,
so it can stay tiny — change just the *selection* keys (`default_provider`,
`defaults.models`, `defaults.quick_models`, `defaults.modes`) and leave the rest:

```yaml
# .multi-llm/providers.yaml — only what this repo changes
default_provider: claude-code
defaults:
  models:
    - claude-code:fable
    - cursor-agent:composer-2.5
  quick_models:
    - claude-code:fable
```

Create one with `/multi-llm:multi-llm --init`. This is **fully automatic and
zero-prompt** — it works the same in a terminal, inside Claude Code, or in CI. It
scans your `PATH` for the supported provider CLIs and writes a preconfigured
override: starting from an inert template, it **uncomments** the full `providers:`
block, `defaults.models` / `quick_models` entries, and `default_provider` for each
CLI it detects, leaving the rest commented. `default_provider` is set to the
**first detected provider** in base order (claude-code, cursor-agent, gemini,
grok, opencode, codex, kilocode, kimi, qwen, copilot, cline, goose, aider, agy, pi, muse, vibe). With **no CLIs installed** it writes the inert
template (which inherits the built-in defaults) and exits 0 with an install
notice — it never errors. Pass `--template-only` to skip detection and write the
pristine commented stub for hand-editing. Add `--gitignore` to keep the file
developer-local, and `--force` to overwrite an existing one — overwriting first
backs up the previous file alongside as `providers.yaml.bak.<timestamp>` (and
prints the path), so you can port any custom settings into the regenerated file.

### Good to know

- **Precedence (low → high):** base `providers.yaml` → project-local
  `.multi-llm/providers.yaml` → `MULTI_LLM_PROVIDERS_CONFIG=/path.yaml` env override.
  Each layer is deep-merged over the one below.
- **Lists replace, they don't append.** Set `defaults.models` and you get exactly
  that list. Omitting or blanking a key inherits the base; an explicit empty list
  `[]` is the only way to clear one — and an empty `defaults.models`/`quick_models`
  falls back to interactive selection (or fails in unattended runs like `--full`).
- **Per-mode lists win.** Modes with a base `defaults.modes.<mode>` entry (e.g.
  `--review-plan`, `--review-code`) ignore a project-wide `defaults.models` —
  override `defaults.modes.<mode>` to change those.
- **Requires a git repo.** Discovery resolves the git root, so there's one override
  per repo, shared repo-wide (no per-subdirectory configs). Non-git projects must
  use `MULTI_LLM_PROVIDERS_CONFIG` instead.
- **Commit or ignore?** Commit it for a team-wide standard; ignore it (`--gitignore`)
  or use the env override for personal preferences.

### Safety

A repo's override is auto-loaded, so a cloned repo can change which models a run
selects — but never *what executes*: provider binaries are hardcoded and the config
`command` field is never run, in any layer. The override now deep-merges its full
contents (including a `providers:` block) over base, exactly like the
`MULTI_LLM_PROVIDERS_CONFIG` env layer — but since `command` is documentation-only,
a merged block can only retune metadata (timeouts, model lists), and a provider
name with no hardcoded adapter is ignored at runtime. A present-but-malformed
explicit override fails fast with a clear error naming the file; set
`MULTI_LLM_PROVIDERS_CONFIG_PERMISSIVE=1` to warn-and-skip instead.

---

## How it works

- **Orchestrators emit instructions, not edits.** Each mode runs a Python orchestrator that queries the selected models in parallel, validates and consolidates their output, and prints structured instructions (and JSON) for Claude Code to act on.
- **State is plan-local.** Workflow state lives in `{plan}/state.json` and all outputs are written next to the plan, so everything is portable and scoped to the plan it belongs to.
- **Relocatable paths.** The skill references its own bundled files through `${CLAUDE_SKILL_DIR}`, so it works wherever Claude Code installs the plugin.

The workflow contract lives in the host-neutral instruction core under [`skills/multi-llm/instructions/`](skills/multi-llm/instructions/); [`skills/multi-llm/SKILL.md`](skills/multi-llm/SKILL.md) is the Claude Code entry shim over it, [`skills/multi-llm/references/host-capabilities.md`](skills/multi-llm/references/host-capabilities.md) is the generated host matrix, and [`skills/multi-llm/AGENTS.md`](skills/multi-llm/AGENTS.md) has the architecture notes.

---

<!-- BEGIN GENERATED: host-install (hosts/render_capabilities.py) -->

## Hosting multi-llm in other harnesses

The workflow itself is host-neutral: the instruction core names three capability *verbs* (*spawn subagents*, *ask the user*, *run detached*) and no tool, and one dispatcher command runs every mode. What a host needs is therefore just an **entry shim** — a small file, in that host's own command/skill format, with this skill's absolute path and its `--host` id baked in. `install-host` renders and writes it:

```bash
# from anywhere; SKILL is the installed skill directory
uv run --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host <host> [--global|--project]
```

It writes a **rendered copy** — never a symlink or junction, so nothing here needs symlink privileges on Windows — atomically, then verifies what it wrote and rolls back if the verification fails. `--check` re-runs that verification on its own: shim present (every copy found, in the host's discovery order, naming the one that loads), baked skill-dir prefix still current, rendered shape valid, shim generation current, and `uv` plus the host's runner CLI resolvable.

### Per-host quickstart

**`codex`**

```bash
uv run --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host codex            # machine-wide
uv run --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host codex --project  # just this repository
uv run --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host codex --check    # verify an install
```

Writes `$HOME/.codex/skills/multi-llm/SKILL.md` (`--project`: `.codex/skills/multi-llm/SKILL.md`), then start a new codex session afterwards (it enumerates its commands/skills at session start). Invoke it as `/multi-llm <the same arguments the Claude Code skill takes>`; unattended is the default on every host but Claude Code, so `--yes` is baked in.

**`opencode`**

```bash
uv run --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host opencode            # machine-wide
uv run --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host opencode --project  # just this repository
uv run --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host opencode --check    # verify an install
```

Writes `$XDG_CONFIG_HOME/opencode/commands/multi-llm.md` (whose `$XDG_CONFIG_HOME` falls back to `~/.config` when unset) (`--project`: `.opencode/commands/multi-llm.md`), then start a new opencode session afterwards (it enumerates its commands/skills at session start). Invoke it as `/multi-llm <the same arguments the Claude Code skill takes>`; unattended is the default on every host but Claude Code, so `--yes` is baked in.

Headless, the argument list needs the host's own separator: `opencode run --command multi-llm -- "<args>"`.

### Capability matrix

Generated from `skills/multi-llm/hosts/capabilities.yaml` — the same data `install-host` renders a shim from. Every value carries a **provenance**: `verified` means a recorded live run observed it, `documented` means host docs only, `unknown` is the shipping default, and a `documented`-only value never selects the non-conservative path (which is why hosts whose native subagent dispatch is merely documented ship as `executor`).

| Host | Installable | Command format | Args | *spawn subagents* | *ask the user* | *run detached* | Default runner | Reload |
|---|---|---|---|---|---|---|---|---|
| `claude-code` | no (ships with the plugin) | `markdown+frontmatter` | `placeholder` | `native` (`Task`) — verified | `structured` (`AskUserQuestion`) — verified | `native` (`run_in_background`) — verified | `claude-code:sonnet` | `none` |
| `codex` | yes | `markdown+frontmatter` | `placeholder` | `executor` (`agent`) — documented | `free-text` — documented | `python` — documented | `codex:gpt-5.2-codex` | `new-session` |
| `opencode` | yes | `markdown+frontmatter` | `placeholder` | `executor` (`task`) — documented | `free-text` — documented | `required` — verified | `opencode:opencode/sonnet` | `new-session` |

### Where the shim goes

One row per host **and scope**, not one path per host. `install-host` writes only the preferred path of the scope you asked for; it *detects and reports* the rest, so an existing install is upgraded rather than silently shadowed. Compatibility locations stay valid inputs and are never written: migrating is re-run `install-host`, then delete the file it names. `~` is `%USERPROFILE%` on Windows and the XDG roots honour `$XDG_CONFIG_HOME`; the tokens below are what the data stores.

| Host | Global (preferred) | Project (preferred) | Compatibility (detected, never written) |
|---|---|---|---|
| `claude-code` | *n/a* — the plugin's own `SKILL.md` | *n/a* | `.claude/skills/multi-llm`, `$HOME/.claude/skills/multi-llm` (non-plugin install) |
| `codex` | `$HOME/.codex/skills/multi-llm` | `.codex/skills/multi-llm` | `$HOME/.codex/prompts/multi-llm.md` |
| `opencode` | `$XDG_CONFIG_HOME/opencode/commands`, `$HOME/.config/opencode/commands` | `.opencode/commands` | — |

### After a plugin update

A plugin update **changes the skill directory path** — the Claude plugin cache is keyed by commit hash — so every installed shim's baked prefix goes stale. That is the common case, not an edge case, and it is one command:

```bash
uv run --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host --check --all   # what moved
uv run --project "$SKILL" -- python "$SKILL/multi_llm.py" install-host --all           # re-render every installed shim
```

Repair is always re-render, never in-place patching, and a re-install with nothing changed is byte-identical and does nothing. A shim you edited by hand is **never** overwritten silently: the install stops and names the file, and `--force` keeps your version as `<shim>.local.bak`. The inverse is `uninstall-host <host> [--global|--project|--all] [--dry-run]`, which removes only files carrying our `multi_llm_shim` block — identified by that block, never by filename.

<!-- END GENERATED: host-install -->

---

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

This plugin is maintained by [BeastLab.ai](https://beastlab.ai) - a frontier lab building multi-agent reasoning models.

**Disclaimer:** BeastLab.ai provides this plugin as-is, with no warranties or liabilities of any kind. BeastLab.ai is not affiliated with, endorsed by, or sponsored by any of the code harnesses (Cursor, Claude Code, Codex, OpenCode, Gemini CLI, Grok Build, Kilo Code, Cline, goose, Aider, Antigravity CLI, Pi, Muse Code, Kimi Code, Qwen Code, GitHub Copilot CLI, Mistral Vibe, etc.) or third-party LLM providers (Anthropic, OpenAI, Google, xAI, Moonshot AI, Z.ai/Zhipu, MiniMax, NVIDIA, Alibaba/Qwen, DeepSeek, Mistral AI, Xiaomi, Arcee, etc.) referenced in this project. Trademarks and product names belong to their respective owners.

One harness is not open source: **GitHub Copilot CLI is proprietary and is used under the GitHub Copilot CLI EULA**. multi-llm ships an *adapter* for it, never the binary — you install it yourself and accept GitHub's terms directly. The project's own end-to-end test image installs it to exercise that adapter, and that image is build-local by construction: its tags carry no registry component, so `docker push` fails outright, and CI builds no image at all.

HTML reports embed small brand marks to label which harness and which model produced each suggestion — nominative use only, and no endorsement is implied. Each bundled mark is fetched from its vendor's own favicon or brand source (see the provenance comment beside every entry in `utils/html_report_generator.py`'s `_ICON_DEFS`), except Primer Octicons' `copilot-16`, which is MIT-licensed. Where no first-party mark is reachable the report renders a generated letter badge instead of a redrawn or re-hosted logo.

---

## License

[MIT](LICENSE) © beastlabai
