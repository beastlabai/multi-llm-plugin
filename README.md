# multi-llm - Harness the harnesses

![multi-llm plugin for Claude Code](https://beastlab.ai/images/multi-llm-plugin-for-claude-code-monster.jpg)

**Wisdom of crowds for your codebase.** Use this plugin to improve planning, implementation, and code review by running the same work through multiple AI coding tools and LLMs in parallel - then consolidating their feedback so you catch bugs, blind spots, and improvements a single model might miss.

The diversity works on two axes: not only do you get different **LLM models** (Opus, GPT, Gemini, Grok, Composer, ...), you also get different **code harnesses** (19 supported CLIs, such as Codex, OpenCode, Cursor Agent, Gemini CLI, and Aider) - each with its own prompting, tooling, and context-gathering behavior. Two harnesses running the *same* model can still surface different findings, so combining both axes widens the crowd further.

📺 **[Short video intro](https://www.youtube.com/watch?v=91lpnxAsyqs)** or **[Full video walkthrough](https://www.youtube.com/watch?v=_ySRt7rr0r8)** of the plugin in action.

Orchestrate **plan reviews, task generation, implementation, and code reviews across multiple LLM providers in parallel** - or ask every model the same free-text question about a plan and get one consolidated answer.

`multi-llm` fans a single workflow out to several CLI-based LLMs ([Cursor Agent](https://cursor.com/docs/cli/overview), [Gemini CLI](https://geminicli.com/docs/get-started/), [Grok Build](https://docs.x.ai/build), [Codex](https://developers.openai.com/codex/cli), [OpenCode](https://opencode.ai/docs/cli/), [Kilocode](https://kilo.ai/docs/cli), [Cline](https://docs.cline.bot/), [goose](https://block.github.io/goose/), [Aider](https://aider.chat/), [Antigravity CLI](https://antigravity.google/docs/cli/using), [Pi](https://pi.dev/docs/latest/usage), [Muse Code](https://dev.meta.ai/docs/muse-code/extending), [Kimi Code](https://code.kimi.com/docs), [Qwen Code](https://qwenlm.github.io/qwen-code-docs/), [GitHub Copilot CLI](https://docs.github.com/en/copilot/concepts/agents/about-copilot-cli), [Mistral Vibe](https://docs.mistral.ai/vibe/), [oh-my-pi](https://github.com/can1357/oh-my-pi), [Vercel fx](https://github.com/vercel-labs/fx), and [Claude Code](https://code.claude.com/docs/en/) itself), validates and consolidates their suggestions, and hands the host structured instructions to apply the results. The orchestrators never modify your code directly - they produce JSON that the host executes through its own tools, so every change stays reviewable.

It installs as a **Claude Code plugin**, and the same workflow runs inside **Codex** and **OpenCode** too - as a Codex plugin, or through `install-host`, which writes a native entry point into either one (the OpenCode npm package is not published yet; until it is, `install-host opencode` is the OpenCode route). See [Installation](#1-installation) and [docs/hosts.md](docs/hosts.md).

---

## 1. Installation

Pick your host (1.1 to 1.3), then configure providers (1.4). The prerequisites in 1.5 apply to every host. Each host subsection has the same shape: install, first run, update.

### 1.1 Claude Code

Installs as a Claude Code plugin. In Claude Code:

```text
/plugin marketplace add beastlabai/multi-llm-plugin
/plugin install multi-llm@beastlabai
/reload-plugins
/reload-skills
/multi-llm:multi-llm --init
```

> **Already added the marketplace before (e.g. in another project)?** `/plugin marketplace add` does **not** refetch an existing marketplace - it silently reuses the locally cached copy, so `/plugin install` will give you a stale version. Run the update steps below instead.

Then invoke the skill (plugin skills are namespaced `plugin:skill`):

```text
/multi-llm:multi-llm plans/my-feature.md
```

> Claude can also load the skill automatically when you ask it to "review my plan with multiple models", "run a multi-LLM code review", etc.

**Update.** The marketplace cache only refreshes when you explicitly ask it to - `/plugin marketplace update` is the step that actually pulls the latest commits from GitHub. Then update (or install) the plugin and reload:

```text
/plugin marketplace update beastlabai
# remove "--scope local" if you installed it as user scope
! claude plugin update multi-llm@beastlabai --scope local
/reload-plugins
/reload-skills
/multi-llm:multi-llm --init --force
```

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
$multi-llm:multi-llm --review-plan plans/my-feature.md --quick
```

A `$` mention asks the model to use the skill, and the model then calls the `multi_llm` tool with your arguments. If it ever reworks them, spell the request out: `Use the multi-llm:multi-llm skill with exactly these arguments: plans/my-feature.md`.

> **Before you install:** the plugin's tool runs outside Codex's exec sandbox, so in a read-only session it refuses the modes that edit your working tree unless `MULTI_LLM_CODEX_UNSANDBOXED_WRITES=1` is set. Despite its name, `MULTI_LLM_CODEX_UNSANDBOXED_WRITES=1` lifts only this tool's own refusal of write-capable modes under a read-only or unreadable session posture. It never lets the Codex processes the orchestrator starts write without a sandbox, and it does not change which sandbox they run under: a Codex editing job refused because Codex's own sandbox cannot start on this host stays refused with it set. Provider API keys must be exported from `~/.profile` or `~/.bash_profile`, because Codex starts the tool with a stripped environment.

**Update.** Run `codex plugin marketplace upgrade`, then `codex plugin add multi-llm@beastlabai`, then start a new session. Uninstalling, platform support and troubleshooting are in [docs/hosts.md](docs/hosts.md#codex-plugin-route).

### 1.3 OpenCode

Installs as the `opencode-multi-llm` npm package: a `/multi-llm` command plus a `multi_llm` tool. Needs opencode 1.19.3 or newer and `uv` on `PATH`.

> **This route is not yet published.** Until the package is on npm, the entry below resolves to nothing, and OpenCode fails silently. Until then, use the file route, `install-host opencode`, described in [docs/hosts.md](docs/hosts.md).

Add the package to `~/.config/opencode/opencode.json` and restart OpenCode:

```json
{ "plugin": ["opencode-multi-llm@0.1.0"] }
```

```text
/multi-llm --init
/multi-llm plans/my-feature.md
```

**Update.** Bump the pinned version and restart. OpenCode resolves an entry once and caches it under that exact string, so an unpinned entry never updates. Local builds, troubleshooting and uninstalling are in [docs/hosts.md](docs/hosts.md#opencode-npm-route).

### 1.4 Configure providers (required before first use)

Edit [`skills/multi-llm/providers.yaml`](skills/multi-llm/providers.yaml) manually to match the code harnesses and models you want to use. The shipped file is a starting point only - keep the providers whose CLIs you have installed, remove the rest, and set `defaults.models` (and optionally `quick_models`) to the `provider:model` pairs you actually run. Without this step, default invocations will call harnesses or models you may not have configured. See [docs/providers.md](docs/providers.md) for the format and available keys.

The optional `fx` provider requires Vercel fx on your `PATH` and
`OPENROUTER_API_KEY`; the test container pins the checksummed 0.0.14 release.
Its `--auto` mode can make a separate billed shell-review call that
is absent from reported token usage; review the [fx notes](docs/providers.md#the-provider-table)
before adding it to your defaults.

> **Where is the file after installing via `/plugin`?** If you installed the plugin (rather than cloning this repo), `providers.yaml` lives inside the installed plugin, under your Claude Code plugin cache - `~/.claude/plugins/cache/<marketplace>/multi-llm/<version-or-commit>/skills/multi-llm/providers.yaml`, where the last segment changes on every update. The quickest way to open it is to ask Claude Code to "open the multi-llm `providers.yaml`": the skill resolves its own install location (via `${CLAUDE_SKILL_DIR}`), so Claude can read and edit the file in place.

> **On the Codex plugin or the OpenCode package**, leave the installed copy alone - the next upgrade replaces it. Run `--init` in each repository instead and edit the `<git-root>/.multi-llm/providers.yaml` it writes ([docs/configuration.md](docs/configuration.md)).

### 1.5 Prerequisites

| Requirement | Why |
| --- | --- |
| [`uv`](https://docs.astral.sh/uv/) on your `PATH` | The orchestrators run as `uv run` Python scripts. uv installs their dependencies (just `pyyaml`) on first use. If `uv` is missing when a mode starts (`--init`, `--review-plan`, `--review-code`, ...), the skill prints the installation instructions from `skills/multi-llm/hosts/uv-missing.txt` and stops (exit 3) — it never installs uv and never offers to. Install location and installer differ per OS: on **Linux/macOS** the official installer (`curl -LsSf https://astral.sh/uv/install.sh \| sh`) puts `uv` in `~/.local/bin/uv`; on **macOS** Homebrew installs to `/opt/homebrew/bin/uv` (Apple Silicon) or `/usr/local/bin/uv` (Intel), and MacPorts to `/opt/local/bin/uv`; on **Windows** the official installer is PowerShell (`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"`) and creates `%USERPROFILE%\.local\bin\uv.exe` — which Git Bash sees as `~/.local/bin/uv.exe` (note the `.exe`); `winget install --id=astral-sh.uv -e` and `pipx install uv` also work. |
| Python >= 3.10 (usually nothing to install) | The orchestrators need Python 3.10 or newer, and `uv` supplies it: if no suitable interpreter is installed, `uv` downloads a managed one on first use, which needs the network once. **Codex plugin only:** the plugin's tool server starts on the system `python3` before `uv` is involved, so python3 >= 3.9 must be on `PATH` under exactly that name. |
| [Git](https://git-scm.com/), and a Git repository to run in | `--review-code`, `--implement` and the apply modes read diffs and stage files with `git`. `--init` writes its config at the Git root, and the per-project config is only discovered inside a Git repository. |
| **Windows only:** [Git for Windows](https://git-scm.com/download/win) (Git Bash) | **Hard prerequisite on Windows.** The skill's entire instruction set and its permission (`allowed-tools`) rules are Bash-only; Claude Code's PowerShell-tool fallback (used when Git for Windows is absent) is explicitly **unsupported** — the skill fails fast with an install pointer rather than attempting PowerShell equivalents. Git for Windows also supplies the bundled coreutils the command blocks depend on (`ls`, `grep`, `diff`, `tail`, `dirname`, ...), which are absent from PowerShell/cmd. Note that `realpath` is *not* one of them: plan paths are passed to the orchestrators as-is (never wrapped in `$(realpath ...)`), and the orchestrators resolve them to OS-native absolute paths themselves. |
| One or more provider CLIs | You only need the CLIs for the models you actually run. See [docs/providers.md](docs/providers.md). |

The plugin ships its own `pyproject.toml`/`uv.lock`, so no manual Python setup is required beyond the Codex plugin's `python3` - `uv` resolves the environment the first time a mode runs.

---

## 2. Plans and outputs

A **plan** is just a markdown file describing the change you want to make - a feature spec, refactor outline, or design doc. You write it (or have Claude draft it), then point multi-llm at it. There's no required template; the more concrete the plan, the more useful the reviews. Most workflows start from a plan, with two exceptions: `--ask` only needs a plan to give the models context for your question, and the commit modes (`--review-commits`, `--apply-commit-fixes`, `--improve-commits`) need none - they take a plan only as optional read-only context (`--context-plan <path>`, the **context plan**), and in place of a plan they write a generated **commit brief**: the range's commits, their messages and its diff stat, which the review reads as the statement of intent ([Review commits](#review-commits)).

The path you pass (e.g. `plans/my-feature.md`) is always a **file**. From it, multi-llm derives a sibling **output directory** in the same folder, named after the plan file without its extension - so `plans/my-feature.md` produces `plans/my-feature/`. All workflow state and generated artifacts live there:

| Path | Written by |
| --- | --- |
| `plans/my-feature/state.json` | every mode (tracks workflow progress) |
| `plans/my-feature/review-plan/` | `--review-plan`, `--apply-suggestions` |
| `plans/my-feature/tasks/` | `--generate-tasks`, `--review-tasks` |
| `plans/my-feature/implement/summary.md` | `--implement` |
| `plans/my-feature/code-review/report.md` | `--review-code`, `--apply-code-fixes` |
| `plans/my-feature/ask/<question-slug>/answers.md` | `--ask` |
| `plans/my-feature/improve-plan/`, `plans/my-feature/improve-code/` | `--improve-plan`, `--improve-code`: the run's own folder (`.status.json`, log and lock); the two phases it runs write their usual folders |
| `plans/commit-reviews/<base>-<head>.md` | `--review-commits`, `--apply-commit-fixes`, `--improve-commits`: the commit brief, one per range |
| `plans/commit-reviews/<base>-<head>/code-review/report.md` (and `report.html`) | `--review-commits`, `--apply-commit-fixes` |
| `plans/commit-reviews/<base>-<head>/improve-commits/` | `--improve-commits`: the run's own folder, as for `--improve-code` |

The commit rows are the one exception to the sibling rule: a commit brief is named after its range (`<base>` and `<head>` are 12-character commit ids; a range that starts at the repository's first commit is `root-<head>`), and its folder beside it holds the `state.json` and phase folders a plan's would. The folder is `code_review.commit_reviews_dir`, `plans/commit-reviews` by default, relative to the repository root ([docs/configuration.md](docs/configuration.md#good-to-know)). When multi-llm creates it, it writes a `.gitignore` holding `*` inside it, so briefs and reports never show in `git status`; delete that file to commit them.

Throughout this README and the linked docs, `{plan}/...` refers to that derived directory. Because everything is stored next to the plan, a workflow is fully portable - move or commit the plan folder and its state travels with it.

---

## 3. Modes

Invoke with `/multi-llm:multi-llm [mode-flag] <plan_path> [options]` - on OpenCode the same command is `/multi-llm`, without the plugin namespace; on Codex it is a skill (model-invoked, not a slash command), so you ask for `multi-llm:multi-llm` by name instead (see [docs/hosts.md](docs/hosts.md)). With no mode flag, `--review-plan` is the default. The three commit modes take an optional `[<range>]` in place of `<plan_path>` ([below](#review-commits)), and `--init` takes neither.

| Flag | What it does |
| --- | --- |
| `--review-plan` | Review an implementation plan with multiple LLMs (default) |
| `--apply-suggestions` | Apply validated review suggestions back into the plan |
| `--improve-plan` | `--review-plan`, then `--apply-suggestions`, in one unattended command ([below](#review-and-apply-in-one-command)) |
| `--generate-tasks` | Generate detailed implementation tasks from a high-level plan |
| `--review-tasks` | Review generated tasks with multiple LLMs |
| `--apply-task-suggestions` | Apply validated task-review suggestions to `tasks.md` |
| `--implement` | Execute implementation tasks via subagent delegation |
| `--review-code` | Review code changes against the plan |
| `--apply-code-fixes` | Apply validated fixes from a code review |
| `--improve-code` | `--review-code`, then `--apply-code-fixes`, in one unattended command that edits source files ([below](#review-and-apply-in-one-command)) |
| `--review-commits` | Review a commit or a range of commits (default: the last commit) ([below](#review-commits)) |
| `--apply-commit-fixes` | Apply validated fixes from a commit review ([below](#review-commits)) |
| `--improve-commits` | `--review-commits`, then `--apply-commit-fixes`, in one unattended command that edits source files ([below](#review-commits)) |
| `--full` | Run the whole pipeline in sequence |
| `--status` | Show workflow state and the suggested next action |
| `--ask` | Ask each model a free-text question about a plan; aggregate the answers |
| `--init` | Write a per-project `.multi-llm/providers.yaml` ([Configuration](#4-configuration)) |

Two notes on the table above:

- **`--full`** chains the phases end to end: review-plan -> apply-suggestions -> generate-tasks -> review-tasks -> apply-task-suggestions -> implement -> review-code -> apply-code-fixes. It pauses at each apply step to let you approve `needs-human-decision` findings; pass `--yes` (alias `--non-interactive`) to run the whole pipeline fully unattended — zero prompts: non-interactive model selection, a judge agent decides every `needs-human-decision` item, and review-tasks runs automatically. For finer control, add just `--no-confirm` and/or `--agent-decide` (see [below](#34-letting-the-agent-decide-ambiguous-findings)) instead.
- **Applying code fixes** can happen two ways. Run `--apply-code-fixes` as a standalone pass to apply a previous review's fixes - this is the phase that handles `needs-human-decision` items (prompts, salvage, HTML badges). Or, to apply only the clearly-valid fixes inline during the review itself, add `--apply-fixes` to a `--review-code` run.

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
- 🟢 **`--full`** runs the entire flow with one command

`--improve-plan` runs the *Implementation Plan* pair and `--improve-code` the *Code Review* pair, each as one unattended command ([below](#review-and-apply-in-one-command)).

You don't have to run the whole chain: **every step is also a standalone entry point** that reads/writes the shared `{plan}/state.json`, so you can start, stop, and resume at any phase. A few modes sit outside the pipeline - **`--ask`** (read-only Q&A about a plan), **`--status`** (show progress and the suggested next step), **`--init`** (write a per-project config), and the commit modes **`--review-commits`**, **`--apply-commit-fixes`** and **`--improve-commits`**, which work on commits rather than a plan ([below](#review-commits)) - while **`--full`** simply runs the entire flow above end to end, and never runs the commit modes.

#### Review and apply in one command

`--improve-plan` and `--improve-code` each run one review and then the apply of its findings, unattended, as one command:

| Command | Runs | The review gets | The apply gets |
| --- | --- | --- | --- |
| `--improve-plan <plan>` | `--review-plan`, then `--apply-suggestions` | `--yes`, plus your `--models` or `--quick` | `--yes --agent-decide --no-confirm` |
| `--improve-code <plan>` | `--review-code`, then `--apply-code-fixes` | `--yes`, plus your `--models` or `--quick` and your `--files` / `--files-from` / `--base-ref` | `--yes --agent-decide --no-confirm`, plus `--base-ref` (yours, else the one the review used) |

- **Models.** The review picks its own list, exactly as when you run it alone: with no flag, `defaults.modes.review-plan` (`--improve-plan`) or `defaults.modes.code-review` (`--improve-code`) when you set one, else `defaults.models`; with `--quick`, that entry's `quick` list, else `defaults.quick_models`. The `--improve-*` modes have no model list of their own, so a `defaults.modes.improve-plan` or `improve-code` entry is read by nothing ([docs/configuration.md](docs/configuration.md#good-to-know)). Nothing asks you to choose; `--models` wins over `--quick`.
- **Consolidation.** `--improve-plan` runs review-plan's consolidation pass between the two steps when the review recommends it, as `--full` does.
- **Re-runs and `--resume`.** Running the command again skips a step already done for the review on record and continues the one that stopped: an unfinished review runs its missing models, and an interrupted apply continues with `--resume` whether or not you type it. Once both steps are done for that review, it runs nothing, prints `[ALREADY_COMPLETE]` and the next command, and exits 2.
- **`--force`** starts a new review round: every model reviews the plan (or the code) as it reads now (`--force --rerun-all`, which replaces the previous round's review files), and the apply then applies that round's findings. A stopped run names the command that continues it, without `--force`; typing `--force` again starts yet another round. `--improve-code` also reviews again with no flag when a newer `--implement` ran after the review, or when the `--files` / `--files-from` you type differ from the files that review covered. A list on stdin (`--files-from -`) is not compared: the completed review stands, and the run says so on a `NOTE:` line (`does not compare --files-from - with the scope on record; the completed code review stands`); pass the list with `--files` or `--files-from <file>`, or add `--force`, to review it.
- **`--interactive` is refused.** To pick models or approve items yourself, run the two phases separately.
- **`--full` remains the whole workflow**; these two run only its first two or its last two phases.

**Before you run it.** Both modes edit files with nobody asked: the mode word is the opt-in, and there is no confirmation and no extra flag.

- *What gets written, and how to undo it.* `--improve-plan` edits the plan in place; the plan as it was before the apply stays in `{plan}/apply-suggestions/backup.md`. After a `--force` round that file is the plan as that round's apply found it, and the earlier round's backup stays beside it as `backup.<sha8>.md`. **`--improve-code` edits source files anywhere in the repository, unattended.** When the run uses `--exec-subagents` (the Codex and OpenCode default) each batch keeps a pre-batch snapshot that `multi_llm.py util restore-batch` restores; nothing backs up the repository as a whole, so git is the way back.
- *Who decides.* A judge agent decides every `needs-human-decision` item, and nobody is asked. The run's summary names the judge and each decision it made.
- *What your tree should be.* Commit or stash first, so `git diff` shows exactly what the run changed and `git restore` undoes it. Nothing refuses a dirty tree. When the apply runs with `--exec-subagents` (the Codex and OpenCode default), it records your uncommitted paths in `{plan}/state.json` (`pre_existing_changes`) as its baseline and names them at its start (`dirty tree at start: N uncommitted path(s) recorded as the baseline …`, the first ten listed), unless the plan already holds a baseline, which it keeps and only counts (an earlier `--implement` records one, before it changed anything). Without `--exec-subagents` (the Claude Code default, where the host runs the batches), the apply records nothing about your tree and says nothing about it.
- *What it costs.* One review call per model in the review's list (`defaults.models`, or the `--quick` or `--models` list), then validation, judging, and one write dispatch per batch. `--quick` is the cheap way to run it; `--force` pays for a whole new review round. `--improve-plan` adds one consolidation pass (one subagent call per consolidation batch) when the review recommends it - 15 or more actionable groups - and that pass changes nothing that gets applied.

#### Review commits

`--review-commits`, `--apply-commit-fixes` and `--improve-commits` are the commit counterparts of the three code modes. They review what was committed, not what is on disk, and need no plan: the commit messages, taken together, are the stated intent, and the range's net change (`git diff <base> <head>`) is what every model reviews.

| Command | Does | Counterpart |
| --- | --- | --- |
| `--review-commits [<range>]` | every model reviews the range; the findings are grouped, validated and reported, each naming the commit that introduced it | `--review-code` |
| `--apply-commit-fixes [<range>]` | applies that review's validated fixes as uncommitted edits | `--apply-code-fixes` |
| `--improve-commits [<range>]` | the two above in one unattended command | `--improve-code` |

Nothing commits, amends or rebases: the fixes are uncommitted edits on top of `HEAD`, for you to read and commit.

**The range.** Python resolves it, the same way on every host. Leave it out to review the last commit:

| You type | Reviews |
| --- | --- |
| nothing | the last commit, against its first parent |
| `<rev>`: `abc1234`, `HEAD~2`, a tag, a branch | that one commit |
| `A..B`, `A..`, `..B` | the commits after `A` up to `B`; an omitted side is `HEAD`, and `A` must be an ancestor of `B` |
| `A...B` | the commits on `B` since it forked from `A` |
| `last commit`, `last N`, `last N commits`, `N commits`, `-N` | the `N` newest commits, ending at `HEAD` |
| `today`, `yesterday`, `this week`, `last N days` (or `hours`, `weeks`; `past` works as `last`), `since YYYY-MM-DD` (optionally with `HH:MM`) | the commits in that window, by committer date in your local time zone |
| `since <rev>` | the commits after `<rev>` up to `HEAD` |
| `this branch`, `branch` | the commits since `HEAD` forked from the default branch (`origin/HEAD`, else `main`, else `master`) |

- **`N`** is a positive whole number, or `one` to `twelve`, so `last two days` works. Commits a merge brought in are reviewed as part of that merge commit (first parent only).
- **Anything else is refused** at exit 3 with `[RANGE_UNRECOGNIZED]` and this list of forms; nothing is guessed. A range with no commits in it (an empty window, `A..A`) exits 2 with `[NOTHING_TO_REVIEW]`. A markdown path is never taken as a plan here: to review commits against one, pass it with `--context-plan`.
- **The range goes right after the mode word, before any flag**, and needs no quotes: `--review-commits last 2 commits --quick`. A word after a flag may be that flag's value (`--models` takes several), so range words typed after one are refused with `[RANGE_AFTER_FLAGS]`, naming the command with the range moved, never reviewed as `HEAD`.
- **One range per run.** To cover two, type one `A..B` that spans them, or run once per range. `--files` narrows a review to some of the range's files and never widens it.

**What it prints first.** Before any model runs, every commit mode prints the commits it acts on, one line each, with the range's size:

```text
[COMMIT_RANGE] a1b2c3d..e4f5a6b (3 commits, 14 files, +412 −88)
  e4f5a6b  2026-10-24  Wait for the clock to tick in the shared-ps-table row
  c3d4e5f  2026-10-24  Windows: kill process trees through Job Objects, not…
  b2c3d4e  2026-10-23  docs: slim skill development guide and extract reference…
[COMMIT_RANGE_END]
```

The block shows the 20 newest commits, then where the full list is. A date phrase adds its window, and a range over 50 commits, 200 files or 20000 changed lines adds a `[RANGE_LARGE]` line naming ways to narrow it; the run continues. `[COMMIT_BRIEF] <path>` follows, naming the commit brief. Every follow-up command a run prints names the range pinned as `<base12>..<head12>` (the commit brief's path for a range that starts at the first commit), never the phrase you typed, so following it tomorrow reaches the same review; `--status` takes that pinned range too.

**Reviewing before applying.** `--review-commits` writes the same two reports as `--review-code`, in `plans/commit-reviews/<base>-<head>/code-review/`. Both open with the commit block, and each finding names the commit that introduced it ("Introduced by `e4f5a6b`" and its subject, marked "(model)" when `git blame` could not decide and the model's claim was taken); `report.html` also filters and groups the findings by commit. Tick `Skip` or a `Mark valid` / `Mark invalid` override in `report.md`, or make the same choices in `report.html` and export `user_selections.json` next to it, exactly as after `--review-code`, then run `--apply-commit-fixes`:

- **With no range,** it applies the newest completed commit review when that review ends at `HEAD` or behind it in `HEAD`'s history, else the newest one ending at `HEAD`, and names its pick on a `[COMMIT_REVIEW_SELECTED]` line before any fix is applied. An unfinished review is never picked. With no match, it exits 3 listing up to five completed reviews, each with its relation to `HEAD` and its command.
- **With a range,** it applies that range's review, which must have completed. A phrase is resolved again when the apply runs, so a day later `last 2 days` names other commits; the pinned range the review printed always names the reviewed one.

`--improve-commits` skips this step: it is unattended, and a judge agent decides every `needs-human-decision` item. For commits you do not fully trust, such as a fetched pull request, `--review-commits`, then the report, then `--apply-commit-fixes` is the safer path.

**`--focus` and `--context-plan`.** `--focus "<text>"` steers the review toward a priority; it is not a filter. `--context-plan <path>` hands the review a plan as read-only context, and the commits are checked against it too.

- **Recorded with the commit brief.** A later run of the same range that leaves them out keeps them, so the printed follow-ups, `--resume` and a no-argument `--apply-commit-fixes` all continue the same review. Typing a different value, or `""` to clear one, restarts the review round (`[COMMIT_REVIEW_RESTARTED]`): every model reviews again, and the replaced round, its reports and your marks in them move to `code-review/restarted-<stamp>/`, with a `[COMMIT_DECISIONS_DROPPED]` line counting the decisions not carried over.
- **The context plan is copied when the review starts.** Every reviewer and validator reads that copy, so editing the file later changes nothing in that review: the next `--review-commits` or `--improve-commits` of the range warns with `[CONTEXT_PLAN_CHANGED]` and carries on with the copy. To review against the edited text, type `--context-plan <path>` again.
- **`--apply-commit-fixes` keeps what the review used.** It never restarts a review: a different value typed there prints `[COMMIT_REVIEW_KEPT]`, with the commands that would review with it, and applies the review on record. A fix aimed at the context plan itself is skipped and named, never applied.

**When files changed since the reviewed commit.** The review reads every file at the range's last commit, never from your working tree, so what is on disk does not change what is reviewed. When reviewed files differ on disk from that commit (later commits, uncommitted edits), the commit block adds a line naming them, `[SCOPE_DRIFT] 2 of 14 reviewed files differ on disk from e4f5a6b: utils/process_control.py, tests/test_detach.py`, and the run continues. The apply compares each fix's files with the reviewed commit before anything is written, and again just before each batch: a finding with a file changed since is skipped whole and named in the `[COMMIT_FIXES]` block the apply prints before its first batch, and no batch may write a file that differed from the reviewed commit when the apply started. So after a further commit, `--apply-commit-fixes` still applies every fix to the files that commit left alone.

**Before you run it.** `--review-commits` writes only the commit brief and its folder. The other two edit source files, as `--improve-code` does:

- *What gets written, and how to undo it.* `--apply-commit-fixes` and `--improve-commits` edit source files anywhere in the repository as uncommitted edits, `--improve-commits` unattended. When the run uses `--exec-subagents` (the Codex and OpenCode default), each batch keeps a pre-batch snapshot that `multi_llm.py util restore-batch` restores; nothing backs up the repository as a whole, so git is the way back. Commit or stash first, so `git diff` shows exactly what the run changed and `git restore` undoes it.
- *Re-runs and `--resume`.* `--improve-commits` follows `--improve-code`'s rules ([above](#review-and-apply-in-one-command)), and its apply always gets the reviewed range's base. A stopped run names `--improve-commits <base12>..<head12> --resume`, which never re-runs a completed review and keeps every batch already applied; once both steps are done it prints `[ALREADY_COMPLETE]` and exits 2. `--interactive` is refused: run `--review-commits --interactive`, then `--apply-commit-fixes`.
- *What it costs.* As for the code modes: one review call per model in the review's list, then validation, judging and one write dispatch per batch. A broad phrase such as `this branch` adds no calls, but each reviewer has more to read and covers it more thinly, which `[RANGE_LARGE]` warns about.

### 3.2 Examples

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

# Review the code changes, then apply the validated fixes to the source files, unattended
/multi-llm:multi-llm --improve-code plans/my-feature.md

# Review your last commit (no plan needed)
/multi-llm:multi-llm --review-commits

# Review the last two commits (no quotes; the range goes before any flag)
/multi-llm:multi-llm --review-commits last 2 commits

# Review the commits after main, with error handling as the priority
/multi-llm:multi-llm --review-commits main..HEAD --focus "error handling"

# Apply the validated fixes of the commit review you just ran
/multi-llm:multi-llm --apply-commit-fixes

# Review this branch's commits, then apply the validated fixes, unattended
/multi-llm:multi-llm --improve-commits this branch

# Review/implement/code-review a plan using the configured default LLMs completely hands-off
/multi-llm:multi-llm --full plans/my-feature.md --agent-decide --non-interactive

# Ask every model the same question (read-only Q&A)
/multi-llm:multi-llm --ask plans/my-feature.md "Is the rollback strategy sufficient?"
```

### 3.3 Options

These flags combine with any mode and are **position-independent** - they may appear before or after the plan path (and, for `--ask`, around the question). The one exception is a commit mode's range, which goes right after the mode word, before any flag ([above](#review-commits)). Not every flag applies to every mode.

| Flag | Effect |
| --- | --- |
| `--models <provider:model> ...` | Use exactly these models (variadic), overriding YAML defaults. Each must exist in `providers.yaml`. |
| `--quick` | Use the smaller `quick_models` subset for a faster  run. In `--improve-plan` / `--improve-code` / `--improve-commits` it goes to the review step. |
| `--interactive` | Force two-step model selection (provider, then models), ignoring YAML defaults. |
| `--no-confirm` | Skip confirmation prompts - for unattended/silent runs. |
| `--dry-run` | Show what would happen (tasks/batches) without making changes. Applies to `--implement` and the apply modes, `--apply-commit-fixes` included. |
| `--agent-decide` | In the apply modes, let a judge agent decide every `needs-human-decision` finding instead of prompting you (see [below](#34-letting-the-agent-decide-ambiguous-findings)). Implied by `--yes`, which the Codex and OpenCode shims bake in. `--claude-decide` / `--let-claude-decide` are accepted legacy spellings. |
| `--force` | **Meaning depends on mode.** In the fan-out review/ask modes (`--review-plan`, `--review-tasks`, `--review-code`, `--review-commits`, `--ask`), resume an interrupted run: keep already-completed per-model results, run only the missing models, and bypass the completed-phase / partial-completion guards. In the apply modes, it's an alias for `--yes` (confirm bulk approval). In `--improve-plan` / `--improve-code` / `--improve-commits`, it starts a new review round, then applies that round's findings ([above](#review-and-apply-in-one-command)). |
| `--rerun-all` | In the fan-out review/ask modes, re-run every model from scratch, discarding any existing per-model results. Combine with `--force` (`--force --rerun-all`) for a fresh full re-run of an already-completed phase. |
| `--focus "<text>"` | Steer a commit review toward a priority (not a filter). Taken by the three commit modes only. Recorded with the commit brief: a later run of the same range keeps it when you leave it out, and a different value (or `--focus ""`, which clears it) restarts the review round. `--apply-commit-fixes` keeps the value the review on record used ([above](#review-commits)). |
| `--context-plan <path>` | Give a commit review a plan as read-only context, and check the commits against it too. Taken by the three commit modes only. Copied into the review's folder when the review starts and recorded like `--focus`; type it again to review against an edited file. `--apply-commit-fixes` keeps the context plan the review on record used, and never edits it. |

The apply and implement modes accept additional approval flags (`--yes`, `--approve-all`, `--resume`, `--task`, and more); run `multi_llm.py <mode> --help` for the authoritative list, or read the per-mode files in `skills/multi-llm/instructions/`.

### 3.4 Letting the agent decide ambiguous findings

When the models disagree or a suggestion is borderline, validation marks it `needs-human-decision`. In the apply modes (`--apply-suggestions`, `--apply-code-fixes`, `--apply-commit-fixes`, `--apply-task-suggestions`) Claude Code prompts you to approve or skip each one through its structured `AskUserQuestion` tool - it is the only host that has one, so on Codex and OpenCode the baked-in `--yes` resolves these without asking ([docs/hosts.md](docs/hosts.md)). Every such prompt also offers a **Let the agent decide** option, per item or for a whole batch, that hands just that judgment to a judge agent instead of you.

To skip the prompts entirely and have a judge agent decide **every** `needs-human-decision` finding up front, pass `--agent-decide` - which is what the baked-in `--yes` already does on Codex and OpenCode. The judge is not necessarily Claude: on a host with native subagents (Claude Code) it is the host's own agent, and under `--exec-subagents` (the Codex and OpenCode default) it is the model your session is running on, e.g. `codex:<chat model>` on Codex. A `driver.subagent_runner` you set explicitly (in `<git-root>/.multi-llm/providers.yaml` or the file `MULTI_LLM_PROVIDERS_CONFIG` names) overrides that, and the host's floor runner (`codex:gpt-6-astra` on Codex) is used only when neither is available. The summary names it on a `judge:` line. `--improve-plan`, `--improve-code` and `--improve-commits` imply `--agent-decide` (with `--yes` and `--no-confirm`) for their apply step on every host.

```text
/multi-llm:multi-llm --apply-code-fixes plans/my-feature.md --agent-decide --no-confirm
```

Unlike the blanket approve/skip flags, "Let the agent decide" is a **per-item** judgment that **salvages** partially-valid findings - it trims each to its worthwhile core and applies that, skipping only findings with nothing worth keeping. The generated HTML report badges every finding **Approved / Salvaged / Skipped**. Add `--no-confirm` (as above) for a fully unattended run.

---

## 4. Configuration

multi-llm calls the code harnesses and models you tell it to, in two layers:

| Layer | File | What it's for |
| --- | --- | --- |
| Base | `skills/multi-llm/providers.yaml`, inside the installed plugin | The provider/model catalog and the defaults every repo inherits. Edit it once, after installing. |
| Per-project | `<git-root>/.multi-llm/providers.yaml` - optional, auto-discovered | One repository's own defaults, without editing the installed plugin. Write it with `--init`. |

A model is referenced as `provider:model` (e.g. `codex:gpt-5.5`); a bare model name uses `default_provider`. Selection priority, highest first: `--models`, then `--interactive`, then `--quick`, then `defaults.models` from YAML, then interactive fallback.

- **[docs/providers.md](docs/providers.md)** - the provider table (every supported CLI, its binary, its measured quirks), `provider:model` naming, and how to point a provider CLI at a different endpoint (GLM / Z.ai worked recipes).
- **[docs/configuration.md](docs/configuration.md)** - the per-project override: what `--init` writes, layer precedence, list-replacement semantics, and why an auto-loaded override can never change *what executes*.

---

## 5. How it works

- **Orchestrators emit instructions, not edits.** Each mode runs a Python orchestrator that queries the selected models in parallel, validates and consolidates their output, and prints structured instructions (and JSON) for the host to act on.
- **State is plan-local.** Workflow state lives in `{plan}/state.json` and all outputs are written next to the plan, so everything is portable and scoped to the plan it belongs to.
- **Relocatable paths.** The skill references its own bundled files through `${CLAUDE_SKILL_DIR}`, so it works wherever Claude Code installs the plugin; a shim installed into another host by `install-host` bakes the same directory in, which is why a plugin update needs one `install-host --all`. The Codex plugin and the OpenCode npm package carry their own copy of the skill and need no such step.

The workflow contract lives in the host-neutral instruction core under [`skills/multi-llm/instructions/`](skills/multi-llm/instructions/); [`skills/multi-llm/SKILL.md`](skills/multi-llm/SKILL.md) is the Claude Code entry shim over it, [`docs/hosts.md`](docs/hosts.md) and [`skills/multi-llm/references/host-capabilities.md`](skills/multi-llm/references/host-capabilities.md) are the generated host documents, and [`skills/multi-llm/AGENTS.md`](skills/multi-llm/AGENTS.md) has the architecture notes.

---

## 6. Your project repository

The repo where you run multi-llm workflows should include one or more `AGENTS.md` files - at the repo root and/or in subdirectories - with project-specific guidance for AI agents (conventions, architecture, pitfalls, etc.). `AGENTS.md` is supported by most code harnesses (Cursor, Codex, OpenCode, Gemini CLI, and others), so the external models multi-llm invokes pick up the same context.

To keep a single source of truth while still working with Claude Code, add a sibling `CLAUDE.md` that contains only:

```text
@AGENTS.md
```

Claude Code includes the referenced file automatically. This plugin follows that pattern in [`skills/multi-llm/AGENTS.md`](skills/multi-llm/AGENTS.md) and [`skills/multi-llm/CLAUDE.md`](skills/multi-llm/CLAUDE.md).

---

## 7. Development

Tests, the containerized end-to-end harness (real provider CLIs against stubbed backends) and the plugin validator are documented in **[docs/development.md](docs/development.md)**.

```bash
uv run --project skills/multi-llm -- pytest   # the test suite, from the repo root
make e2e-container                            # containerized E2E, provider lane (needs Docker)
```

---

## 8. Documentation

| Document | What's in it |
| --- | --- |
| [docs/providers.md](docs/providers.md) | Every supported harness CLI, `provider:model` naming, and endpoint recipes (GLM / Z.ai) |
| [docs/configuration.md](docs/configuration.md) | The per-project `.multi-llm/providers.yaml` override, layer precedence, and its safety model |
| [docs/hosts.md](docs/hosts.md) | Installing into Codex and OpenCode in full: paths, capability matrix, verification, uninstall |
| [docs/development.md](docs/development.md) | Test suite, the containerized E2E harness (Docker), and plugin validation |
| [docs/attribution.md](docs/attribution.md) | Trademark disclaimer, the one proprietary harness, and HTML-report brand marks |
| [`skills/multi-llm/AGENTS.md`](skills/multi-llm/AGENTS.md) | Architecture notes for anyone changing the skill itself |

---

## 9. Roadmap

Planned improvements and open ideas are tracked in [`TODO.md`](TODO.md).

Contributions are welcome - pull requests, bug reports, feature suggestions, and documentation improvements. Open an issue to discuss a larger change before starting work, or send a PR directly for smaller fixes. See [docs/development.md](docs/development.md) for running tests locally.

---

## 10. About BeastLab.ai

This plugin is maintained by [BeastLab.ai](https://beastlab.ai) - a frontier lab building multi-agent reasoning models.

**Disclaimer:** this plugin is provided as-is, with no warranties or liabilities of any kind, and BeastLab.ai is not affiliated with, endorsed by, or sponsored by any of the code harnesses or LLM providers it talks to. The full notice - including the one proprietary harness (GitHub Copilot CLI, used under its own EULA) and the provenance of the brand marks in HTML reports - is in [docs/attribution.md](docs/attribution.md).

---

## 11. License

[MIT](LICENSE) © beastlabai
