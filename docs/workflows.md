# Workflows in depth

Where each mode writes its output, what `--full` and the review-and-apply modes
do on every run, every option, and how ambiguous findings get decided. Commit
reviews have a page of their own, [Commit reviews](commit-reviews.md). Back to
the [README](../README.md).

---

## Plans and outputs

A **plan** is just a markdown file describing the change you want to make - a feature spec, refactor outline, or design doc. You write it (or have Claude draft it), then point multi-llm at it. There's no required template; the more concrete the plan, the more useful the reviews. Most workflows start from a plan, with two exceptions: `--ask` only needs a plan to give the models context for your question, and the commit modes (`--review-commits`, `--apply-commit-fixes`, `--improve-commits`) need none - they take a plan only as optional read-only context (`--context-plan <path>`, the **context plan**), and in place of a plan they write a generated **commit brief**: the range's commits, their messages and its diff stat, which the review reads as the statement of intent ([Commit reviews](commit-reviews.md)).

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

The commit rows are the one exception to the sibling rule: a commit brief is named after its range (`<base>` and `<head>` are 12-character commit ids; a range that starts at the repository's first commit is `root-<head>`), and its folder beside it holds the `state.json` and phase folders a plan's would. The folder is `code_review.commit_reviews_dir`, `plans/commit-reviews` by default, relative to the repository root ([Per-project configuration](configuration.md#good-to-know)). When multi-llm creates it, it writes a `.gitignore` holding `*` inside it, so briefs and reports never show in `git status`; delete that file to commit them.

Throughout the README and these docs, `{plan}/...` refers to that derived directory. Because everything is stored next to the plan, a workflow is fully portable - move or commit the plan folder and its state travels with it.

## Running the pipeline

- **`--full`** chains the phases end to end: review-plan -> apply-suggestions -> generate-tasks -> review-tasks -> apply-task-suggestions -> implement -> review-code -> apply-code-fixes. It pauses at each apply step to let you approve `needs-human-decision` findings; pass `--yes` (alias `--non-interactive`) to run the whole pipeline fully unattended — zero prompts: non-interactive model selection, a judge agent decides every `needs-human-decision` item, and review-tasks runs automatically. For finer control, add just `--no-confirm` and/or `--agent-decide` (see [below](#letting-the-agent-decide-ambiguous-findings)) instead.
- **Applying code fixes** can happen two ways. Run `--apply-code-fixes` as a standalone pass to apply a previous review's fixes - this is the phase that handles `needs-human-decision` items (prompts, salvage, HTML badges). Or, to apply only the clearly-valid fixes inline during the review itself, add `--apply-fixes` to a `--review-code` run.

## Review and apply in one command

`--improve-plan` and `--improve-code` each run one review and then the apply of its findings, unattended, as one command:

| Command | Runs | The review gets | The apply gets |
| --- | --- | --- | --- |
| `--improve-plan <plan>` | `--review-plan`, then `--apply-suggestions` | `--yes`, plus your `--models` or `--quick` | `--yes --agent-decide --no-confirm` |
| `--improve-code <plan>` | `--review-code`, then `--apply-code-fixes` | `--yes`, plus your `--models` or `--quick` and your `--files` / `--files-from` / `--base-ref` | `--yes --agent-decide --no-confirm`, plus `--base-ref` (yours, else the one the review used) |

- **Models.** The review picks its own list, exactly as when you run it alone: with no flag, `defaults.modes.review-plan` (`--improve-plan`) or `defaults.modes.code-review` (`--improve-code`) when you set one, else `defaults.models`; with `--quick`, that entry's `quick` list, else `defaults.quick_models`. The `--improve-*` modes have no model list of their own, so a `defaults.modes.improve-plan` or `improve-code` entry is read by nothing ([Per-project configuration](configuration.md#good-to-know)). Nothing asks you to choose; `--models` wins over `--quick`.
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

## Options

These flags combine with any mode and are **position-independent** - they may appear before or after the plan path (and, for `--ask`, around the question). The one exception is a commit mode's range, which goes right after the mode word, before any flag ([Commit reviews](commit-reviews.md)). Not every flag applies to every mode.

| Flag | Effect |
| --- | --- |
| `--models <provider:model> ...` | Use exactly these models (variadic), overriding YAML defaults. Each must exist in `providers.yaml`. |
| `--quick` | Use the smaller `quick_models` subset for a faster  run. In `--improve-plan` / `--improve-code` / `--improve-commits` it goes to the review step. |
| `--interactive` | Force two-step model selection (provider, then models), ignoring YAML defaults. |
| `--no-confirm` | Skip confirmation prompts - for unattended/silent runs. |
| `--dry-run` | Show what would happen (tasks/batches) without making changes. Applies to `--implement` and the apply modes, `--apply-commit-fixes` included. |
| `--agent-decide` | In the apply modes, let a judge agent decide every `needs-human-decision` finding instead of prompting you (see [below](#letting-the-agent-decide-ambiguous-findings)). Implied by `--yes`, which the Codex and OpenCode shims bake in. `--claude-decide` / `--let-claude-decide` are accepted legacy spellings. |
| `--force` | **Meaning depends on mode.** In the fan-out review/ask modes (`--review-plan`, `--review-tasks`, `--review-code`, `--review-commits`, `--ask`), resume an interrupted run: keep already-completed per-model results, run only the missing models, and bypass the completed-phase / partial-completion guards. In the apply modes, it's an alias for `--yes` (confirm bulk approval). In `--improve-plan` / `--improve-code` / `--improve-commits`, it starts a new review round, then applies that round's findings ([above](#review-and-apply-in-one-command)). |
| `--rerun-all` | In the fan-out review/ask modes, re-run every model from scratch, discarding any existing per-model results. Combine with `--force` (`--force --rerun-all`) for a fresh full re-run of an already-completed phase. |
| `--focus "<text>"` | Steer a commit review toward a priority (not a filter). Taken by the three commit modes only. Recorded with the commit brief: a later run of the same range keeps it when you leave it out, and a different value (or `--focus ""`, which clears it) restarts the review round. `--apply-commit-fixes` keeps the value the review on record used ([Commit reviews](commit-reviews.md)). |
| `--context-plan <path>` | Give a commit review a plan as read-only context, and check the commits against it too. Taken by the three commit modes only. Copied into the review's folder when the review starts and recorded like `--focus`; type it again to review against an edited file. `--apply-commit-fixes` keeps the context plan the review on record used, and never edits it. |

The apply and implement modes accept additional approval flags (`--yes`, `--approve-all`, `--resume`, `--task`, and more); run `multi_llm.py <mode> --help` for the authoritative list, or read the per-mode files in `skills/multi-llm/instructions/`.

## Letting the agent decide ambiguous findings

When the models disagree or a suggestion is borderline, validation marks it `needs-human-decision`. In the apply modes (`--apply-suggestions`, `--apply-code-fixes`, `--apply-commit-fixes`, `--apply-task-suggestions`) Claude Code prompts you to approve or skip each one through its structured `AskUserQuestion` tool - it is the only host that has one, so on Codex and OpenCode the baked-in `--yes` resolves these without asking ([Hosts](hosts.md)). Every such prompt also offers a **Let the agent decide** option, per item or for a whole batch, that hands just that judgment to a judge agent instead of you.

To skip the prompts entirely and have a judge agent decide **every** `needs-human-decision` finding up front, pass `--agent-decide` - which is what the baked-in `--yes` already does on Codex and OpenCode. The judge is not necessarily Claude: on a host with native subagents (Claude Code) it is the host's own agent, and under `--exec-subagents` (the Codex and OpenCode default) it is the model your session is running on, e.g. `codex:<chat model>` on Codex. A `driver.subagent_runner` you set explicitly (in `<git-root>/.multi-llm/providers.yaml` or the file `MULTI_LLM_PROVIDERS_CONFIG` names) overrides that, and the host's floor runner (`codex:gpt-6-astra` on Codex) is used only when neither is available. The summary names it on a `judge:` line. `--improve-plan`, `--improve-code` and `--improve-commits` imply `--agent-decide` (with `--yes` and `--no-confirm`) for their apply step on every host.

```text
/multi-llm:multi-llm --apply-code-fixes plans/my-feature.md --agent-decide --no-confirm
```

Unlike the blanket approve/skip flags, "Let the agent decide" is a **per-item** judgment that **salvages** partially-valid findings - it trims each to its worthwhile core and applies that, skipping only findings with nothing worth keeping. The generated HTML report badges every finding **Approved / Salvaged / Skipped**. Add `--no-confirm` (as above) for a fully unattended run.

---

Back to the [README](../README.md) · [Commit reviews](commit-reviews.md) · [Per-project configuration](configuration.md)
