# Workflows in depth

Where each mode writes its output, what `--full` and the review-and-apply modes
do on every run, every option, how ambiguous findings get decided, and how the
scope policy keeps a run to its agreed scope. Commit reviews have a page of their
own, [Commit reviews](commit-reviews.md). Back to the [README](../README.md).

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
| `plans/my-feature/improve-plan/`, `plans/my-feature/improve-code/` | `--improve-plan`, `--improve-code`: the run's own folder (`.status.json`, log and lock), and with `--cascade` each cascade's per-round files ([below](#cascade-one-model-at-a-time)); the two phases it runs write their usual folders |
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
- *What to check afterwards.* A writer that meets a lint or test check can quiet it instead of satisfying it. After every batch of `--improve-code`, `--apply-code-fixes`, `--apply-commit-fixes` and `--implement`, the plugin scans what the batch's edits added, on both legs, and lists under `## Check these: suppressions added` in the phase's `summary.md` each suppression or skip marker (`# noqa`, `# type: ignore`, `# pragma: no cover`, `eslint-disable`, `@ts-ignore`, `@ts-expect-error`, `//nolint`, a skipped or xfailed test) and each changed lint, type-checker or test configuration entry, with its `file:line` and whether a reason sits beside it. The run's result names them too. Nothing there fails the run: read each one, and keep it only where the code is right as written.
- *What your tree should be.* Commit or stash first, so `git diff` shows exactly what the run changed and `git restore` undoes it. Nothing refuses a dirty tree. When the apply runs with `--exec-subagents` (the Codex and OpenCode default), it records your uncommitted paths in `{plan}/state.json` (`pre_existing_changes`) as its baseline and names them at its start (`dirty tree at start: N uncommitted path(s) recorded as the baseline …`, the first ten listed), unless the plan already holds a baseline, which it keeps and only counts (an earlier `--implement` records one, before it changed anything). Without `--exec-subagents` (the Claude Code default, where the host runs the batches), the apply records nothing about your tree and says nothing about it.
- *What it costs.* One review call per model in the review's list (`defaults.models`, or the `--quick` or `--models` list), then validation, judging, and one write dispatch per batch. `--quick` is the cheap way to run it; `--force` pays for a whole new review round. `--improve-plan` adds one consolidation pass (one subagent call per consolidation batch) when the review recommends it - 15 or more actionable groups - and that pass changes nothing that gets applied.

### Cascade: one model at a time

`--cascade` changes how `--improve-plan` and `--improve-code` use your model list. Without it, every model reviews in parallel and the merged findings are applied once. With it, each model gets a round of its own, one after another: it reviews the plan (or the code) as the previous round left it, and its own validated findings are applied before the next model starts.

```text
--improve-plan plans/feature.md --cascade --models <cheap> <mid> <top>

round 1: <cheap> reviews the plan; its findings are validated and applied
round 2: <mid> reviews the result; its findings are validated and applied
round 3: <top> reviews the result; its findings are validated and applied
```

**Why use it.**

- *No duplicated effort.* In a parallel round every model reads the same text, so the models mostly report the same issues: the merged groups say "raised by 3 models", but you paid three times for one finding. In a cascade each model reviews what the earlier rounds left, so a cheap model clears out the obvious issues first and each stronger model spends its findings on what remains.
- *The edits get reviewed too.* Applying findings writes new text (and, for `--improve-code`, new code), and that is where new mistakes come from: a lock that a nested call takes a second time and deadlocks, an acceptance test that passes without the behaviour it guards, a function the plan calls but never defines, a line the edit forgot to update. A parallel round applies once and nobody reviews the result. In a cascade every later round does, and it also flags fixes that landed only halfway or arrived without tests.
- *No conflicting suggestions.* Each round's suggestions are written against the current text, so they cannot conflict the way merged parallel suggestions can.
- *Rejected ideas stay rejected.* From the second round on, the reviewers, the validators and the judge are told what earlier rounds rejected, so they do not raise it again without a new argument, and what earlier rounds changed, so they check those edits rather than take them for your intent. A finding that repeats an earlier rejection is marked `re-raises R1-3` in the report (R1-3 is round 1's third rejected entry), and it is rejected unless it answers the recorded reason.

**When to use it.** In our test runs (see *What we measured*, below), the same three models on the same input did better as a cascade than as a parallel round, at about 1.7 times the time and host tokens:

- *Plans came out better.* A blind judge preferred the cascade's plan in all four runs, and the cascade left fewer errors in it: none of high severity, against two in the parallel runs (a lock that deadlocks on every call, and a sale step that silently does nothing). The apply step wrote both errors, and in a parallel round nobody reviews the apply.
- *Code came out as correct, and better tested.* The final code passed about as many checks either way. The difference was the tests: every cascade fix came with tests, and those tests caught most of the defects we put back into the code. Half of the parallel runs wrote no tests.
- *Neither mode found more of what was already wrong.* Both found about as many of the issues already in the original. The gap was no bigger than the spread between two runs of the same model. In a parallel round, the second and third models mostly repeated the first.

So the cascade's gain is that each round reviews the previous round's edits. Choose by how much an unreviewed edit would cost you:

- **Use a cascade** for a plan that someone else, or another model, will implement as written, and for a code fix you want to keep, guarded by tests that catch a regression.
- **Use a parallel round** for a quick pass, a draft you will rework anyway, or a small change you will check yourself. It finds as many of the existing issues in about 60% of the time.
- **Expect longer plans.** The cascade's plans came out about 40% longer, because each round can add to them. `--scope-policy preserve` (below) keeps that in check.

**What we measured.** We compared the two on a small plan (24 lines) and on a small code change against it (21 lines), both with planted defects. Each mode ran four times, from identical inputs, with the same three models in the same order (`claude-code:haiku`, `codex:gpt-6-luna:high`, `claude-code:sonnet`). The plans were compared blind by a separate judge model, and the final code was scored by hidden checks for the planted defects.

| | Parallel | Cascade |
| --- | --- | --- |
| Wall time, plan / code (mean) | 6.6 / 5.0 min | 11.2 / 8.1 min |
| Host model tokens, plan run | 1x | about 1.7x |
| Issues found that were already in the original plan | 13.0 per run | 13.25 per run |
| Issues found in text an earlier round added | none (never reviewed) | 6.75 per run |
| Findings of the 2nd and 3rd model that repeated the 1st model's, plan | 60 of 62 | none |
| Final plan preferred by the blind judge | 0 of 4 | 4 of 4 |
| Errors left in the final plan, high / medium severity (4 runs) | 2 / 7 | 0 / 4 |
| Final plan length (mean) | 126 lines | 177 lines |
| Final code: planted-defect checks passed (4 runs) | 33 of 40 | 35 of 40 |
| Runs whose fix came with tests | 2 of 4 | 4 of 4 |
| Re-introduced defects the run's own tests caught | 5 of 28 | 21 of 30 |

On the issues already in the original, the two modes differed in only one case: whether a hold expires at exactly 15 minutes. Every cascade run fixed it and no parallel run did. The judge also preferred the longer plan every time, which may favour the cascade, but the error counts and the tests point the same way. This was one small plan and one small change, so treat the numbers as a sense of the trade-off, not a benchmark.

- **The order is your model list's order: put the cheapest model first.** The rounds follow the review's model list exactly as it resolves without `--cascade` ([Models](#review-and-apply-in-one-command), above): `--models` as you typed it, else the `--quick` list, else the review's configured defaults, in their configured order. Nothing reorders it. With no model flag, the run names the list it picked once, on an `[AUTO_RESOLVED] models: …` line. One model makes a one-round cascade (`NOTE: --cascade with one model is a single round`).
- **Cost and time.** A cascade of n models makes n one-model reviews, then n validations, n judge passes and n applies, one after another. The reviews cost about what one parallel round with the same models costs, but validation and the apply are paid n times, and the run takes the sum of the rounds instead of the time of the slowest model. Suggestions an earlier round already rejected come back less often, which saves validating and rejecting them again. From the second round on, each prompt carries a few thousand characters of earlier results.
- **Use `--scope-policy preserve` with it.** Extensions compound over rounds: under the default, each round's new review is `expand`, so every model may add its own. A policy you type when the cascade starts applies to every round. Typed on a later run of an unfinished cascade, it applies to the rounds not started yet, and the round under way keeps its own ([Scope policy](#scope-policy)). Findings `preserve` holds back are counted in the table's "held back" column, and later rounds are told they were held back, not rejected. Only the last round's can still be applied: once the cascade is finished, run the mode again without `--cascade`, with `--scope-policy expand`. Each round replaces the review on record, so earlier rounds' held-back findings cannot be released that way.
- **What `--improve-code` reviews.** `--files`, `--files-from <file>` and `--base-ref` are fixed when the cascade starts, and every round reviews the same files, so a fix an earlier round made outside that list is not reviewed again. With none of them, each round reviews review-code's default scope as usual, which takes in the files earlier rounds' fixes touched. `--files-from -` is refused with `--cascade`: pass the list in a file.
- **No consolidation pass.** `--improve-plan --cascade` skips it (its plan line reads `skip (cascade)`); the per-round table is the summary. You can still run it by hand on the last round's review.
- **What it writes.** Each round's phases write their usual folders (`review-plan/` and `apply-suggestions/`, or `code-review/` and `apply-fixes/`), and the next round replaces them as a `--force` round does, so every finished round is also copied aside, under `{plan}/improve-plan/cascades/<time>-<id>/` (`improve-code/` for `--improve-code`):
  - `summary.md`: a table of what each model found and what got applied, one row per round: findings, valid, invalid, needs decision, re-raises, applied, no-op, declined, held back (by `--scope-policy preserve`), the judge's decisions, time (from the round's start to its finish, retries included) and the reviewer's recorded cost (`not recorded` when the CLI reported none);
  - `ledger.md`: everything earlier rounds rejected or held back, and everything they changed;
  - `record.json`: the cascade's state; `start.md`: the plan when the cascade started (`--improve-plan`); `files.txt`: your `--files` list, when you typed one (`--improve-code`);
  - `stage-1/`, `stage-2/`, …: each round's inputs, frozen when it started, and once it is done, copies of its review and apply folders, plus the plan after the round and the round's diff (`--improve-plan`) or the files the round changed (`--improve-code`).

  `{plan}/improve-plan/cascade-current.json` names the cascade under way. `{plan}/apply-suggestions/backup.md` is the plan before the last round's apply, as after a `--force` round; `start.md` is the plan before the first.
- **Resuming.** Run the same command again, or the command a stopped run prints (it carries `--cascade`): the cascade goes on at the round where it stopped and never redoes a finished round. While a cascade is unfinished, running the mode again continues it even without `--cascade`. A different `--models`, `--quick`, `--files`, `--files-from` or `--base-ref` stops at exit 3 with `[CASCADE_CONFLICT]`, naming both values and the two ways out: leave the flag off to continue, or start a new cascade with `--force --cascade`. A continuation (`--cascade --resume` with no `--models` or `--quick`) typed when no cascade is left to resume, for example after `--force` alone abandoned it, stops the same way instead of starting a new one.
- **A model that returns nothing.** When a round's one model produces no review result, a round in the middle is skipped (`[CASCADE_STAGE_SKIPPED]`) and the cascade goes on with the next model. The last round is never skipped: the run stops at exit 3 with the earlier rounds applied, and running the command again starts the last round over. Any other stop (an apply batch that failed, a decision owed) stops the cascade at that round, and running the command again continues it.
- **Running a finished cascade again** prints `[ALREADY_COMPLETE] cascade finished at <time> (<n> stages); --force --cascade starts a new one` and exits 2, as every composite does when there is nothing left to do. That includes the continuation a stopped run printed (`--cascade --resume`, with no model list). The command with a different `--models`, `--quick`, `--files`, `--files-from` or `--base-ref` typed starts a new cascade. **`--force --cascade` starts another pass** over the result, abandoning an unfinished cascade first. **`--force` without `--cascade` abandons an unfinished cascade** (a `NOTE:` line says so) and runs one ordinary round with every model in parallel.
- **Status.** While the cascade is unfinished, `--status` shows `cascade: stage <k> of <n> (<model>)` and a `resume with:` command that carries `--cascade`: under a background (`--detach`) run of the mode, or, when no background run is recorded (Claude Code runs the steps itself, or a run stopped in the foreground), on an entry of its own, `improve-plan  no run record` (`improve-code` for `--improve-code`). Its suggested next step is the cascade's continuation, not the next workflow phase. The next command a run prints, and `util next-command`, continue the cascade too.

The run's report opens with the per-round table from `summary.md`, then the last round's review and apply, and lists the cascade's files. One command runs every round, on every host: Claude Code runs them one after another in one turn, and under `--exec-subagents` (the Codex and OpenCode default) the run does the same, detached or not.

**Not covered.** `--improve-commits` does not take `--cascade`: each round's fixes would move the range under review. There is no `--full --cascade`, no cascade of the review modes alone, no round with several models, no checks or tests between rounds, no rule that repeats the last round until it finds nothing, and no setting for the order: `--models` sets it.

## Options

These flags combine with any mode and are **position-independent** - they may appear before or after the plan path (and, for `--ask`, around the question). The one exception is a commit mode's range, which goes right after the mode word, before any flag ([Commit reviews](commit-reviews.md)). Not every flag applies to every mode.

| Flag | Effect |
| --- | --- |
| `--models <provider:model> ...` | Use exactly these models (variadic), overriding YAML defaults. Each must exist in `providers.yaml`. |
| `--quick` | Use the smaller `quick_models` subset for a faster  run. In `--improve-plan` / `--improve-code` / `--improve-commits` it goes to the review step. |
| `--interactive` | Force two-step model selection (provider, then models), ignoring YAML defaults. |
| `--no-confirm` | Skip confirmation prompts - for unattended/silent runs. |
| `--dry-run` | Show what would happen (tasks/batches) without making changes. Applies to `--implement` and the apply modes, `--apply-commit-fixes` included. |
| `--dry-run-subagents` | Preview what would be dispatched, without spending: every subagent command is rendered and none starts. A review mode prints each reviewer as it would run (provider, model, effort, sandbox, command line) and calls no model; `--improve-plan` / `--improve-code` / `--improve-commits` and `--full` preview only the first step that would run, and no step after it (`--full` walks past a phase with nothing to do, such as a review already completed, as a real run does, and previews the next). Nothing is recorded; the preview is also written to `{phase}/subagent_dry_run.json`. |
| `--agent-decide` | In the apply modes, let a judge agent decide every `needs-human-decision` finding instead of prompting you (see [below](#letting-the-agent-decide-ambiguous-findings)). Implied by `--yes`, which the Codex and OpenCode shims bake in. `--claude-decide` / `--let-claude-decide` are accepted legacy spellings. |
| `--force` | **Meaning depends on mode.** In the fan-out review/ask modes (`--review-plan`, `--review-tasks`, `--review-code`, `--review-commits`, `--ask`), resume an interrupted run: keep already-completed per-model results, run only the missing models, and bypass the completed-phase / partial-completion guards. In the apply modes, it's an alias for `--yes` (confirm bulk approval). In `--improve-plan` / `--improve-code` / `--improve-commits`, it starts a new review round, then applies that round's findings ([above](#review-and-apply-in-one-command)). |
| `--cascade` | Taken by `--improve-plan` and `--improve-code` only. One review-then-apply round per model, in your model list's order (cheapest first), each model reviewing what the previous round left. Running the command again continues an unfinished cascade; `--force --cascade` starts a new one ([above](#cascade-one-model-at-a-time)). |
| `--rerun-all` | In the fan-out review/ask modes, re-run every model from scratch, discarding any existing per-model results. Combine with `--force` (`--force --rerun-all`) for a fresh full re-run of an already-completed phase. |
| `--focus "<text>"` | Steer a commit review toward a priority (not a filter). Taken by the three commit modes only. Recorded with the commit brief: a later run of the same range keeps it when you leave it out, and a different value (or `--focus ""`, which clears it) restarts the review round. `--apply-commit-fixes` keeps the value the review on record used ([Commit reviews](commit-reviews.md)). |
| `--context-plan <path>` | Give a commit review a plan as read-only context, and check the commits against it too. Taken by the three commit modes only. Copied into the review's folder when the review starts and recorded like `--focus`; type it again to review against an edited file. `--apply-commit-fixes` keeps the context plan the review on record used, and never edits it. |
| `--scope-policy expand\|preserve` | Whether the run may carry changes beyond the agreed scope. `expand`, the default for a new run, keeps today's behaviour and only tags each finding's scope; `preserve` asks the reviewers for in-scope corrections only, and validation and the apply hold every extension back. Taken by the review, apply and commit modes, `--generate-tasks`, `--implement`, the `--improve-*` modes and `--full`, not by `--ask`, `--status` or `--init`. Left out, a continuation keeps the policy on record ([below](#scope-policy)). |

The apply and implement modes accept additional approval flags (`--yes`, `--approve-all`, `--resume`, `--task`, and more); run `multi_llm.py <mode> --help` for the authoritative list, or read the per-mode files in `skills/multi-llm/instructions/`.

## Letting the agent decide ambiguous findings

When the models disagree or a suggestion is borderline, validation marks it `needs-human-decision`. In the apply modes (`--apply-suggestions`, `--apply-code-fixes`, `--apply-commit-fixes`, `--apply-task-suggestions`) Claude Code prompts you to approve or skip each one through its structured `AskUserQuestion` tool - it is the only host that has one, so on Codex and OpenCode the baked-in `--yes` resolves these without asking ([Hosts](hosts.md)). Every such prompt also offers a **Let the agent decide** option, per item or for a whole batch, that hands just that judgment to a judge agent instead of you.

To skip the prompts entirely and have a judge agent decide **every** `needs-human-decision` finding up front, pass `--agent-decide` - which is what the baked-in `--yes` already does on Codex and OpenCode. The judge is not necessarily Claude: on a host with native subagents (Claude Code) it is the host's own agent, and under `--exec-subagents` (the Codex and OpenCode default) it is the model your session is running on, e.g. `codex:<chat model>` on Codex. A `driver.subagent_runner` you set explicitly (in `<git-root>/.multi-llm/providers.yaml` or the file `MULTI_LLM_PROVIDERS_CONFIG` names) overrides that, and the host's floor runner (`codex:gpt-6-astra` on Codex) is used only when neither is available. The summary names it on a `judge:` line. `--improve-plan`, `--improve-code` and `--improve-commits` imply `--agent-decide` (with `--yes` and `--no-confirm`) for their apply step on every host.

```text
/multi-llm:multi-llm --apply-code-fixes plans/my-feature.md --agent-decide --no-confirm
```

Unlike the blanket approve/skip flags, "Let the agent decide" is a **per-item** judgment that **salvages** partially-valid findings - it trims each to its worthwhile core and applies that, skipping only findings with nothing worth keeping. The generated HTML report badges every finding **Approved / Salvaged / Skipped**. Add `--no-confirm` (as above) for a fully unattended run.

## Scope policy

`--scope-policy expand|preserve` decides whether a run may carry changes that go beyond the **agreed scope**: the goals, requirements and limits it was given. Whichever policy a run uses, every report tags every finding with its scope.

**`expand`, the default, changes no decision.** A new run without the flag is `expand`, and it reviews, validates, judges, selects and applies exactly as before. Reviewers still report the features, improvements, refactors and cleanup they find, and a warranted extension stays eligible under the usual validity, priority and selection rules. The tags are informational: nothing is suppressed, rejected, deferred or asked because a finding reads `Scope extension` or `Scope unclear`. `expand` is permission to extend, not an instruction to: it never adds work, and it never overrides a limit the plan states, such as "do not change the API".

**`preserve` adds three checks, each independent of the one before it.**

1. **Reviewers are asked not to propose extensions.** They report only findings whose fix stays within the agreed outcome - no new features, unrelated refactors or optional cleanup, however beneficial - and keep reporting missing or incomplete requirements, the bugs and security regressions the work introduced, and existing scope creep: removing something the plan never asked for is within scope. A requirement that cannot be met without going beyond the scope is reported as a blocker naming the decision you need to make, neither hidden nor quietly expanded.
2. **Validators reject any extension that got through.** Each group's whole proposed fix is classified again against the agreed scope, independently of the reviewers' own tags, and an extension or an unclear fix is rejected for this run. Correctness stays separate: a real finding whose fix extends the scope stays `valid` and stays in the report, marked `Deferred: scope preservation requested`; it is never turned into a false positive. The review prints one `[SCOPE_POLICY] preserve: …` line counting them.
3. **The apply defers everything not validated as within scope.** Just before batching, the apply checks each item again: only a finding validated `Within scope` against this review round's scope, and unchanged since, is applied. Everything else is **deferred** - kept apart from applied, skipped and invalid findings, named on `[SCOPE_DEFERRED]` lines and in `summary.md`'s "Deferred by scope policy" section - and the phase completes with its eligible work done. No flag lets a deferred item through: not `--yes`, `--no-confirm`, `--agent-decide`, a bulk approval, `--force` or `--accept-stale-consolidation`; `--min-priority` still applies on top. Writers are told to decline an item that turns out to need an extension. When an agreed requirement itself cannot be met inside the scope, the apply stops at exit 3 with `[SCOPE_BLOCKER]`, and the phases that depend on it refuse to start until you decide: amend the plan and review it again, or apply with `--scope-policy expand`.

Under `preserve`, `--generate-tasks` writes tasks only for what the plan asks for, and `--implement` tells each task's model to add no outcome beyond its task. The review modes stay read-only under either policy; the `--improve-*` modes edit their target unattended, as always.

**What the scope is measured against.** The plan's own goals, requirements and limits, for a plan review. For a task review, the plan the tasks came from: a task cannot bring a feature into scope by listing it. For a code review, the accepted plan and the decisions recorded in it; the diff and the task list show what was built, not what was asked for, so drift in the implementation cannot authorize itself, while fixing a regression the change introduced is within scope. For a commit review, the commit brief ([Commit reviews](commit-reviews.md)). A review round records its scope once, by hash, in `{plan}/state.json`, and keeps it: a plan edited while the round is under way does not move it (validation warns `[SCOPE_BASELINE_MOVED]`), a revalidation after an apply judges correctness against the current plan but scope against the round's, and the next round reads the edited plan. A limit that lives only in your conversation reaches no model: write it into the plan, or the `--context-plan` of a commit review.

**The tags in the reports.** Every `report.md` and `report.html` - plan, tasks, code and commit reviews, and the consolidated reports - uses three labels:

| Tag | The proposed fix |
| --- | --- |
| `Within scope` | completes, corrects or clarifies what the scope already asks for, even when that is substantial work: a prerequisite in another file, a fix for a regression the work introduced, the removal of existing scope creep |
| `Scope extension` | adds a capability, an outcome or unrelated cleanup the scope does not ask for, or changes an agreed outcome, even when it is worthwhile |
| `Scope unclear` | the scope does not settle it (missing or vague intent), the fixes on offer differ in scope and none was chosen, or there is no valid classification on record |

A tag describes the proposed fix, not whether the finding is real or how important it is, and it is shown apart from validity, selection and outcome. A finding's own `**Scope:**` line is its reviewer's tag, marked provisional; the group's `**Group scope:**` line is the validated one, and the one the apply reads. Each report's header carries a `**Scope policy:**` line, `**Scope tags:**` counts (per distinct group, not per finding) and a legend. Under `preserve` each group also shows a `**Scope outcome:**`, `Eligible under scope preservation` or `Deferred: scope preservation requested`, and the policy line says that additional scope was not solicited, so its extension count covers only the findings the reviewers returned. A consolidated group reads `Scope extension` when any group under it is one, else `Scope unclear` when any is unclear, and each group under it is still decided on its own. `report.html` shows the same tags as badges.

**Examples.** These are the skill's arguments: start each line with `/multi-llm:multi-llm` in Claude Code, `$multi-llm:multi-llm` in Codex, or `/multi-llm` in OpenCode. Each standalone phase assumes the phases before it have run; the `--full` line is the alternative to running them one by one.

```text
# New plan review: the default behaviour, scope tags are informational
--review-plan plans/feature.md

# Plan review and application with a preserved scope
--review-plan plans/feature.md --scope-policy preserve
--apply-suggestions plans/feature.md --scope-policy preserve

# Task generation, review, correction, and implementation
--generate-tasks plans/feature.md --scope-policy preserve
--review-tasks plans/feature.md --scope-policy preserve
--apply-task-suggestions plans/feature.md --scope-policy preserve
--implement plans/feature.md --scope-policy preserve

# Code review and application in a single unattended command
--improve-code plans/feature.md --scope-policy preserve

# Alternative: preserve scope throughout the full unattended pipeline
--full plans/feature.md --scope-policy preserve --yes

# Commit review and application, without requiring a plan
--improve-commits main..HEAD --scope-policy preserve

# Explicitly allow expansion when applying remaining code-review findings
--apply-code-fixes plans/feature.md --scope-policy expand
```

The `--apply-*` lines would keep their review's `preserve` without the flag; it is spelled out so each line says what it does.

**Which policy a run uses.**

- **A typed `--scope-policy` wins.** An empty or unknown value stops at exit 3 before anything runs. The policy is never a question: under `--yes`, which the Codex and OpenCode entry points pass, one nobody typed is recorded as `[AUTO_RESOLVED] scope-policy: …`.
- **Left out on a new run, it is `expand`.** A new review round, `--generate-tasks` and `--implement` each start their own run, so to preserve the scope through the standalone phases type the flag on each one, as above. The next-phase command a run prints never adds it.
- **Left out on a continuation, it is the one on record.** A review's `--reaggregate` keeps its round's policy; an apply, and any revalidation it runs, keep the review round's (or the apply's own typed override, below); a re-run or `--resume` of a stopped `--generate-tasks`, `--implement`, `--improve-*` or `--full` run keeps the value typed when it started. Repeating the same value is harmless. A review's `--force` resume of the missing models keeps the policy the results it reuses were requested under, and says so in one `Scope policy notice:` line; `--force --rerun-all` starts a fresh round instead, under `expand` or the policy you type.
- **`--improve-*` and `--full` resolve it once** and pass that one value to every step, including the apply `--full` runs itself when a prerequisite is missing. With none typed they pass none, and each step resolves on its own as above. `--force` starts them over, without the earlier choice.

**Changing it after the review.**

- **On an apply, a typed policy overrides the review's** and is kept for that apply's later runs. It changes only which of the findings the review returned may be applied: `--apply-code-fixes plans/feature.md --scope-policy expand` after a `preserve` review applies what that review deferred, and never applies a landed batch twice. It cannot recover suggestions the `preserve` reviewers were never asked for, nor change their tags; the reports name both policies. A completed `--improve-*` run typed again with a different policy runs only its apply again, when that apply deferred something. Changing the policy part-way through an apply rebuilds only the batches that never started; a batch of the earlier run that did not finish stops it at exit 3, naming the ways out.
- **A review round keeps the policy its reviewers were asked under.** A different one typed for a round already under way stops at exit 3 with `[SCOPE_POLICY_CONFLICT]`, and running a completed review again does not start a new round. To ask the reviewers again - for the broader suggestions a `preserve` round never solicited, say - start a new round with the existing controls and the policy you want: `--review-code plans/feature.md --force --rerun-all --scope-policy expand`, or `--improve-code plans/feature.md --force --scope-policy expand`.

**Old and unclear findings.**

- **A review from before `--scope-policy` existed reads as `expand`.** Its findings show `Scope unclear` and apply exactly as before, with no migration and no model call.
- **Under `preserve`, a finding with no valid classification is deferred,** whether it comes from such an old review or from a validation reply that carried none, with the way forward: run a fresh review, or apply with `--scope-policy expand`.
- **A finding changed after validation reads `Scope unclear`** - its description edited in `report.md` or `report.html`, one of its members skipped, or its fix trimmed by the judge - and its tag is marked possibly stale. Under `expand` that changes nothing; under `preserve` the finding is deferred.
- **Missing or vague intent gives `Scope unclear`, not an invented requirement.** Under `preserve` that defers the fix, so state the limits you care about in the plan.

---

Back to the [README](../README.md) · [Commit reviews](commit-reviews.md) · [Per-project configuration](configuration.md)
